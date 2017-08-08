/*-
 * #%L
 * ImageJ plugin to analyze focus quality of microscope images.
 * %%
 * Copyright (C) 2017 Google, Inc. and Board of Regents of the
 * University of Wisconsin-Madison.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

package sc.fiji.imageFocus;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import net.imagej.Dataset;
import net.imagej.tensorflow.TensorFlowService;
import net.imagej.tensorflow.Tensors;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

/**
 * Command to apply the Microscopy image focus quality classifier model on an
 * input (16-bit, greyscale image).
 * <p>
 * The model has been trained on raw 16-bit microscope images with
 * integer-valued inputs in {@code [0, 65535]}, where the black level is usually
 * in {@code [0, ~1000]} and the typical brightness of cells is in
 * {@code [~1000, ~10,000]}.
 * </p>
 * <p>
 * This command will show both the input image and an annotated image marking
 * regions of the image with their focus quality.
 * <p>
 * This is a first draft, some TODOs:
 * <ul>
 * <li>Generate the annotated image from the model's output quality for each
 * tensor (and then set {@code annotatedImage} to the annotated image). For now,
 * the patch qualities are just dumped into the console log.
 * <li>Avoid loading the model from disk on every invocation of the command, as
 * that slows down the classifer (loading ~100MB of data into memory on every
 * invocation)
 * <li>Perhaps package the classification model with the plugin instead of
 * asking the user to locate the model on their local disk.
 * </ul>
 */
@Plugin(type = Command.class, menuPath = "Microscopy>Focus Quality")
public class MicroscopeImageFocusQualityClassifier<T extends RealType<T>>
	implements Command
{

	private static final String MODEL_NAME = "microscope-image-quality";

	// Same as the tag used in export_saved_model in the Python code.
	private static final String MODEL_TAG = "inference";

	// Same as
	// tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
	// in Python. Perhaps this should be an exported constant in TensorFlow's Java
	// API.
	private static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY =
		"serving_default";

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private LogService log;

	@Parameter(label = "Microscope Image")
	private Img<T> originalImage;

	@Parameter(label = "Focus Quality Model (zip archive)")
	private File modelArchive;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset annotatedImage;

	@Override
	public void run() {
		try {
			validateFormat(originalImage);

			final long loadModelStart = System.nanoTime();
			final FileLocation source = new FileLocation(modelArchive);
			final SavedModelBundle model = //
					tensorFlowService.loadModel(source, MODEL_NAME, MODEL_TAG);
			final long loadModelEnd = System.nanoTime();
			log.info(String.format(
				"Loaded microscope focus image quality model in %dms", (loadModelEnd -
					loadModelStart) / 1000000));

			// Extract names from the model signature.
			// The strings "input", "probabilities" and "patches" are meant to be
			// in sync with the model exporter (export_saved_model()) in Python.
			final SignatureDef sig = MetaGraphDef.parseFrom(model.metaGraphDef())
				.getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
			try (Tensor inputTensor = Tensors.tensor(originalImage, true)) {
				final long runModelStart = System.nanoTime();
				final List<Tensor> fetches = model.session().runner() //
					.feed(opName(sig.getInputsOrThrow("input")), inputTensor) //
					.fetch(opName(sig.getOutputsOrThrow("probabilities"))) //
					.fetch(opName(sig.getOutputsOrThrow("patches"))) //
					.run();
				final long runModelEnd = System.nanoTime();
				try (Tensor probabilities = fetches.get(0);
						Tensor patches = fetches.get(1))
				{
					processPatches(runModelStart, runModelEnd, probabilities, patches);
				}
			}
		}
		catch (final Exception exc) {
			// Use the LogService to report the error.
			log.error(exc);
		}
	}

	private void validateFormat(final Img<T> image)
		throws IOException
	{
		final int ndims = image.numDimensions();
		if (ndims != 2) {
			final long[] dims = new long[ndims];
			image.dimensions(dims);
			throw new IOException(
				"Can only process 2D images, not an image with " + ndims +
					" dimensions (" + Arrays.toString(dims) + ")");
		}
		if (!(image.firstElement() instanceof UnsignedShortType)) {
			throw new IOException("Can only process uint16 images. " +
				"Please convert your image first via Image > Type > 16-bit.");
		}
	}

	/**
	 * The SignatureDef inputs and outputs contain names of the form
	 * {@code <operation_name>:<output_index>}, where for this model,
	 * {@code <output_index>} is always 0. This function trims the {@code :0}
	 * suffix to get the operation name.
	 */
	private static String opName(final TensorInfo t) {
		final String n = t.getName();
		if (n.endsWith(":0")) {
			return n.substring(0, n.lastIndexOf(":0"));
		}
		return n;
	}

	private void processPatches(final long runModelStart, final long runModelEnd,
		Tensor probabilities, Tensor patches)
	{
		log.info(String.format("Ran image through model in %dms",
			(runModelEnd - runModelStart) / 1000000));
		log.info("Probabilities shape: " + Arrays.toString(probabilities
			.shape()));
		log.info("Patches shape: " + Arrays.toString(patches.shape()));

		final float[][] floatProbs = new float[(int) probabilities
			.shape()[0]][(int) probabilities.shape()[1]];
		probabilities.copyTo(floatProbs);
		for (int i = 0; i < probabilities.shape()[0]; ++i) {
			log.info(String.format("Patch %02d probabilities: %s", i,
				Arrays.toString(floatProbs[i])));
		}

		final int npatches = (int) patches.shape()[0];
		final int patchSide = (int) patches.shape()[1];
		assert patchSide == (int) patches.shape()[2]; // Square patches
		assert patches.shape()[3] == 1;

		// Log an error to force the console log to display
		// (otherwise the user will have to know to display the console
		// window).
		// Of course, this will go away once the annotate image is generated.
		log.error(
				"TODO: Display annotated image. Till then, see the beautiful log messages above");
	}
}
