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

import ij.ImagePlus;
import ij.gui.Line;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.gui.TextRoi;

import java.awt.Color;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imagej.display.ColorTables;
import net.imagej.tensorflow.TensorFlowService;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.display.ColorTable8;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.Initializable;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.Previewable;
import org.scijava.io.http.HTTPLocation;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.NumberWidget;
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
 * This command optionally produces a multi-channel image of probability values
 * where each channel corresponds to one focus class, and optionally adds an
 * overlay annotation to the input image to visualize the most likely focus
 * class of each region.
 * </p>
 */
@Plugin(type = Command.class,
	menuPath = "Plugins>Classification>Microscope Image Focus Quality")
public class MicroscopeImageFocusQualityClassifier<T extends RealType<T>>
	implements Command, Initializable, Previewable
{

	private static final String MODEL_URL =
		"https://storage.googleapis.com/microscope-image-quality/static/model/fiji/microscope-image-quality-model.zip";

	private static final String MODEL_NAME = "microscope-image-quality";

	// Same as the tag used in export_saved_model in the Python code.
	private static final String MODEL_TAG = "inference";

	// Same as
	// tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
	// in Python. Perhaps this should be an exported constant in TensorFlow's Java
	// API.
	private static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY =
		"serving_default";

	private static final int TILE_SIZE = 84;

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private DatasetService datasetService;

	@Parameter
	private LogService log;

	@Parameter(label = "Microscope Image")
	private Img<T> originalImage;

	/**
	 * ImageJ 1.x version of the image to process.
	 * <p>
	 * Only used for overlaying patches.
	 * </p>
	 */
	@Parameter(required = false)
	private ImagePlus originalImagePlus;

	private Overlay originalOverlay;

	@Parameter(label = "Number of tiles in X", persist = false,
		callback = "refreshTilePreview", min = "1",
		description = "<html>The number of tiles to process in the X direction. " +
			"The smaller this value, the less<br>of the image will be covered " +
			"horizontally, but the faster the processing will be.")
	private long tileCountX = 1;

	@Parameter(label = "Number of tiles in Y", persist = false,
		callback = "refreshTilePreview", min = "1",
		description = "<html>The number of tiles to process in the Y direction. " +
			"The smaller this value, the less<br>of the image will be covered " +
			"vertically, but the faster the processing will be.")
	private long tileCountY = 1;

	@Parameter(label = "Generate probability image",
		description = "<html>When checked, a multi-channel image will be created " +
			"with one channel per focus level,<br>and each value corresponding " +
			"to the probability of that sample being at that focus level.")
	private boolean createProbabilityImage = true;

	@Parameter(label = "Overlay probability patches",
		description = "<html>When checked, each classified region of the image " +
			"will be overlaid with a color<br>whose hue denotes the most likely " +
			"focus level and whose brightness denotes<br>the confidence " +
			"(i.e., probability) of the region being at that level.")
	private boolean overlayPatches = true;

	@Parameter(label = "Show patches as solid rectangles",
		description = "<html>When checked, overlaid probability patches will be " +
			"filled semi-transparent<br>and solid; when unchecked, they will be " +
			"drawn as hollow boundary boxes.")
	private boolean solidPatches;

	@Parameter(label = "Displayed patch border width", //
		min = "1", max = "10", style = NumberWidget.SCROLL_BAR_STYLE,
		description = "When drawing probability patches as boundary boxes, " +
			"this option controls the box thickness.")
	private int borderWidth = 4;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset probDataset;

	@Override
	public void run() {
		try {
			validateFormat(originalImage);
			final RandomAccessibleInterval<T> tiledImage = //
				Images.tile(originalImage, tileCountX, tileCountY, TILE_SIZE, TILE_SIZE);
			final RandomAccessibleInterval<FloatType> normalizedImage = //
				Images.normalize(tiledImage);

			final long loadModelStart = System.nanoTime();
			final HTTPLocation source = new HTTPLocation(MODEL_URL);
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
			try (final Tensor<?> inputTensor = Tensors.tensor(normalizedImage)) {
				// Run the model.
				final long runModelStart = System.nanoTime();
				final List<Tensor<?>> fetches = model.session().runner() //
					.feed(opName(sig.getInputsOrThrow("input")), inputTensor) //
					.fetch(opName(sig.getOutputsOrThrow("probabilities"))) //
					.fetch(opName(sig.getOutputsOrThrow("patches"))) //
					.run();
				final long runModelEnd = System.nanoTime();
				log.info(String.format("Ran image through model in %dms", //
					(runModelEnd - runModelStart) / 1000000));

				// Process the results.
				try (final Tensor<?> probabilities = fetches.get(0);
						final Tensor<?> patches = fetches.get(1))
				{
					processPatches(probabilities, patches);
				}
			}
		}
		catch (final Exception exc) {
			// Use the LogService to report the error.
			log.error(exc);
		}
	}

	@Override
	public void initialize() {
		if (originalImage == null) return;
		tileCountX = Math.max(1, originalImage.dimension(0) / TILE_SIZE);
		tileCountY = Math.max(1, originalImage.dimension(1) / TILE_SIZE);
		if (originalImagePlus != null) {
			originalOverlay = originalImagePlus.getOverlay();
		}
		refreshTilePreview();
	}

	@Override
	public void preview() {
		// NB: No action needed.
	}

	@Override
	public void cancel() {
		if (originalImagePlus != null && originalOverlay != null) {
			originalImagePlus.setOverlay(originalOverlay);
		}
	}

	/** Callback method for {@link #tileCountX} and {@link #tileCountY}. */
	private void refreshTilePreview() {
		if (originalImagePlus == null) return;

		final int tileOpacity = 64;
		final Color evenColor = new Color(100, 255, 255, tileOpacity);
		final Color oddColor = new Color(255, 255, 100, tileOpacity);

		final long w = originalImage.dimension(0);
		final long h = originalImage.dimension(1);

		final Overlay overlay = new Overlay();
		for (long y = 0; y < tileCountY; y++) {
			final long offsetY = Images.offset(y, tileCountY, TILE_SIZE, h);
			for (long x = 0; x < tileCountX; x++) {
				final long offsetX = Images.offset(x, tileCountX, TILE_SIZE, w);
				final Roi tile = new Roi(offsetX, offsetY, TILE_SIZE, TILE_SIZE);
				tile.setFillColor((x + y) % 2 == 0 ? evenColor : oddColor);
				overlay.add(tile);
			}
		}
		originalImagePlus.setOverlay(overlay);
	}

	private void validateFormat(final Img<T> image) throws IOException {
		final int ndims = image.numDimensions();
		if (ndims != 2) {
			final long[] dims = new long[ndims];
			image.dimensions(dims);
			throw new IOException("Can only process 2D images, not an image with " +
				ndims + " dimensions (" + Arrays.toString(dims) + ")");
		}
		if (!(image.firstElement() instanceof UnsignedShortType)) {
			throw new IOException("Can only process uint16 images. " +
				"Please convert your image first via Image > Type > 16-bit.");
		}
	}

	private void processPatches(final Tensor<?> probabilities,
		final Tensor<?> patches)
	{
		// Extract probability values.
		final long[] probShape = probabilities.shape();
		log.debug("Probabilities shape: " + Arrays.toString(probShape));
		final int probPatchCount = (int) probShape[0];
		final int classCount = (int) probShape[1];
		final float[][] probValues = new float[probPatchCount][classCount];
		probabilities.copyTo(probValues);

		// Extract and validate patch layout.
		final long[] patchShape = patches.shape();
		log.debug("Patches shape: " + Arrays.toString(patchShape));
		assert patchShape.length == 4;
		final int patchCount = (int) patchShape[0];
		assert patchCount == probPatchCount;
		final int patchHeight = (int) patchShape[1];
		final int patchWidth = (int) patchShape[2];
		assert patchShape[3] == 1;

		// Dump probabilities to the log.
		for (int i = 0; i < probShape[0]; ++i) {
			log.info(String.format("Patch %02d probabilities: %s", i, //
				Arrays.toString(probValues[i])));
		}

		// Synthesize matched-size image with computed probabilities.
		if (createProbabilityImage) {
			createProbabilityImage(classCount, probValues, patchHeight, patchWidth);
		}

		// Add ImageJ 1.x overlay to the active image.
		if (overlayPatches && originalImagePlus != null) {
			addOverlay(probValues, patchWidth, patchHeight);
		}
	}

	private void createProbabilityImage(final int classCount,
		final float[][] probValues, final int patchHeight, final int patchWidth)
	{
		// Create probability image.
		final long width = originalImage.dimension(0);
		final long height = originalImage.dimension(1);
		final long[] dims = { width, height, classCount };
		final AxisType[] axes = { Axes.X, Axes.Y, Axes.CHANNEL };
		final FloatType type = new FloatType();
		probDataset = datasetService.create(type, dims, "Probabilities", axes,
			false);

		// Set the probability image to normalized grayscale.
		probDataset.initializeColorTables(classCount);
		for (int c = 0; c < classCount; c++) {
			probDataset.setColorTable(ColorTables.GRAYS, c);
			probDataset.setChannelMinimum(c, 0);
			probDataset.setChannelMaximum(c, 1);
		}

		final ImgPlus<FloatType> probImg = probDataset.typedImg(type);

		// Cover the probability image with NaNs.
		// Real values will be written only to tile-covered areas.
		for (final FloatType sample : probImg) {
			sample.set(Float.NaN);
		}

		// Populate the probability image's sample values.
		final RandomAccess<FloatType> access = probImg.randomAccess();
		for (int t = 0; t < probValues.length; t++) {
			for (int c = 0; c < probValues[t].length; c++) {
				// Compute tile coordinates from probability value index.
				final long tx = t % tileCountX;
				final long ty = t / tileCountX;

				// Compute offset of tile in original image.
				final long offsetX = Images.offset(tx, tileCountX, patchWidth, width);
				final long offsetY = Images.offset(ty, tileCountY, patchHeight, height);

				// Copy the current value to every sample within the tile.
				final float value = probValues[t][c];
				access.setPosition(c, 2);
				access.setPosition(offsetY, 1);
				for (int y = 0; y < TILE_SIZE; y++) {
					access.setPosition(offsetX, 0);
					for (int x = 0; x < TILE_SIZE; x++) {
						access.get().set(value);
						access.fwd(0);
					}
					access.fwd(1);
				}
			}
		}
	}

	private void addOverlay(final float[][] probValues, //
		final int patchWidth, final int patchHeight)
	{
		final int patchCount = probValues.length;
		final int classCount = probValues[0].length;

		final Overlay overlay = new Overlay();

		final int strokeWidth = solidPatches ? 0 : borderWidth;
		final ColorTable8 lut = ColorTables.SPECTRUM;
		final int lutMaxIndex = 172;

		final long width = originalImage.dimension(0);
		final long height = originalImage.dimension(1);

		for (int p = 0; p < patchCount; p++) {
			final long tx = p % tileCountX;
			final long ty = p / tileCountX;
			final long offsetX = Images.offset(tx, tileCountX, patchWidth, width) + strokeWidth / 2;
			final long offsetY = Images.offset(ty, tileCountY, patchHeight, height) + strokeWidth / 2;

			final Roi roi = new Roi(offsetX, offsetY, //
				patchWidth - strokeWidth, patchHeight - strokeWidth);
			final int classIndex = maxIndex(probValues[p]);
			final double confidence = probValues[p][classIndex];

			// NB: We scale to (0, 172) here instead of (0, 255) to avoid the high
			// indices looping from blue and purple back into red where we started.
			final int lutIndex = lutMaxIndex * classIndex / (classCount - 1);

			final int r = (int) (lut.get(0, lutIndex) * confidence);
			final int g = (int) (lut.get(1, lutIndex) * confidence);
			final int b = (int) (lut.get(2, lutIndex) * confidence);
			final int opacity = solidPatches ? 128 : 255;
			final Color color = new Color(r, g, b, opacity);
			if (solidPatches) {
				roi.setFillColor(color);
			}
			else {
				roi.setStrokeColor(color);
				roi.setStrokeWidth(strokeWidth);
			}
			overlay.add(roi);
		}

		// Add color bar with legend.

		final int barHeight = 24, barPad = 5;
		final int barY = originalImagePlus.getHeight() - barHeight - barPad;

		final TextRoi labelGood = new TextRoi(barPad, barY, "In focus");
		labelGood.setStrokeColor(Color.white);
		overlay.add(labelGood);

		final int barOffset = 2 * barPad + (int) labelGood.getBounds().getWidth();

		final TextRoi labelBad = new TextRoi(barOffset + lutMaxIndex + barPad,
			barY, "Out of focus");
		labelBad.setStrokeColor(Color.white);
		overlay.add(labelBad);

		for (int i = 0; i < lutMaxIndex; i++) {
			final int barX = barOffset + i;
			final Roi line = new Line(barX, barY, barX, barY + barHeight);
			final int r = lut.get(0, i);
			final int g = lut.get(1, i);
			final int b = lut.get(2, i);
			line.setStrokeColor(new Color(r, g, b));
			overlay.add(line);
		}

		originalImagePlus.setOverlay(overlay);
	}

	private static int maxIndex(final float[] values) {
		float max = values[0];
		int index = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > max) {
				max = values[i];
				index = i;
			}
		}
		return index;
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
}
