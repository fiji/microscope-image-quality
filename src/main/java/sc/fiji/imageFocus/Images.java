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

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * Utility methods for manipulating images.
 *
 * @author Curtis Rueden
 */
public final class Images {

	private Images() {
		// NB: Prevent instantiation of utility class.
	}

	/** Normalizes an image to {@link FloatType} in range {@code [0, 1]}. */
	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType>
		normalize(final RandomAccessibleInterval<T> image)
	{
		final T sample = Util.getTypeFromInterval(image);
		final double min = sample.getMinValue();
		final double max = sample.getMaxValue();
		final Converter<T, FloatType> normalizer = //
			(in, out) -> out.setReal((in.getRealDouble() - min) / (max - min));
		return Converters.convert(image, normalizer, new FloatType());
	}
}
