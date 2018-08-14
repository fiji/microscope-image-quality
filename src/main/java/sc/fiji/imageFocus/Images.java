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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

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

	/** TODO */
	public static <T> RandomAccessibleInterval<T> tile(
		final RandomAccessibleInterval<T> image, final long xTileCount,
		final long yTileCount, final long xTileSize, final long yTileSize)
	{
		final long[] min = new long[2];
		final long[] max = new long[2];
		final long width = image.dimension(0);
		final long height = image.dimension(1);
		final ArrayList<RandomAccessibleInterval<T>> strips = new ArrayList<>();
		final ArrayList<RandomAccessibleInterval<T>> tiles = new ArrayList<>();
		for (long ty = 0; ty < yTileCount; ty++) {
			tiles.clear();
			final long y = offset(ty, yTileCount, yTileSize, height);
			for (long tx = 0; tx < xTileCount; tx++) {
				final long x = offset(tx, xTileCount, xTileSize, width);
				min[0] = x;
				min[1] = y;
				max[0] = x + xTileSize - 1;
				max[1] = y + yTileSize - 1;
				tiles.add(Views.interval(image, min, max));
			}
			strips.add(concatenateX(tiles));
		}
		return Views.concatenate(1, strips);
	}

	/** TODO */
	public static long offset(final long i, final long total,
		final long tileSize, final long dimSize)
	{
		// The total amount of space needing to be distributed between tiles.
		final long space = dimSize - tileSize * total;
		// Return the tile offset plus the gap offset, minimizing rounding error.
		return i * tileSize + (i + 1) * space / (total + 1);
	}

	/** Work around a limitation when concatenating dims before the last one. */
	private static <T> RandomAccessibleInterval<T> concatenateX(
		final ArrayList<RandomAccessibleInterval<T>> tiles)
	{
		final List<IntervalView<T>> flippedTiles = tiles.stream() //
			.map(tile -> Views.permute(tile, 0, 1)) //
			.collect(Collectors.toList());
		final RandomAccessibleInterval<T> concatenated = //
			Views.concatenate(1, flippedTiles);
		return Views.permute(concatenated, 1, 0);
	}
}
