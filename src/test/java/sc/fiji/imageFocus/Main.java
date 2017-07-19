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

import net.imagej.ImageJ;

/**
 * Test drive for the {@link MicroscopeImageFocusQualityClassifier} command.
 *
 * @author Curtis Rueden
 */
public final class Main {

	public static void main(final String[] args) {
		final ImageJ ij = new ImageJ();
		ij.launch(args);
		ij.command().run(MicroscopeImageFocusQualityClassifier.class, true);
	}
}
