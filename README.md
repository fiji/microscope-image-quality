[![](https://github.com/fiji/microscope-image-quality/actions/workflows/build-main.yml/badge.svg)](https://github.com/fiji/microscope-image-quality/actions/workflows/build-main.yml)

This project is an [ImageJ](https://imagej.net) plugin for Google's
[microscope image focus quality classifier](https://github.com/google/microscopeimagequality).

## Publication

Yang, S. J., Berndl, M., Ando, D. M., Barch, M., Narayanaswamy, A. ,
Christiansen, E., Hoyer, S., Roat, C., Hung, J., Rueden, C. T., Shankar, A.,
Finkbeiner, S., & and Nelson, P. (2018), "[Assessing microscope image focus
quality with deep learning](https://doi.org/10.1186/s12859-018-2087-4)",
BMC BioInformatics 19(1).

## Quickstart

Assuming you already have [Apache Maven](https://maven.apache.org) installed:

```sh
mvn -Pexec
```
Instructions for installing [Apache Maven](https://maven.apache.org) might be
as simple as `apt-get install maven` on Ubuntu and `brew install maven` on OS X
with [homebrew](https://brew.sh).

See the [ImageJ Development](https://imagej.net/Development) page for
further details.

## Installation in Fiji

If you have [Fiji](https://fiji.sc) installed and want to incorporate this
plugin into your installation:

```sh
# Set this to the path there Fiji.app is installed
FIJI_APP_PATH="/Users/me/Desktop/Fiji.app"
mvn -Dimagej.app.directory="${FIJI_APP_PATH}"
```

Then restart Fiji and search for "quality", or navigate to
_Plugins › Classification › Microscope Image Focus Quality_ in the menu.
