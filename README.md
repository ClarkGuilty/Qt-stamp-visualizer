# Qt-stamp-visualizer
Qt tool for the visual inspection and classification of astronomical stamps.
This is a fork of [Visualisation-tool](https://github.com/esavary/Visualisation-tool) using the Qt framework.

Originally developed with single-band `.fits` files in mind, it can also be used with (mutichannel) `.png`, `.jpg` and `.jpeg` files, albeit with limited functionality.

## Features
- Fast classification of thousands of images using the mosaic app.
- Easy access to Legacy Survey stamps using the sequential app.
- Nice visualization of single band astronomical images with different color maps and color scales.
- Easy handling of many different classifications for the same dataset using the argument: `-N name_of_the_classification`.
- Easy handling of different datasets using the argument: `-p /path/to/the/dataset`.
- Direct access to **ds9** and the Legacy Survey webapp for detailed inspection.
- Keyboard shortcuts for even faster classification!


## Usage
### Mosaic app
```

python mosaic_viewer.py -p /path/to/the/files -N name_of_the_classification -s seed

optional arguments:
-h, --help                          Show help message and exit.
-p PATH, --path PATH                Path to the images to inspect.
-N NAME, --name NAME                Name of the classifying session.
-l NCOLS, --ncols NCOLS             Number of columns per page.
                                    (default: 10).
-m NROWS, --nrows NROWS             Number of rows per page.
                                    (default: 10).
-s SEED, --seed SEED                Seed used to shuffle the images, must 
                                    be a positive integer. (default: None).
--minimum_size MINIMUM_SIZE         Minimum size of the stamps in the mosaic.
                                    The optimal value depends on your screen
                                    and on the stampsize.
--printname, --no-printname         Whether to print the filename when you                               
                                    click. (default: False).
--page PAGE                         Initial page.
--resize, --no-resize               Set to allow the resizing of the stamps
                                    with the window. (default: False).
--fits, --no-fits                   Forces app to only use fits (--fits)
                                    or png/jp(e)g (--no-fits).
                                    If unset, the app searches for fits files
                                    in the path, but defaults to png/jp(e)g
                                    if no fits files are found. (default: None).
```
- `left click` to mark a stamp as a gravitational lens candidate.
- `shift+left` click to mark a stamp as *interesting*.
- `left click` on an already marked stamp to undo the classification.
- `d` go to previous page.
- `f` go to following page.

### Sequential app
```
python single_viewer.py -p /path/to/the/files -N name_of_the_classification -s seed

optional arguments:

-h, --help            Show this help message and exit.
-p PATH, --path PATH  Path to the images to inspect. (default: Stamps_to_inspect).
-N NAME, --name NAME  Name of the classifying session.
--reset-config        Removes the configuration dictionary during startup.
--verbose             Activates loging to terminal. (default: False).
--clean               Cleans the legacy survey folder.
--fits, --no-fits     Forces app to only use fits (--fits) or png/jp(e)g (--no-fits).
                      If unset, the app searches for fits files in the path, but defaults
                      to png/jp(e)g if no fits files are found. (default: None).
-s SEED, --seed SEED  Seed used to shuffle the images. (default: None).

```
- Toggle `Legacy Survey (LS)` to automatically display a Legacy Survey cutout of the same area.
- Toggle `Large Fov` to display a Legacy Survey cutout covering of 4.5 arcmin².
- Toggle `Residuals` to display a Legacy Survey cutout of the residuals of the area.
- Toggle `Pre-fetch` to download in the background the Legacy Survey images of all the stamps to classify.
- Toggle `Auto-next` to automatically show the next stamp after making a classification.
- Toggle `Keyboard shortcuts` to activate the keyboard shortcuts:
    
|Key|Action|Key|Action|
|--------------|---------|--------------|---------|
|`q`|Grade A|`a`|Merger|
|`w`|Grade B|`s`|Spiral|
|`e`|Grade C|`d`|Ring|
|`r`|Grade X|`f`|Elliptical|
|||`g`|Disc|
|||`h`|Edge-on|

Note that classifying an image as Merger, Spiral, Ring, Elliptical, Disc, or Edge-on will also classify them as **Grade X**.

## Installation

### Requirements
- Python (>= 3.9, < 3.12)
- numpy
- pandas
- matplotlib
- pyside6
- pillow
- pyparsing
- astropy
 
### Conda
First create the environment:

```bash
conda create -n qt_classifier -c conda-forge "python<3.12,>=3.9" numpy pandas matplotlib pyside6 pillow pyparsing astropy
```

Don't forget to activate the environment before using the app:

```bash
conda activate qt_classifier
```

### pip

```bash
pip install numpy pandas matplotlib pyside6 pillow pyparsing astropy
```

## Issues
You can report bugs, suggest new features, or ask questions about usage by opening an issue on this Github repository.

Also feel free to email me with questions on usage.