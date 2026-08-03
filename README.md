# Qt-stamp-visualizer
Qt tool for the visual inspection and classification of astronomical stamps.
This is a fork of [Visualisation-tool](https://github.com/esavary/Visualisation-tool) using the Qt framework.

> **Note:** the `unified_dev` branch is experimental — it was vibecoded with Claude. See [CHANGES.md](CHANGES.md) for details.

# Visualization tool for ERO inspection Cheat Sheet

Tool for the visual inspection and classification of astronomical stamps.

It includes:
* Mosaic tool for quick classification and 1-by-1 tool for detailed classification.
* Nice and simple visualization of ERO data for single band VIS, and color composites (HYVIS and HJY).
* Keyboard shortcuts for fast classification.
* Direct access ds9.

## Downloading the just the app

For any given visual inspection, it is always better to get the tool with the data. However, if you want to inspect ERO images on your own, you can get just the app by cloning the git repo:

<pre><code class="shell">
git clone -b ERO_edition --single-branch https://github.com/ClarkGuilty/Qt-stamp-visualizer.git
</code></pre>

Note that we only care about the _ERO_edition_ branch.


## Installation

You need to install the following libraries. You can install them using your system's package manager, but I recommend that you simply use a [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)


###### Requirements

* Python (>= 3.9, < 3.12)
* numpy
* pandas
* matplotlib
* pyside6
* pillow
* pyparsing
* astropy


 
### Installation using Conda (recommended)

* *Step zero* : install a Conda distribution (you can skip this if you already have one). Here is the official download link with instructions for [Microconda](https://docs.anaconda.com/free/miniconda) (a lightweight Conda distribution).

* *Step one*: create a (Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html] with all the required packages.


<pre><code class="bash">
conda create -n qt_classifier -c conda-forge "python<3.12,>=3.9" numpy pandas matplotlib pyside6 pillow pyparsing astropy
</code></pre>


* *Step two*: Activate the environment before using the app:


<pre><code class="bash">
conda activate qt_classifier
</code></pre>


### Installation using pip (not recommended)

_Alternatively_ , you can also use pip to get the packages. First, make sure that you have pip installed. If not, install it using your package manager. Remember that if you use "pip" inside a Conda environment, you cannot use the "conda" command inside that environment ever again. All new packages must be installed via "pip".

<pre><code class="bash">
pip install numpy pandas matplotlib pyside6 pillow pyparsing astropy
</code></pre>


If this fails try replacing "pip" for "pip3" (and from then on replace every "python" for "python3".

## Usage


### Mosaic tool

<pre><code class="shell">

python mosaic_viewer_ERO_edition.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the images to inspect.
  -N NAME, --name NAME  Name of the expert.
  -l NCOLS, --ncols NCOLS
                        Number of columns per page. (default 5). 
                        Find the optimal value before starting the classification.
                        Once you start the classification do not change this.
  -m NROWS, --nrows NROWS
                        Number of rows per page. (default: 8).
                        Find the optimal value before starting the classification.
                        Once you start the classification do not change this.
  -s SEED, --seed SEED  Seed used to shuffle the images.
  --minimum_size MINIMUM_SIZE
                        Minimum size of the stamps in the mosaic. 
                        The default (66) should be good enough, but you can try smaller values if the mosaic is too big for your screen.
                        You can change this even after you started a classification.
  --printname, --no-printname
                        Whether to print the name when you click. (default False)
  --page PAGE           Initial page.

</code></pre>



| Key               | Key               | Action                                       |
|-------------------|-------------------|----------------------------------------------|
| *left click*      |                   | Mark as lens candidate or remove classification |
| *shift+left click*| *ctrl+left click* | Mark as _interesting_                        |
| *f*               | *j*               | Next page                                    |
| *d*               | *k*               | Previous page                                |

Inside the app, you can change the number of images shown per galaxy. 1 image corresponds to only the high resolution VIS. 2 images corresponds to VIS + lower resolution HYVIS. 3 images adds a NISP-only composite rgb.

Note: If you find the images to be too big for your screen, you can try having less images (via `--ncols` and `--nrows`) or a smaller image size (via `--minimum_size`).
*However*, changing `--ncols` or `--nrows` also changes the filename of your classification csv (the file that stores your classifications!), so once you start the classification, abstain from changing `--ncols` and `--nrows`, since this means starting a new classification.


### 1-by-1 sequential tool

Please replace NAME with your name, and SEED_NUMBER with any number larger than 1000.

<pre><code class="bash">
python single_viewer_multiband_ERO_edition.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER

optional arguments:

-h, --help            Show this help message and exit.
-p PATH, --path PATH  Path to the images to inspect.
-N NAME, --name NAME  Name of the classifying session.
--reset-config        Removes the configuration dictionary during startup.
-s SEED, --seed SEED  Seed used to shuffle the images. (default: None).


</code></pre>

* Toggle `Show NISP bands` to see the single NISP bands (in visualization order Y, J and H).
* Toggle `Show NISP RGB` to see the composite RGB (HJY) NISP bands.
* Toggle `Auto-next` to automatically show the next stamp after making a classification.
* Toggle `Keyboard shortcuts` to activate the keyboard shortcuts:
    
| Key       | Key     | Action          |
|-----------|---------|-----------------|
| *1*       |         | Grade A         |
| *2*       |         | Grade B         |
| *3*       |         | Grade C         |
| *4*       |         | Grade X         |
| *5*       |         | Interesting     |
| *PgDn*    | *j*     | Next object     |
| *PgUp*    | *k*     | Previous object |
| *c*       |         | Copy filename   |
| *ctrl+c*  |         | Copy coordinates|
