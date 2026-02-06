# Visualization tool for lensed QSO searches

Tool for the visual inspection and classification of astronomical stamps. Adapted to show PNGs and JPGs.

It includes:
* 1-by-1 tool for detailed classification.
* Keyboard shortcuts for fast classification.

## Installation

You need to install the following libraries. You can install them using your system's package manager, but I recommend that you simply use a "Conda environment":https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html


###### Requirements

* Python (>= 3.10)
* numpy
* pandas
* matplotlib
* pyside6
* pillow
* pyparsing
* astropy


 
### Installation using Conda (recommended)

* **Step zero** : install a Conda distribution (you can skip this if you already have one). Here is the official download link with instructions for "Microconda":https://docs.anaconda.com/free/miniconda/ (a lightweight Conda distribution).

* **Step one**: create a "Conda environment":https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html with all the required packages.


<pre><code class="bash">
conda create -n qt_classifier -c conda-forge "python>=3.10" numpy pandas matplotlib pyside6 pillow pyparsing astropy
</code></pre>


* **Step two**: Activate the environment before using the app:


<pre><code class="bash">
conda activate qt_classifier
</code></pre>


### Installation using pip (not recommended)

__Alternatively__ , you can also use pip to get the packages. First, make sure that you have pip installed. If not, install it using your package manager. Remember that if you use "pip" inside a Conda environment, you cannot use the "conda" command inside that environment ever again. All new packages must be installed via "pip".

<pre><code class="bash">
pip install numpy pandas matplotlib pyside6 pillow pyparsing astropy
</code></pre>


If this fails try replacing `pip` for `pip3` (and from then on replace every `python` for `python3`).

## Usage



### 1-by-1 sequential tool

Please replace **NAME** with your name, and **SEED_NUMBER** with any number larger than 1000.

<pre><code class="bash">
python Euclid_jpg_edition_for_lensed_QSO_dev.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER

optional arguments:

-h, --help            Show this help message and exit.
-p PATH, --path PATH  Path to the images to inspect.
-N NAME, --name NAME  Name of the classifying session.
--reset-config        Resets your configuration preferences (auto-next and the displayed bands)
-s SEED, --seed SEED  Seed used to shuffle the images. (default: None).


</code></pre>

* Use drop-down menus to change the number of rows being displayed, as well as the bands shown in each row.
* Toggle `Auto-next` to automatically show the next stamp after making a classification.
* Toggle `Keyboard shortcuts` to activate the keyboard shortcuts:
    
| Key | Key | Action |
| ------------- | ------------- | ------------- |
| **1**||Grade A/B|
| **4** ||Grade C/X|
| **PgDn**|**j**|Next object|
| **PgUp**|**k**|Previous object|
| **c**||Copy filename|


#### Exporting your classification

Your classifications are saved in `Classifications`, with filename: `classification_lensed_QSOs_NAME_SIZE_SEED.csv`, where `SIZE` is the number of objects graded. Please upload this file using the corresponding google form.

## Issues

You can report bugs or ask for assistance via Github issues or the Euclid slack.

## Downloading the just the app

It is always better to get the tool with the data. However, if you want to inspect RR2 images on your own, you can get just the app by cloning the git repo:

<pre><code class="shell">
git clone -b Euclid_jpg_edition_for_lensed_QSO --single-branch https://github.com/ClarkGuilty/Qt-stamp-visualizer.git
</code></pre>

Note that we only care about the _Euclid_jpg_edition_for_lensed_QSO_ branch. 
