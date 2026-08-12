# Qt-stamp-visualizer
Qt tool for the visual inspection and classification of astronomical stamps.
This is a fork of [Visualisation-tool](https://github.com/esavary/Visualisation-tool) using the Qt framework.

> **Note:** the `unified_dev_with_lobby` branch is experimental — it was vibecoded with Claude. See [CHANGES.md](CHANGES.md) for details.

# Visualization tool for ERO inspection Cheat Sheet

Tool for the visual inspection and classification of astronomical stamps.

It includes:
* Lobby launcher GUI to configure and chain the mosaic and 1-by-1 tools without touching the command line.
* Mosaic tool for quick classification and 1-by-1 tool for detailed classification.
* Nice and simple visualization of ERO data for single band VIS, and configurable RGB composites.
* Keyboard shortcuts for fast classification.
* Direct access ds9.

## Downloading the just the app

For any given visual inspection, it is always better to get the tool with the data. However, if you want to inspect ERO images on your own, you can get just the app by cloning the git repo:

<pre><code class="shell">
git clone -b unified_dev_with_lobby --single-branch https://github.com/ClarkGuilty/Qt-stamp-visualizer.git
</code></pre>

Note that we only care about the _unified_dev_with_lobby_ branch.


## Installation

The repo ships a `pyproject.toml`, so the recommended way to install it is
with [uv](https://docs.astral.sh/uv/) or plain `pip` in a `venv`, both shown
below. A Conda environment works too and is documented further down, but for
this project it's not doing anything uv/pip can't; none of the dependencies
need Conda's system-level package management.


###### Requirements

* Python (>= 3.9, < 3.12)
* numpy
* pandas
* matplotlib
* pyside6
* pillow
* pyparsing
* astropy


### Installation using uv or pip (recommended)

*uv* is the fastest option and needs no separate Python install step. From the
repo root:

<pre><code class="bash">
uv venv
uv pip install -e .
</code></pre>

If you don't have uv, plain `venv` + `pip` work just as well; `venv` ships
with Python itself, so there's nothing extra to install:

<pre><code class="bash">
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
</code></pre>

Either way, this also installs three console commands you can run from
anywhere in the environment: `qtstamp-mosaic`, `qtstamp-single`, and
`qtstamp-lobby`, equivalent to running `mosaic_viewer_ERO_edition.py`,
`single_viewer_multiband_ERO_edition.py`, and `lobby.py` directly.

If plain `pip`/`python` fail, try `pip3`/`python3` instead.


### Installation using Conda (alternative)

Reach for this if you already manage your environments with Conda and would
rather keep everything in one place; otherwise the uv/pip route above is
simpler and does the same job.

* *Step zero* : install a Conda distribution (you can skip this if you already have one). Here is the official download link with instructions for [Microconda](https://docs.anaconda.com/free/miniconda) (a lightweight Conda distribution).

* *Step one*: create a (Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html] with all the required packages.


<pre><code class="bash">
conda create -n qt_classifier -c conda-forge "python<3.12,>=3.9" numpy pandas matplotlib pyside6 pillow pyparsing astropy
</code></pre>


* *Step two*: Activate the environment before using the app:


<pre><code class="bash">
conda activate qt_classifier
</code></pre>

## Usage


### Lobby (launcher)

`lobby.py` is a launcher GUI that sits in front of the two tools below. It's the
easiest way to run a full classification session without typing out CLI flags
by hand.

<pre><code class="shell">
python lobby.py
</code></pre>

or, if installed via pip/uv (see Installation above):

<pre><code class="shell">
qtstamp-lobby
</code></pre>

It takes no command-line arguments; everything is set in the window, and it
remembers your settings (data path, session name, seed, mosaic layout, 1-by-1
classification scheme, etc.) between runs.

In the window you can:

* Pick the path to the images and an optional session name and seed, the same
  values you'd normally pass via `-p`, `-N` and `-s`.
* Choose a run mode: mosaic only, 1-by-1 only, or mosaic then 1-by-1 chained
  together. In chained mode, Lobby runs the mosaic tool first; once you close
  it, Lobby auto-detects the classification CSV it just wrote (or asks you to
  pick it if it can't), extracts everything you marked as "positive" (via
  symlink or copy) into the output path, then launches the 1-by-1 tool on
  that extracted subset.
* Set which mosaic codes count as "positive" for extraction: Uninteresting,
  Lens candidate, Interesting, any combination.
* Build the 1-by-1 tool's classification scheme (major and subclass buttons
  and their keyboard shortcuts) in a table instead of writing the
  `--classifications` string by hand, and mark which major classes count as
  "positive" for extraction.
* Pick an output path and whether extraction copies files or symlinks them.
* Use `Extract only...` to run just the CSV-to-extracted-subset step on its
  own, without launching either tool afterward.

A log pane at the bottom streams the output of whichever tool Lobby launches.

---

### Direct use (without Lobby)

The two tools below can also be run directly from the command line instead of
through Lobby. Useful for scripting, batch launches, or once you already know
the flags you want.

#### Mosaic tool

Page-based, grid-of-stamps view for quick first-pass triage. Click through a
mosaic of many objects at once to mark lens candidates and flag interesting
ones, then hand anything worth a closer look to the 1-by-1 tool.

<pre><code class="shell">

python mosaic_viewer_ERO_edition.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the images to inspect.
  -N NAME, --name NAME  Name of the expert.
  -l NCOLS, --ncols NCOLS, --gridsize NCOLS
                        Number of columns per page. (default 5). 
                        Find the optimal value before starting the classification.
                        Once you start the classification do not change this.
  -m NROWS, --nrows NROWS
                        Number of rows per page. (default: 8).
                        Find the optimal value before starting the classification.
                        Once you start the classification do not change this.
  -s SEED, --seed SEED  Seed used to shuffle the images.
  -b MAIN_BAND, --main_band MAIN_BAND
                        High resolution band, and the panel that's always
                        individually shown and pre-selected. (default: "VIS").
  -B COLOR_BANDS, --color_bands COLOR_BANDS
                        Comma-separated bands to make individually selectable
                        as their own panel, in addition to the RGB composites.
                        (default: "Y,J,H").
  --rgb-composites RGB_COMPOSITES
                        Semicolon-separated R,G,B band-name triples for the RGB
                        composites. A composite's label is its comma-joined
                        band list. (default: "H,Y,I;H,J,Y").
  --minimum_size MINIMUM_SIZE
                        Minimum size of the stamps in the mosaic. 
                        The default (66) should be good enough, but you can try smaller values if the mosaic is too big for your screen.
                        You can change this even after you started a classification.
  --printname, --no-printname
                        Whether to print the name when you click. (default False)
  --page PAGE           Initial page.
  --resize, --no-resize
                        Whether to let the stamps resize with the window. (default False)

</code></pre>

Any directory under `--path` can be used as a band -- not just VIS/Y/J/H/I.
At startup the tool checks that every band you reference (main band, color
bands, and every RGB composite's members) has a matching directory, and
lists what's actually there if one's missing. All three bands in a composite
must have the exact same image dimensions (and ideally the same zero-point).



| Key               | Key               | Action                                       |
|-------------------|-------------------|----------------------------------------------|
| *left click*      |                   | Mark as lens candidate or remove classification |
| *shift+left click*| *ctrl+left click* | Mark as _interesting_                        |
| *f*               | *j*               | Next page                                    |
| *d*               | *k*               | Previous page                                |

Inside the app, use the `Panels` dropdown to choose which bands show for each galaxy: the main band, the RGB composites (`H,Y,I` and `H,J,Y` by default), and now the individual color bands too (Y, J and H by default) -- available but unchecked out of the box, so the default view is unchanged.

Note: If you find the images to be too big for your screen, you can try having less images (via `--ncols` and `--nrows`) or a smaller image size (via `--minimum_size`).
*However*, changing `--ncols` or `--nrows` also changes the filename of your classification csv (the file that stores your classifications!), so once you start the classification, abstain from changing `--ncols` and `--nrows`, since this means starting a new classification.


#### 1-by-1 sequential tool

Detailed, one-object-at-a-time classification tool. Multiband FITS
visualization (VIS, configurable RGB composites, individual NISP bands),
configurable major and subclass grading buttons with keyboard shortcuts, and
optional per-object lookups in Legacy Survey, PanSTARRS, ds9, and ESASky.

Please replace NAME with your name, and SEED_NUMBER with any number larger than 1000.

<pre><code class="bash">
python single_viewer_multiband_ERO_edition.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER

optional arguments:

-h, --help            Show this help message and exit.
-p PATH, --path PATH  Path to the images to inspect.
-N NAME, --name NAME  Name of the classifying session.
-b MAIN_BAND, --main_band MAIN_BAND
                      High resolution band, and the panel that's always
                      individually shown and pre-selected. (default: "VIS").
-B COLOR_BANDS, --color_bands COLOR_BANDS
                      Comma-separated bands to show individually in the
                      Panels dropdown. (default: "Y,J,H").
--reset-config        Removes the configuration dictionary during startup.
--verbose             Activates logging to the terminal.
--clean               Cleans the Legacy Survey cache folder.
--legacysurvey, --no-legacysurvey
                      Enables the Legacy Survey panel and downloads. Off by
                      default, since the Legacy Survey server can be
                      unreliable.
-s SEED, --seed SEED  Seed used to shuffle the images. (default: None).
--classifications CLASSIFICATIONS
                      Classification buttons: semicolon-separated MAJOR=KEY or
                      MAJOR:SUB=KEY entries. A bare MAJOR=KEY (or empty SUB)
                      makes a major-class button; MAJOR:SUB=KEY makes a
                      subclass button under that major, setting both the
                      classification and subclassification when clicked.
                      (default: "A=1;B=2;C=3;X=4;I=5", i.e. today's 5 buttons).
                      Example with subclasses:
                      "A=1;B=2;C=3;X=4;I=5;X:Merger=a;X:Spiral=s"
--rgb-composites RGB_COMPOSITES
                      Semicolon-separated R,G,B band-name triples for the RGB
                      composites. A composite's label is its comma-joined
                      band list. (default: "H,Y,I;H,J,Y").


</code></pre>

* Use the `Panels` dropdown to choose which bands show in each of the three
  display rows: VIS, the RGB composites (`H,Y,I` and `H,J,Y` by default),
  the individual NISP bands (Y, J and H), and PanSTARRS. It also has a
  `Large FoV` toggle for wider Legacy Survey / PanSTARRS cutouts.
* Toggle `Pre-fetch` (in the `Settings` dropdown) to download the next
  object's Legacy Survey / PanSTARRS cutouts in the background while you
  classify the current one.
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

The classification/subclassification buttons and their shortcuts above are just the
default `--classifications` scheme; pass your own to change the classes, subclasses,
and keys (see the CLI options above).
