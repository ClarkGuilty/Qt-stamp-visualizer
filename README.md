# Qt-stamp-visualizer

Qt tools for the visual inspection and classification of astronomical stamps.

A fork of [Visualisation-tool](https://github.com/esavary/Visualisation-tool), rebuilt on
the Qt framework (PySide6).

> **Note:** the `unified_dev_with_lobby` branch is experimental — it was vibecoded with
> Claude. See [CHANGES.md](CHANGES.md) for what changed and why.

## Contents

The repo ships three programs that are meant to be used together:

| Tool | Script | What it's for |
| --- | --- | --- |
| **Lobby** | `lobby.py` | Launcher GUI that configures and chains the two viewers, with no CLI flags to type. Also runs headless. |
| **Mosaic** | `mosaic.py` | Page-based grid of many stamps at once, for fast first-pass triage. |
| **1-by-1** | `single_viewer.py` | One object at a time, for detailed classification. |

Highlights:

* Multiband FITS with configurable bands and on-the-fly RGB composites — nothing is
  hardcoded to VIS/Y/J/H/I.
* PNG/JPG input works too, detected per band directory, so a session can mix formats.
* Configurable classification schemes (major classes and subclasses) with keyboard
  shortcuts.
* Per-object lookups in ds9, Legacy Survey, PanSTARRS and ESASky.
* Chained workflow: triage in the mosaic, extract what you marked, classify the subset
  1-by-1 — all driven from the lobby.

## Getting the code

For any given visual inspection it's usually better to get the tool together with the
data. To inspect your own images, clone the repo directly:

```bash
git clone -b unified_dev_with_lobby --single-branch https://github.com/ClarkGuilty/Qt-stamp-visualizer.git
cd Qt-stamp-visualizer
```

Only the `unified_dev_with_lobby` branch matters here.

## Installation

### Requirements

* Python >= 3.9, < 3.12
* numpy, pandas, matplotlib, pyside6, pillow, pyparsing, astropy

All of these are declared in `pyproject.toml` and installed for you below.

### uv or pip (recommended)

[uv](https://docs.astral.sh/uv/) is the fastest option and needs no separate Python
install step. From the repo root:

```bash
uv venv
uv pip install -e .
```

Plain `venv` + `pip` work just as well — `venv` ships with Python itself, so there's
nothing extra to install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Either route also installs three console commands, usable from anywhere in the
environment:

| Command | Equivalent to |
| --- | --- |
| `qtstamp-lobby` | `python lobby.py` |
| `qtstamp-mosaic` | `python mosaic.py` |
| `qtstamp-single` | `python single_viewer.py` |

If plain `pip`/`python` fail, try `pip3`/`python3`.

### Conda (alternative)

Reach for this only if you already manage your environments with Conda and would rather
keep everything in one place. None of the dependencies need Conda's system-level package
management, so the uv/pip route above does the same job more simply.

1. Install a Conda distribution if you don't have one —
   [Miniconda](https://docs.anaconda.com/free/miniconda) is the lightweight option.
2. Create an [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
   with the required packages:

   ```bash
   conda create -n qt_classifier -c conda-forge "python<3.12,>=3.9" \
       numpy pandas matplotlib pyside6 pillow pyparsing astropy
   ```

3. Activate it before using the app:

   ```bash
   conda activate qt_classifier
   ```

## Data layout

Both viewers expect one subdirectory per band under `--path`, with the same objects
matched across bands by file stem (the name without its extension):

```
my_stamps/
├── VIS/          # main band (-b/--main_band)
│   ├── source_001.fits
│   └── source_002.fits
├── Y/            # color bands (-B/--color_bands)
├── J/
├── H/
└── I/
```

* **Any** directory name can be a band — not just VIS/Y/J/H/I. At startup each tool
  checks that every band you reference (main band, color bands, and every RGB
  composite's members) has a matching directory, and lists what's actually there if
  one is missing.
* The **main band's** directory decides the object list: FITS if it has any, otherwise
  PNG/JPG. Format is detected per band directory, so `source_001.fits` in one band and
  `source_001.png` in another is fine.
* **RGB composites need all three member bands in FITS** — they're computed from pixel
  values, which a display-ready PNG no longer carries. A composite naming a non-FITS
  band is skipped at startup with a message saying which band disqualified it.
* All three bands in a composite must have the exact same image dimensions (and ideally
  the same zero-point).
* For non-FITS input, everything that needs real pixel values or WCS coordinates — ds9,
  Legacy Survey, PanSTARRS, ESASky, the pre-fetch toggles, copy RA/Dec, and the
  scale/colormap controls — is disabled up front rather than failing when you reach for it.

## Quick start

```bash
python lobby.py
```

Set the data path, a session name and a seed in the window, pick a run mode, hit **Run**.
Everything below is detail.

## Lobby (launcher)

`lobby.py` is a launcher GUI that sits in front of the two viewers. It's the easiest way
to run a full classification session without typing CLI flags by hand.

```bash
python lobby.py       # or: qtstamp-lobby
```

Run it with no arguments — everything is set in the window, and it remembers your
settings (data path, session name, seed, mosaic layout, 1-by-1 classification scheme,
panel arrangement, …) between runs in `.config_lobby.json`.

In the window you can:

* Pick the **path** to the images, plus an optional **session name** and **seed** — the
  same values you'd otherwise pass as `-p`, `-N` and `-s`.
* Choose a **run mode**: mosaic only, 1-by-1 only, or mosaic → 1-by-1 chained. In chained
  mode the lobby runs the mosaic tool first; once you close it, the lobby auto-detects
  the classification CSV it just wrote (or asks you to pick it), extracts everything you
  marked as "positive" into the output path via symlink or copy, then launches the
  1-by-1 tool on that extracted subset.
* Set which **mosaic codes count as positive** for extraction: Uninteresting (0), Lens
  candidate (1), Interesting (2), in any combination.
* Configure **bands and RGB composites** for both tools.
* Build the 1-by-1 tool's **classification scheme** — major and subclass buttons and their
  keyboard shortcuts — in a table instead of writing the `--classifications` string by
  hand, and mark which major classes count as positive for extraction.
* Pick an **output path** and whether extraction copies or symlinks files.
* Use **Extract only…** to run just the CSV-to-extracted-subset step on its own, without
  launching either tool afterward.

Setup groups live in dockable panels (Session, Bands, Mosaic options, Extraction options,
1-by-1 classification scheme, Log), so you can drag them side by side, stack them, tab
them together or float them into their own window, and resize whichever one you're using.
The **Run** button stays put in the top toolbar.

The **Log** panel streams the output of whichever tool the lobby launches, live while it
runs. The mosaic prints the file name of every stamp you click there — untick
`Print name on click` to silence it.

### Presets

The toolbar carries a **Preset** dropdown alongside **Save as…** and **Restore previous**.
Presets are plain JSON files in `.predefined_configs/`, one per name — local to your
checkout, not tracked by git, and created by **Save as…**. Picking one from the dropdown
applies it immediately, and **Restore previous** undoes the swap. If a preset references
bands that don't exist under your current data path, it still loads and the Log says
which bands went missing.

### Headless

The lobby also runs the whole workflow from the command line, chained extraction
included, without ever opening its window. Anything you don't pass is read from the
saved lobby config, so the usual pattern is to set a run up once in the GUI, then script
the reruns.

```bash
# Run the configured workflow headless
python lobby.py --no-gui -m chained -p PATH_TO_FILES -o OUTPUT_PATH -N NAME -s SEED

# Just print the viewer command(s) it would launch, then exit
# (chained mode prints both the mosaic and the 1-by-1 stage)
python lobby.py --print-command -m chained -p PATH_TO_FILES -o OUTPUT_PATH -N NAME
```

| Option | Description |
| --- | --- |
| `--no-gui` | Run the configured workflow without opening the lobby window. |
| `--print-command` | Print the viewer command line(s) that would run, then exit (implies `--no-gui`). |
| `--config PATH` | Lobby config JSON to read defaults from (default: `.config_lobby.json`). |

Workflow overrides, valid only with `--no-gui` or `--print-command`:

| Option | Description |
| --- | --- |
| `-m`, `--mode` | `mosaic`, `single`, or `chained`. |
| `-p`, `--path` | Path to the images to inspect. |
| `-o`, `--output` | Output path for chained-mode extraction. |
| `-N`, `--name` | Session name. |
| `-s`, `--seed` / `--no-seed` | Shuffle seed, or ignore any seed set in the config. |
| `-b`, `--main-band` | Main / high-resolution band (e.g. `"VIS"`). |
| `-B`, `--color-bands` | Comma-separated individually-selectable bands. |
| `--rgb-composites` | Semicolon-separated `R,G,B` band-name triples. |
| `--classifications` | 1-by-1 scheme string; overrides the scheme rows from the config. |
| `--ncols`, `--nrows` | Mosaic columns and rows per page. |
| `--printname` / `--no-printname` | Mosaic: print the filename on click. |
| `--copy` / `--symlink` | Chained extraction copies or symlinks files (symlink is the default). |

See `python lobby.py --help` for the authoritative list.

---

## Direct use (without the lobby)

Both viewers run straight from the command line too — useful for scripting, batch
launches, or once you already know the flags you want.

### Mosaic tool

Page-based, grid-of-stamps view for quick first-pass triage. Click through many objects
at once to mark lens candidates and flag interesting ones, then hand anything worth a
closer look to the 1-by-1 tool.

```bash
python mosaic.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER
```

| Option | Default | Description |
| --- | --- | --- |
| `-h`, `--help` | | Show the help message and exit. |
| `-p`, `--path` | `Color_stamps_to_inspect` | Path to the images to inspect. |
| `-N`, `--name` | none | Name of the classifying session. |
| `-b`, `--main_band` | `VIS` | High-resolution band: the panel that's always individually shown and pre-selected. |
| `-B`, `--color_bands` | `Y,J,H` | Comma-separated bands made individually selectable as their own panel, in addition to the RGB composites. |
| `--rgb-composites` | `H,Y,I;H,J,Y` | Semicolon-separated `R,G,B` band-name triples. A composite's label is its comma-joined band list. |
| `-l`, `--ncols`, `--gridsize` | `5` | Columns per page. Find the right value **before** starting a classification. |
| `-m`, `--nrows` | `8` | Rows per page. Same caveat. |
| `-s`, `--seed` | none | Seed used to shuffle the images. |
| `--minimum_size` | `66` | Minimum stamp size in the mosaic. Safe to change mid-classification; try smaller values if the mosaic doesn't fit your screen. |
| `--printname` / `--no-printname` | on | Print the file name of every stamp you click (shown in the lobby's Log panel). |
| `--page` | none | Initial page. |
| `--resize` / `--no-resize` | off | Let the stamps resize with the window. |

**Controls**

| Key | Alt | Action |
| --- | --- | --- |
| *left click* | | Mark as lens candidate, or clear the classification |
| *shift+left click* | *ctrl+left click* | Mark as *interesting* |
| *f* | *j* | Next page |
| *d* | *k* | Previous page |

Inside the app, the **Panels** dropdown chooses which bands show for each object: the
main band, the RGB composites (`H,Y,I` and `H,J,Y` by default), and the individual color
bands (`Y`, `J`, `H` by default). The color bands start unchecked, so the out-of-the-box
view is the main band plus the two composites.

> **Careful with `--ncols` / `--nrows`.** They're part of the classification CSV's
> filename, so changing either after you've started means starting a *new* classification.
> If the images are too big for your screen, prefer `--minimum_size`, which is safe to
> change at any time.

### 1-by-1 sequential tool

Detailed, one-object-at-a-time classification. Multiband FITS visualization (main band,
configurable RGB composites, individual color bands), configurable major and subclass
grading buttons with keyboard shortcuts, and optional per-object lookups in Legacy
Survey, PanSTARRS, ds9 and ESASky.

Replace `NAME` with your name and `SEED_NUMBER` with any number larger than 1000.

```bash
python single_viewer.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER
```

| Option | Default | Description |
| --- | --- | --- |
| `-h`, `--help` | | Show the help message and exit. |
| `-p`, `--path` | `Color_stamps_to_inspect` | Path to the images to inspect. |
| `-N`, `--name` | none | Name of the classifying session. |
| `-b`, `--main_band` | `VIS` | High-resolution band: the panel that's always individually shown and pre-selected. |
| `-B`, `--color_bands` | `Y,J,H` | Comma-separated bands to make individually selectable in the **Panels** dropdown. |
| `--rgb-composites` | `H,Y,I;H,J,Y` | Semicolon-separated `R,G,B` band-name triples. A composite's label is its comma-joined band list. |
| `--classifications` | `A=1;B=2;C=3;X=4;I=5` | Classification buttons — see below. |
| `-s`, `--seed` | none | Seed used to shuffle the images. |
| `--legacysurvey` / `--no-legacysurvey` | **on** | Legacy Survey panel and downloads. Pass `--no-legacysurvey` to drop the panel entirely (the LS server can be unreliable). |
| `--ls-big-fov-residuals` / `--no-...` | off | Also pre-fetch the large-FoV Legacy Survey *residual* cutout, so **Large FoV** + **Residuals** shows a cached image instead of downloading on demand. Costs a fourth LS request per object. |
| `--reset-config` | off | Remove the saved configuration dictionary during startup. |
| `--verbose` | off | Log to the terminal. |
| `--clean` | off | Clean the Legacy Survey cache folder. |

**Classification scheme.** `--classifications` takes one semicolon-separated string of
`MAJOR=KEY` or `MAJOR:SUB=KEY` entries. A bare `MAJOR=KEY` (or an empty `SUB`) makes a
major-class button; `MAJOR:SUB=KEY` makes a subclass button under that major, setting
both the classification and the subclassification in one click. For example:

```
--classifications "A=1;B=2;C=3;X=4;I=5;X:Merger=a;X:Spiral=s"
```

**Panels and settings**

* The **Panels** dropdown assigns each panel — the main band, the RGB composites, the
  individual color bands, Legacy Survey and PanSTARRS — to one of 3 display rows
  (mutually exclusive across rows). A row appears as soon as you check a panel into it,
  and the title bar shows a live summary of what's in each one. The dropdown also carries
  a **Large FoV** toggle for wider Legacy Survey / PanSTARRS cutouts.
* The **Tools** dropdown holds *Open ds9*, *Open LS*, *Open PanSTARRS* and *Open ESASky*.
* The **Settings** dropdown holds:
  * **Pre-fetch PanSTARRS** and **Pre-fetch Legacy Survey** — independent toggles, each
    with its own thread, downloading the next object's cutouts in the background while
    you classify the current one.
  * **Auto-next** — advance to the next stamp automatically after a classification.
  * **Keyboard shortcuts** — enable the shortcuts below.

**Controls**

| Key | Alt | Action |
| --- | --- | --- |
| *1* | | Grade A |
| *2* | | Grade B |
| *3* | | Grade C |
| *4* | | Grade X |
| *5* | | Interesting |
| *PgDn* | *j* | Next object |
| *PgUp* | *k* | Previous object |
| *c* | | Copy filename |
| *ctrl+c* | | Copy coordinates |

The grading keys above are just the default `--classifications` scheme; pass your own to
change the classes, subclasses and keys.

## Files the tools write

| Path | Contents |
| --- | --- |
| `Classifications/` | Classification CSVs, auto-saved as you work. Named after the session, the number of images, the seed — and, for the mosaic, the page layout. |
| `.config_lobby.json` | Saved lobby settings. |
| `.config_mosaic.json` | Saved mosaic settings. |
| `.config.json` | Saved 1-by-1 settings (`--reset-config` clears it). |
| `.predefined_configs/` | Named lobby presets. |
| `Legacy_survey/`, `PanSTARRS/` | Downloaded cutout caches (`--clean` clears the Legacy Survey one). |

## License and contributors

MIT — see [LICENSE](LICENSE). Contributors are listed in
[CONTRIBUTORS.md](CONTRIBUTORS.md).
