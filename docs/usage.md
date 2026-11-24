# Usage

This page is the full reference for the three tools. It covers how the data should be
laid out, what the lobby can do, and every flag and control of the two viewers. The
[README](../README.md) has the short version, and [settings.md](settings.md) explains
where the tools keep their settings and which files they write.

The commands below are written as `python lobby.py` and so on, which is how you run the
tools from a checkout. Once installed with pip or uv, the same tools are called
`qtstamp-lobby`, `qtstamp-mosaic` and `qtstamp-single`, and they take the same flags.

## Data layout

Both viewers read one subdirectory per band under `--path`, with the same objects
matched across bands by file stem (the name without its extension):

```
my_stamps/
├── VIS/          # default band (-b/--main_band)
│   ├── source_001.fits
│   └── source_002.fits
├── Y/            # color bands (-B/--color_bands)
├── J/
├── H/
└── I/
```

* **Any** directory name can be a band — not just VIS/Y/J/H/I. With no band arguments
  the viewers use `VIS`, `Y`, `J`, `H` and the two default composites where the data has
  them; without a `VIS` directory they use every band directory they find. Stamps sitting
  directly under `--path`, with no band directories, are shown as one band (`-b .`).
  A band you *name* must have a directory: the tool stops and lists what is there.
* **The objects are every stamp found in any of the bands in use.** No band has to hold
  all of them. A band an object lacks, or whose file cannot be read, shows as a grey
  *no data* cell, and the status bar names the object and the band.
* **The default band** (`-b`) is the panel shown first and pre-selected, nothing more.
  `-b ''` (or `(none)` in the lobby) hands that to the first `-B` band.
* FITS (`.fits`, `.fit`, `.fts`, also gzipped as `.fits.gz` or tile-compressed by
  fpack as `.fits.fz`) and PNG/JPG (`.png`, `.jpg`, `.jpeg`) are recognised in any
  letter case; `source_001.fits.gz` is the object `source_001`. Each file is read as
  what it is, so formats can mix between bands and inside one: `source_001.fits` beside
  `source_002.png` is fine. When one band holds both `source_001.fits` and
  `source_001.png`, the FITS file is the object's stamp.
  Hidden files (`.DS_Store`, `._source_001.fits`) are ignored.
* **An RGB composite is made, per object, from three FITS stamps of the same
  dimensions** — it is computed from pixel values, which a display-ready PNG no longer
  carries. When an object lacks one of the three, has it as PNG/JPG, or has them in
  different sizes, the composite is not made and its cell reads *no data*. A composite
  naming a band with no FITS at all is dropped at startup with a message.
* RA/Dec for the survey panels, ds9 and the browser links come from the object's first
  FITS stamp with a sky WCS, the default band's first. In a dataset with no FITS at
  all, everything that needs pixel values or coordinates — ds9, Legacy Survey,
  PanSTARRS, ESASky, the pre-fetch toggles, copy RA/Dec, and the scale/colormap
  controls — is disabled up front rather than failing when you reach for it.

### Multi-extension FITS

The other layout is one FITS file per object, with each band as an image extension
named by its `EXTNAME`:

```
my_stamps/
├── source_001.fits   # extensions VIS, NIR_Y, NIR_J, NIR_H, ...
└── source_002.fits
```

Pass `--mef` to either viewer; the lobby recognises the layout by itself. Band names are
then extension names (`-b VIS -B NIR_Y,NIR_J`), looked up in each object's own file, so
files may differ: a band one file lacks shows as a *no data* cell for that object.
Without `--mef`, a directory of FITS files is shown as one band, each file's first image.

## Lobby (launcher)

`lobby.py` is a launcher GUI that sits in front of the two viewers. It's the easiest way
to run a full classification session without typing CLI flags by hand.

```bash
python lobby.py       # or: qtstamp-lobby
```

Run it with no arguments — everything is set in the window, and it remembers your
settings between runs. Workspace-wide things (data path, run mode, panel arrangement,
extraction options, …) live in `.qtstamp/config_lobby.json` in whatever directory you
launched it from; the **classification scheme, band setup and mosaic grid shape travel
with the session** instead, so reopening an earlier session brings back the scheme it
was graded with rather than whatever you used last — see
[settings.md](settings.md).

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
  hand, and mark which major classes count as positive for extraction. A subclass row
  picks its major(s) from a multi-select dropdown, so one subclass can be shared by
  several majors. They are offered in the order you tick them.
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
Presets are plain JSON files, one per name, created by **Save as…**. The dropdown lists
three sources at once, most specific first: `.qtstamp/presets/` in your working directory,
then the per-user config directory, then the presets shipped with the tool — so the
dropdown is useful before you have saved anything. Four ship: three per-survey band
setups (Euclid ERO, Legacy Survey, PanSTARRS) and `ERO_edition_classic`, which
reproduces the old `ERO_edition` scripts — VIS main band, Y/J/H colour bands, the
H,Y,I and H,J,Y composites, a 5x8 mosaic with `Print name on click` off, and the
`A` / `B` / `C` / `X` / `I` grades on keys 1-5, with `A` the one marked positive
for extraction. Use it to resume a classification those scripts started: the labels
match what they wrote into the CSV.

Saving always writes to `.qtstamp/presets/`, so saving under a shipped preset's name
shadows it locally rather than overwriting it. Picking one from the dropdown applies
it immediately, and **Restore previous** undoes the swap.

A preset holds the classification scheme, the band setup, the mosaic layout and the
run options. It never holds the data path, the output or classifications paths, the
session name or the seed, so loading one never changes which dataset or session you
are on. An older preset that still has those keys loads, and the Log lists them as
ignored. Loading a preset always rescans the current data path and shows only the
bands found there. If the preset uses bands that aren't there, or there are bands the
preset doesn't use, a warning lists both.

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
| `--config PATH` | Lobby config JSON to read defaults from (default: the first `config_lobby.json` found along the state-dir search path). |

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
| `--classifications-dir` | Directory for the autosaved classification CSVs (default: `./Classifications`, relative to wherever the lobby is launched from). Passed straight through to the viewer(s). |

See `python lobby.py --help` for the authoritative list.

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
| `-b`, `--main_band` | `VIS` if present | The default band: the panel shown first and pre-selected. `-b ''` for none: the first `-B` band takes its place. See [Data layout](#data-layout) for what is used when it is not given. |
| `-B`, `--color_bands` | those of `Y,J,H` present | Comma-separated bands made individually selectable as their own panel, in addition to the RGB composites. |
| `--rgb-composites` | those of `H,Y,I;H,J,Y` whose bands are present | Semicolon-separated `R,G,B` band-name triples. A composite's label is its comma-joined band list. |
| `--mef` | off | `--path` holds one multi-extension FITS file per object; bands are `EXTNAME`s. See [Multi-extension FITS](#multi-extension-fits). |
| `-l`, `--ncols`, `--gridsize` | `5` | Columns per page. Find the right value **before** starting a classification. |
| `-m`, `--nrows` | `8` | Rows per page. Same caveat. |
| `-s`, `--seed` | none | Seed used to shuffle the images. |
| `--minimum_size` | `66` | Minimum stamp size in the mosaic. Safe to change mid-classification; try smaller values if the mosaic doesn't fit your screen. |
| `--printname` / `--no-printname` | on | Print the file name of every stamp you click (shown in the lobby's Log panel). |
| `--page` | none | Initial page (1-based). Overrides the resume page. |
| `--resize` / `--no-resize` | off | Let the stamps resize with the window. |
| `--reset-config` | off | Forget the saved preferences (colormap, scale, panels). Resume positions are kept. |
| `--reset-position` | off | Forget the resume page for this `--path`/`--name`/`--seed`, so the session restarts at the first page. Preferences are kept. |
| `--classifications-dir` | `./Classifications` | Directory for the autosaved classification CSVs. Absolute, or relative to the directory you launch from; created if it doesn't exist. |

**Controls**

| Key | Alt | Action |
| --- | --- | --- |
| *left click* | | Mark as lens candidate, or clear the classification |
| *shift+left click* | *ctrl+left click* (macOS: *cmd+left click*) | Mark as *interesting* |
| *f* | *j* | Next page |
| *d* | *k* | Previous page |

Inside the app, the **Panels** dropdown chooses which bands show for each object: the
main band, the RGB composites (`H,Y,I` and `H,J,Y` by default), and the individual color
bands (`Y`, `J`, `H` by default). The color bands start unchecked, so the out-of-the-box
view is the main band plus the two composites.

> **`--ncols` / `--nrows` are safe to change mid-classification.** The grid shape is a
> display choice — how many stamps fit comfortably on your screen — so it is no longer
> part of the classification CSV's filename. Reshaping keeps the same file and the grades
> already in it, and your *place* in the deck is kept too: the position is stored as a
> filename, so the tool reopens on whichever page now holds the object you were on.
> `--minimum_size` is equally safe, and is still the better knob if the stamps themselves
> are too big.
>
> What *does* start a new classification is a change of dataset identity: `-N/--name`,
> `-s/--seed`, or the number of images under `--path`. Those are in the filename.

### 1-by-1 sequential tool

Detailed, one-object-at-a-time classification. Multiband FITS visualization (main band,
configurable RGB composites, individual color bands), configurable major and subclass
grading buttons with keyboard shortcuts, and optional per-object lookups in Legacy
Survey, PanSTARRS, ds9 and ESASky.

`-N` names the session (your name works well) and `-s` sets the shuffle: the same seed
always gives the same order.

```bash
python single_viewer.py -p PATH_TO_FILES -N NAME -s SEED_NUMBER
```

| Option | Default | Description |
| --- | --- | --- |
| `-h`, `--help` | | Show the help message and exit. |
| `-p`, `--path` | `Color_stamps_to_inspect` | Path to the images to inspect. |
| `-N`, `--name` | none | Name of the classifying session. |
| `-b`, `--main_band` | `VIS` if present | The default band: the panel shown first and pre-selected. `-b ''` for none: the first `-B` band takes its place. See [Data layout](#data-layout) for what is used when it is not given. |
| `-B`, `--color_bands` | those of `Y,J,H` present | Comma-separated bands to make individually selectable in the **Panels** dropdown. |
| `--rgb-composites` | those of `H,Y,I;H,J,Y` whose bands are present | Semicolon-separated `R,G,B` band-name triples. A composite's label is its comma-joined band list. |
| `--mef` | off | `--path` holds one multi-extension FITS file per object; bands are `EXTNAME`s. See [Multi-extension FITS](#multi-extension-fits). |
| `--classifications` | `A=1;B=2;C=3;X=4;I=5` | Classification buttons — see below. |
| `-s`, `--seed` | none | Seed used to shuffle the images. |
| `--legacysurvey` / `--no-legacysurvey` | **on** | Legacy Survey panel and downloads. Pass `--no-legacysurvey` to drop the panel entirely (the LS server can be unreliable). |
| `--ls-big-fov-residuals` / `--no-ls-big-fov-residuals` | off | Also pre-fetch the large-FoV Legacy Survey *residual* cutout, so **Large FoV** + **Residuals** shows a cached image instead of downloading on demand. Costs a fourth LS request per object. |
| `--reset-config` | off | Forget the saved preferences (colormap, scale, panels, toggles). Resume positions are kept. |
| `--reset-position` | off | Forget the resume position for this `--path`/`--name`/`--seed`, so the session restarts at the first image. Preferences are kept. |
| `--verbose` | off | Log to the terminal. |
| `--clean` | off | Empty the Legacy Survey and PanSTARRS cutout caches (`.qtstamp/cache/`, or under `--state-dir`) before starting. |
| `--classifications-dir` | `./Classifications` | Directory for the autosaved classification CSVs. Absolute, or relative to the directory you launch from; created if it doesn't exist. |

**Classification scheme.** `--classifications` takes one semicolon-separated string of
`MAJOR=KEY` or `MAJOR:SUB=KEY` entries. A bare `MAJOR=KEY` (or an empty `SUB`) makes a
major-class button; `MAJOR:SUB=KEY` makes a subclass button under that major, setting
both the classification and the subclassification in one click. A subclass shared by
several majors lists them comma-separated, `MAJOR1,MAJOR2:SUB=KEY`. Clicking it writes
nothing yet: it highlights those majors and dims every other classification button, and
the major you click next sets both fields, keeping the subclass (and auto-advances as
usual). Clicking the subclass again cancels. For example:

```
--classifications "A=1;B=2;C=3;X=4;I=5;X:Merger=a;A,B:Spiral=s"
```

**Panels and settings**

* The **Panels** dropdown assigns each panel — the main band, the RGB composites, the
  individual color bands, Legacy Survey and PanSTARRS — to one of 3 display rows
  (mutually exclusive across rows). A row appears as soon as you check a panel into it,
  and the title bar shows a live summary of what's in each one.
* The **Large FoV** checkbox, next to **Residuals**, switches the Legacy Survey and
  PanSTARRS panels to wider cutouts.
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
