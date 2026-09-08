# `unified_dev_with_lobby` — changes from `ERO_edition_2026`

> **Experimental branch:** this was vibecoded with Claude. Review before relying on it
> for real classification work.

This branch merges the best parts of `main` (rich external-tool integration, single-band)
into `ERO_edition_2026` (multiband FITS, on-the-fly VIS / H+Y+I / H+J+Y color composites),
plus a round of new features and fixes on top. Multiband FITS is the baseline going
forward, but PNG/JPG input works again — see "PNG/JPG input, detected per band" below.

## Renamed: the two viewer scripts
* `mosaic_viewer_ERO_edition.py` is now `mosaic.py`, and
  `single_viewer_multiband_ERO_edition.py` is now `single_viewer.py`. The
  `_ERO_edition` suffixes described which survey the scripts were forked for, not
  what they do, and both have since grown well past that. The `qtstamp-mosaic` /
  `qtstamp-single` / `qtstamp-lobby` console commands are unchanged, so if you
  installed with `pip install -e .` nothing you type changes.

## New: Lobby launcher
* `lobby.py` is a GUI in front of the two tools: pick a data path, run mode
  (mosaic only, 1-by-1 only, or mosaic → 1-by-1 chained), mosaic layout, and
  the 1-by-1 tool's classification scheme (built in a table instead of
  writing the `--classifications` string by hand), then hit Run.
* In chained mode, Lobby launches the mosaic tool, auto-detects the
  classification CSV it wrote once you close it, extracts everything marked
  "positive" (via symlink or copy) into an output directory, and launches
  the 1-by-1 tool on that subset.
* An `Extract only...` action runs just the CSV-to-extracted-subset step on
  its own. Settings persist between runs in `.config_lobby.json`.
* The same launcher also runs without its window: `python lobby.py --no-gui`
  runs the configured workflow start to finish (chained extraction included),
  and `--print-command` just prints the viewer command line(s) it would launch
  and exits (chained mode prints both stages). Anything you don't pass on the
  command line is read from the saved lobby config, so the usual pattern is to
  set a run up once in the GUI and then script the reruns. Overrides:
  `-m/--mode`, `-p/--path`, `-o/--output`, `-N/--name`, `-s/--seed`
  (`--no-seed`), `-b/--main-band`, `-B/--color-bands`, `--rgb-composites`,
  `--classifications`, `--ncols`, `--nrows`, `--printname/--no-printname`,
  `--copy/--symlink`, and `--config` to read a different config file.

## Configurable bands and RGB composites (both tools)
* Bands are no longer hardcoded to VIS/Y/J/H/I. `--main_band` (default `VIS`)
  and `--color_bands` (default `Y,J,H`) are now real flags in both tools, and
  any directory under `--path` can be used as a band — the tool validates at
  startup that every band you reference actually has a matching directory,
  and if not, lists what directories are actually there.
* RGB composites are now fully configurable via `--rgb-composites`:
  semicolon-separated `R,G,B` band-name triples, e.g. `"H,Y,I;H,J,Y"`
  (the default, reproducing today's two composites). A composite no longer
  needs a separate short name — its label is just its own comma-joined
  member list, e.g. "H,Y,I" instead of the old, misleading "HYVIS" (the
  VIS-resampled-to-NISP-grid band isn't treated as anything special anymore,
  it's just another band a composite can reference). All three bands in a
  composite must have the exact same image dimensions (and ideally the same
  zero-point) — mismatched bands now produce a clear error instead of a raw
  crash.
* The mosaic tool can now show any individual band standalone (previously
  only VIS + the two composites were ever shown; Y/J/H were composite-only
  ingredients) — the "Panels" picker lists them as available, unchecked by
  default so the out-of-the-box view is unchanged.

## PNG/JPG input, detected per band
* Both viewers read PNG/JPG again, having been FITS-only on this branch. The
  main band's directory decides the object list: FITS if it has any, otherwise
  PNG/JPG.
* Format is detected per band *directory*, not per dataset, so one session can
  mix FITS and compressed bands freely. Files are matched across bands by stem
  (the name without its extension), so the same object can be `source_001.fits`
  in one band and `source_001.png` in another.
* RGB composites still need all three member bands in FITS — they're computed
  from pixel values, which a display-ready PNG no longer carries. A composite
  naming a non-FITS band is skipped at startup with a message saying which band
  disqualified it, instead of crashing. The lobby greys out the composites table
  entirely unless the path has at least 3 FITS bands.
* In the 1-by-1 tool, everything that needs real pixel values or WCS RA/Dec —
  ds9, Legacy Survey, PanSTARRS, ESASky, both pre-fetch toggles, and copy RA/Dec
  — is disabled up front for non-FITS input, rather than failing at the moment
  you use it. Scale/colormap controls likewise switch off in the mosaic when no
  FITS band is loaded.

## Configurable classes/subclasses (1-by-1 tool)
* Classification buttons are no longer hardcoded to A/B/C/X/Interesting. The new
  `--classifications` flag takes a single semicolon-separated string of
  `MAJOR=KEY` (major-only button) or `MAJOR:SUB=KEY` (subclass button, tied to
  a major) entries, e.g. `"A=1;B=2;C=3;X=4;I=5;X:Merger=a;X:Spiral=s"`. Clicking
  a subclass button sets both `classification` and `subclassification` in one
  click. Default matches today's 5-button scheme exactly.
* Revived the previously dead `subclassification` CSV column and its button row
  (second button row, below the major classes); old CSVs without that column
  are backfilled with `'Empty'` on load.

## New: PanSTARRS panel (1-by-1 tool)
* In-app cutout panel (fetched via the STScI PS1 image cutout API), alongside the existing
  Legacy Survey panel.
* "Open PanSTARRS" browser button, "Large FoV" toggle, and Pre-fetch support, matching the
  existing Legacy Survey controls.

## Legacy Survey behind its own flag (on by default)
* The LS panel/checkbox/prefetch now sit behind a `--legacysurvey` flag. It was
  briefly off by default, since the LS server has proven unreliable, but it is
  back on by default — pass `--no-legacysurvey` to drop the panel entirely.
* LS and PanSTARRS pre-fetching are independent toggles with their own threads,
  rather than one combined "Pre-fetch" setting.

## Reorganizable panel layout (1-by-1 tool)
* A "Panels" dropdown lets you assign VIS, the two composites, the individual Y/J/H bands,
  and PanSTARRS to any of 3 display rows (mutually exclusive across rows). Legacy Survey
  keeps its own dedicated checkbox (fetch on/off is separate from where it's placed).
  Row visibility is automatic — a row appears once you check a panel into it.
* The title bar now shows a live summary of what's in each row instead of a fixed label.

## Mosaic tool
* Restored `--resize` and `--gridsize` (alias for `--ncols`).
* The old "1/2/3 visible bands" count-only dropdown is now a proper panel picker — choose
  *which* bands/composites show, not just how many (now includes individual bands too,
  see "Configurable bands and RGB composites" above).
* Scale/colormap comboboxes now open on a click anywhere in the box, not just the arrow.
* Clicked file names now reach the lobby's log pane while you click, instead of arriving
  in one dump when mosaic exits (and being lost entirely if it was killed): the lobby
  launches its children with `PYTHONUNBUFFERED=1`, since Python block-buffers a piped
  stdout. Same fix for the 1-by-1 tool's status prints.
* `--printname` now defaults to on (`--no-printname` to silence it); the lobby passes
  the flag explicitly either way, so an unticked checkbox still means off.
* The lobby no longer omits `-B`/`--rgb-composites` when the selection is empty, which
  made both tools fall back to their `Y,J,H` / `H,Y,I;H,J,Y` defaults and exit with "band
  directory not found" before the window opened. The composites case hit every PNG/JPG
  dataset (the lobby disables composites there -- they need >=3 FITS bands) and any FITS
  set whose composite rows had been cleared by the missing-band pruning; the `-B` case hit
  anyone who unchecked every color band to view the main band alone.
* Both viewers now read `-B ''` as "no color bands" instead of one band named `''` -- the
  empty name slipped past the missing-directory check (it resolves to `--path` itself) and
  crashed later with `FileNotFoundError` while loading the first stamp.

## UI cleanup (1-by-1 tool)
* ds9 / Open LS / Open PanSTARRS / Open ESASky consolidated into one "Tools" dropdown.
* Scale and colormap are now dropdowns instead of button rows.
* Pre-fetch / Auto-next / Keyboard-shortcuts consolidated into one "Settings" dropdown.
* Net effect: the window can now be resized much smaller (minimum size roughly halved).

## Bug fixes along the way
* `Pre-fetch` no longer crashes the app when toggled off mid-download (was calling
  `QThread.terminate()`, now uses the class's own cooperative interrupt + waits for the
  thread on close).
* Legacy Survey / PanSTARRS panels now actually refresh when you page to a new object
  (previously stayed frozen on whichever object was showing when first enabled).
* Network failures (e.g. "no route to host") degrade gracefully to a placeholder instead
  of crashing the app.

## Restored from `main`
* `--clean`, `--verbose` CLI flags.
* `Legacy_survey/` cache directory (was missing entirely in `ERO_edition_2026`).

See the CLI `--help` on either tool for the full current argument list.
