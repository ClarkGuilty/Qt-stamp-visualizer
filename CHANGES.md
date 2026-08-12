# `unified_dev_with_lobby` — changes from `ERO_edition_2026`

> **Experimental branch:** this was vibecoded with Claude. Review before relying on it
> for real classification work.

This branch merges the best parts of `main` (rich external-tool integration, single-band)
into `ERO_edition_2026` (multiband FITS, on-the-fly VIS / H+Y+I / H+J+Y color composites),
plus a round of new features and fixes on top. Multiband FITS is the baseline going
forward — the JPG/PNG fallback pipeline has been removed from both viewers.

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

## Legacy Survey now off by default
* The LS server has proven unreliable, so the LS panel/checkbox/prefetch are now behind a
  new `--legacysurvey` flag. Pass it to get the previous behavior back.

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
