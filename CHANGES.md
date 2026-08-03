# `unified_dev` — changes from `ERO_edition_2026`

> **Experimental branch:** this was vibecoded with Claude. Review before relying on it
> for real classification work.

This branch merges the best parts of `main` (rich external-tool integration, single-band)
into `ERO_edition_2026` (multiband FITS, on-the-fly VIS / H+Y+I / H+J+Y color composites),
plus a round of new features and fixes on top. Multiband FITS is the baseline going
forward — the JPG/PNG fallback pipeline has been removed from both viewers.

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
  *which* of VIS/HYI/HJY show, not just how many.
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
