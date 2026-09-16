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

## New: `--ls-big-fov-residuals` (1-by-1 tool)
* The Legacy Survey pre-fetch can now also warm the large-field-of-view
  *residual* cutout, so switching on both "Large FoV" and "Residuals" shows a
  cached image instead of waiting on a download. Off by default — it adds a
  fourth Legacy Survey request per object. This replaces a commented-out
  line in the pre-fetch loop that previously had to be edited in by hand.

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
* Network failures (e.g. "no route to host") degrade gracefully instead of crashing the
  app. (The placeholder image this originally used is gone -- see "Survey cutout caching
  rewritten" below for why it caused more trouble than it solved.)

## Survey cutout caching rewritten (1-by-1 tool)
A failed download used to be written into the cache as a black 66x66 JPEG, at the exact
path a successful one would occupy. Every reader treated that file's existence as a cache
hit, so a single missed download -- one coverage gap, one dropped connection, one slow
server -- blacked out those coordinates permanently, across sessions, with no retry. In
practice this poisoned most of a working cache: one local `PanSTARRS/` directory had 201
of 251 entries stuck on the placeholder, for coordinates the survey covers perfectly well.

* **Misses are no longer images.** A failure now writes a `<cutout>.jpg.miss` JSON marker
  next to where the image would go, recording the reason and the time. The cache lookup
  is tri-state: image on disk, known-miss, or go-fetch.
* **Misses expire.** A genuine coverage gap is trusted for 7 days (so panning back to the
  object doesn't re-query the survey every time); a transient failure -- timeout, 5xx,
  garbled response -- expires after 15 minutes, so the cutout is simply retried once the
  connection is back.
* **Failures are classified.** HTTP 400/404 and an empty `ps1filenames` answer mean "no
  coverage"; everything else is transient. Notably, a `ValueError` from `ps1filenames.py`
  answering with an HTML error page instead of a table is now caught -- it used to escape
  the `except (URLError, OSError)` handler, kill the pre-fetch thread outright, and leave
  every later object unfetched for the rest of the session.
* **Downloads are bounded and atomic.** `urlretrieve` (which accepts no timeout, and so
  could hang on a stalled server forever) is replaced by `urlopen(..., timeout=15)` into a
  temp file that is moved into place only once complete. An interrupted transfer can no
  longer leave a truncated image behind for the next run to serve as valid.
* **Blank Legacy Survey cutouts are rejected.** Out-of-footprint requests come back as a
  valid but completely flat JPEG, which was being cached as real data. A uniform image is
  now recorded as a coverage gap (a real cutout of empty sky is noise, never uniform).
* **Pre-existing placeholders are cleaned up on startup**, identified precisely: under
  1 KB, exactly 66x66, and entirely black. Real cutouts never match. `--clean` also drops
  miss markers now.
* **Stale callbacks can no longer touch the wrong panel.** A late *failure* from the
  previous object used to overwrite the current object's perfectly good image with "No
  data available"; a late *success* cleared the axes before checking whether it was still
  relevant. Both paths now check first and clear second.
* Pre-fetch toggles survive a pass that ends on its own (the config flag and the checkbox
  used to keep claiming a thread was running, and the next toggle called into a destroyed
  C++ object), and `FetchThread` keeps going when one object fails instead of ending the
  whole pass.
* The four near-duplicate copies of the download logic (two in `workers.py`, two in
  `single_viewer.py`) are now one implementation in `workers.py` -- that duplication is
  why the same bug existed in four places. Cache filenames are byte-for-byte unchanged,
  so already-downloaded cutouts still hit.

## Restored from `main`
* `--clean`, `--verbose` CLI flags.
* `Legacy_survey/` cache directory (was missing entirely in `ERO_edition_2026`).

## Dead-code cleanup
* Removed ~350 lines of commented-out code from `mosaic.py` and
  `single_viewer.py` — alternative `resizeEvent`/`sizeHint` implementations,
  layout experiments, debug `print()`s, superseded glob strings, and
  `# raise` lines left after exception handlers. Anything that survived only
  to feed those comments went with them (unused locals, a never-added
  `QSpacerItem`, a no-op `residual = residual`), along with three dead
  functions (`log_0`, `scale_val_percentile`, and both copies of
  `background_rms_image_old`) and 22 unused imports. No behavior changes.
* The ESASky Euclid ERO overlay (`&euclid_image=perseus`) went with them. It
  had been commented out with a note that the overlay is always centered on
  the same coordinate, so it was never generally useful.
* Deleted `extract_files_from_mosaic.py`. `extraction.py` — the module Lobby
  already uses — does the same job and more: symlinks as well as copies,
  understands both the mosaic tool's numeric class codes and the 1-by-1
  tool's string codes, and reports missing files. The old script was also
  listed in `pyproject.toml`'s `py-modules` despite parsing `sys.argv` at
  import time.

## Fixed: autosave crash outside the repo root
Both viewers built the classification CSV path relative to the current
working directory and never created it, so running either one from anywhere
but the repo checkout — exactly what the installed `qtstamp-*` console
scripts invite — raised `FileNotFoundError` on the first autosave, after the
user had already classified a page.

* All three tools now take `--classifications-dir PATH` (absolute, or
  relative to wherever you launch from). It defaults to `./Classifications`
  and is created automatically if it doesn't exist.
* `lobby.py` used to resolve `Classifications/` against its own install
  directory (`REPO_ROOT`) rather than the CWD, so it silently disagreed with
  the viewers whenever a viewer was launched from somewhere else. It now
  resolves against its own CWD, the same rule the viewers use, and always
  passes the resolved absolute path down to the mosaic/1-by-1 processes it
  launches — so launching the lobby and launching a viewer directly, from
  the same directory, land on the same `Classifications/` and can resume
  each other's CSVs.
  **Behavior change:** a lobby launched from outside the checkout now
  writes to its own CWD instead of `<repo>/Classifications`. Launching from
  inside the checkout, as the README documents, is unchanged.

## Fixed: crash when classifying a PNG/JPG stamp (1-by-1 tool)
Classifying any non-FITS object raised `AttributeError: 'ApplicationWindow'
object has no attribute 'image_pixel_size'`. `classify()` recorded the stamp's
pixel scale and dimension unconditionally, but both came from the WCS, which
only the FITS path reads — so the grade was lost and the autosave never
happened. The `ra`/`dec` columns right above them were already guarded.

* `pixel_size` (arcsec/pixel) genuinely needs a WCS, so it is now written only
  for FITS input and left empty for PNG/JPG, like `ra`/`dec`.
* `image_dim` does not: the stamp is loaded whatever its format. It is now read
  off the loaded image for every filetype, so PNG/JPG rows carry a real size
  instead of a blank.
* `pixel_size` was also missing from the column list used when a classification
  CSV is created from scratch, so it only showed up — appended after `time` —
  once a FITS object had been graded. It is declared with the others now.

## Fixed: resuming at the wrong place (both viewers)
Both viewers kept *everything* in one config file per tool — `.config.json`
and `.config_mosaic.json` — shared across every invocation regardless of
`--path`, `--seed` or grid shape, and invalidated the saved position only
when the `--name` string changed. Since `--name` defaults to empty on both
runs when omitted, opening a *different* dataset under no name at all passed
that check and silently resumed whatever position was last saved, landing you
mid-deck in a dataset you had never classified.

The underlying problem was that one file held two kinds of state with
different lifetimes, guarded by one weak key. They are now split three ways:

* **Identity** — `(tool, realpath(path), name, seed)`. Note what's *not* in
  there: grid shape and file count.
* **Session state**, per identity, in one `sessions.json`: where you are, the
  CSV it belongs to, when it was last opened. Keyed by a short hash, with the
  identity stored in plain text beside it so the file stays readable.
* **Preferences**, global per tool, in `.preferences_single.json` /
  `.preferences_mosaic.json`: colormap, scale, autonext, panel rows, prefetch
  toggles. These follow *you*, so changing the seed no longer resets your
  colormap — the reason simply keying the old file on the session identity
  would have been the wrong fix.

**Positions are stored as filenames, not indices.** That one decision pays for
itself several times over:

* The mosaic's grid shape drops out of the session identity entirely. The page
  is re-derived from where that same object now falls, so switching a 100-image
  deck from 5×8 to 5×2 reopens on page 4 instead of page 0, and switching back
  returns to page 1. The old code reset to page 0 on *any* shape change,
  in-place, destroying the position it was leaving. (The grid shape had to leave
  the CSV filename as well before this worked in every case — see the next
  entry.)
* Adding or removing files no longer invalidates the position.
* A filename that has since disappeared is a *defined* fallback (start at 0)
  rather than an out-of-range index. That removed the stale-counter `assert`
  in the 1-by-1 tool's `classify()` (and its `#TODO handling this possibility
  better`), and an off-by-one where a saved counter exactly equal to the file
  count cleared `counter > len(listimage)` and then indexed off the end —
  reachable any time the dataset shrank between runs. The mosaic had the same
  off-by-one in its page bounds, where `update_grid`'s `try:` absorbed it into
  a gridful of blank buttons instead of a crash.

**Cross-checked against the CSV.** The session entry records which
classification CSV the position belongs to; if `obtain_df()` later resolves to
a different one, the two disagree about which session is open, so the position
is stale by definition and resets to 0. This replaces the partial, accidental
invalidation that used to fire on only one of the several paths that needed it.

`seed` deliberately *stays* in the identity: a new seed is a new pass in a new
order, and resuming on the same object would strand you mid-deck with
unclassified stamps on both sides.

* New `state.py` holds all of it — `SessionId`, `session_key`, `load_store`,
  `save_session`, `resolve_position`, `load_preferences`, `migrate_legacy_config`.
  Pure functions, no Qt, covered by `tests/test_state.py` (28 tests, runnable
  with pytest or directly as `python tests/test_state.py`).
* Store writes are read-modify-write plus an atomic `os.replace` from a temp
  file in the same directory — the two viewers wrote to separate files before,
  so consolidating them introduced a clobber risk that didn't exist. The store
  is pruned to the 50 most recently opened sessions.
* **Migration is automatic.** The first run after this change splits each
  existing `.config*.json` into preferences plus one session entry, then keeps
  the original as `.config.json.bak` / `.config_mosaic.json.bak` rather than
  deleting it. The old file only ever guarded its position with a `--name`
  check, so that's all the migration honours: an old position is carried over
  only if the name still matches (and, for the mosaic, the grid shape it was
  saved under).
* `--reset-config` is now well-defined against the split — it forgets
  *preferences* only — and the new `--reset-position` forgets the resume
  position for the current `--path`/`--name`/`--seed`. Both viewers take both
  flags; the mosaic had neither before.

`lobby.py` has the same class of bug one level up (`scheme_rows`, `main_band`,
`mosaic_ncols`/`nrows` all live in one global `.config_lobby.json` and persist
across a change of session name). That's a real change to the lobby's UI rather
than a correctness fix, so it's sequenced separately — see PLAN.md.

## Classification autosaves are now crash-safe
The classification CSV is the only irreplaceable thing either tool produces, and
both rewrite it *in full* on every grade — a page at a time in the mosaic, an
object at a time in the 1-by-1 tool. That is hundreds of full-file rewrites in a
session, over a file that may hold hours of work.

Every one of those rewrites was a `DataFrame.to_csv` straight onto the live
path. `to_csv` truncates the target and streams into it, so the file spends a
moment empty on every single click, and an interruption in that window — a
crash, a kill, a full disk, a closed laptop — leaves a truncated or empty CSV
where the completed classification used to be.

Measured, killing the writing process with `SIGKILL` mid-save, ten times, on a
20,000-row file:

| Write path | Intact | Lost or truncated |
| --- | --- | --- |
| `df.to_csv(path)` (before) | 0/10 | **10/10** — eight unreadable, two cut to 9,463 rows |
| `ClassificationWriter` (now) | **10/10** | 0/10 |

Saves now go through `imaging.ClassificationWriter`, which writes a temp file in
the same directory, `fsync`s it, and `os.replace`s it over the target. The rename
is atomic, so a reader sees either the whole previous file or the whole new one,
and an interrupted save leaves the previous one exactly as it was.

**A save also never overwrites a file this process did not write.** The writer
remembers the target's fingerprint after each save; if it changed underneath —
two viewers open on the same session being the realistic way that happens — the
other process's grades would otherwise be silently replaced on the next click.
The same is true of a target that *appeared* underneath us, which is what
running the same command twice on a session nobody has started yet looks like:
both viewers begin with no fingerprint at all, so the check has to compare
against what is on disk right now rather than only against our own last save.
Either way the other file is left alone and this session continues in the next
free `-simultaneous_N` name, so both sets of work survive to be reconciled
afterwards. The session entry records whichever name was actually written.

**A save does not change how the file is shared.** `os.replace` carries the
temp file's permissions onto the target, and the obvious way to make a temp
file — `tempfile.mkstemp` — creates it at 0600. Taken together that meant every
autosave silently rewrote the CSV as owner-only: a classification sitting in a
group directory lost its group read on the first grade of the session, with no
error on either side. So the temp file is now opened by hand with `os.open(...,
0o666)`, letting the kernel apply the umask exactly as it would for any other
file the tools write, and an *existing* target's mode is copied onto the temp
before the rename. A new CSV therefore comes out exactly as `df.to_csv(path)`
would have made it, and an existing one keeps whatever mode it had — in both
directions, so a deliberately private file is not widened either. (Ownership,
ACLs and xattrs cannot survive a rename and are not preserved; in the usual
setgid shared directory the group is inherited by the temp file anyway.)

Reading the umask back to reproduce it would have been the other way to do
this, and is not safe here: `os.umask` is read-modify-write and process-global,
and both viewers have worker threads creating files of their own.

The mechanism is `imaging.atomic_write`, with `atomic_to_csv` a two-line wrapper
over it. `state.py` writes `sessions.json` and the two `.preferences_*.json`
through the same helper, so the whole of what the tools persist follows one rule
rather than two.

Covered by `tests/test_classification_writer.py` (21 tests: completeness, a
failed write leaving the previous file byte-identical, no scratch files left
behind, fork-on-foreign-write, no re-forking afterwards, a file that appeared
since startup forking to `-simultaneous_N` instead of clobbering it, a third
viewer taking the next `-simultaneous_N`, a deleted target being recreated
rather than forked, both markers' numbering and de-stacking rules, a legacy
`-(N)` name forking to a named marker, the forks surviving the viewers'
seedless-discovery glob, and the four permission properties above — asserted against a probe file written with a plain
`open()` rather than a literal 0644, so they hold under any umask) and
`tests/test_state.py` (33 tests, three of them the same rule for the store).

## Fork names say which collision happened, and have no parentheses
A classification CSV is named after the dataset, so two runs can legitimately
want the same name. Two quite different things cause that, and both used to be
spelled `-(N)`:

* the name is taken by a **different dataset** — same `--name` (or none) and
  the same image count over different files;
* **another viewer is grading this same session** right now, and the writer
  refuses to overwrite its file (see above).

A bare number says nothing about which, so the directory afterwards is a
guessing game. Each case now names itself: `-new_dataset_N` and
`-simultaneous_N`, counting up as the situation repeats.

Parentheses are gone with them. `-(1)` is a syntax error in bash and zsh, so
every one of these files had to be quoted or escaped before it could be copied,
moved or passed to another tool — a papercut on exactly the files a user reaches
for by hand when reconciling two of them.

Files already named `-(N)` are still found, read and resumed, and forking one
lands on a named marker rather than stacking (`c-(1).csv` → `c-simultaneous_1.csv`).
Nothing on disk is renamed; the suffix is simply not written any more.

One helper now covers both cases — `imaging.next_free_csv_path(path, marker)`,
which scans for the first free name. That retired `find_filename_iteration`,
which parsed the number out of the name it had just read and added one to it,
and so could hand back a name that was already taken if the numbering had gaps.
The `-` introducing the marker is load-bearing, by the way: both viewers glob
`{base}-*.csv` minus `{base}_*.csv` to find a seedless dataset's files, so a
marker spelled `_new_dataset_1` would read as a seed and hide the fork from
relaunch and from the lobby's auto-detect.

## The mosaic's grid shape is no longer part of the CSV filename
Changing `--ncols`/`--nrows` used to start a *new* classification, leaving the
grades you had already made behind in the old file. The grid shape is a display
choice — how many stamps fit comfortably on your screen, changeable at any time,
and `--minimum_size` exists for exactly that — so it says nothing about *which*
objects are being classified and has no business forking the file that records
the grades.

The mosaic's CSV is now named the same way `single_viewer.py`'s always was,
which never had a grid to encode:

    classification_mosaic_autosave_{name}_{n_images}[_{seed}]

Dataset identity — name, image count, seed — and nothing else. Reshaping the grid
now keeps the same file, the grades in it, and your place in the deck.

This also removes a long-standing asymmetry that made the old behaviour hard to
predict: the shape only *partly* made it into the name. The unseeded branch wrote
`..._{ncols}_99`, dropping `nrows` entirely, because the grid was square back when
the scheme was written and one number described it. The seeded branch wrote
`..._{ncols}_{nrows}_{seed}`. So changing `--nrows` alone forked the file when a
seed was set and silently reused it when one wasn't — the single case where the
resume-position feature above appeared to work. `lobby.py` carried a copy of the
same formula, quirk included, to guess which CSV the mosaic had just written;
it now shares the simpler rule.

**This breaks compatibility with classifications started before it, deliberately.**
A pre-existing CSV has the grid shape in its name and will not be found; the
session starts a new file. Continuity is guaranteed from this release forward,
not backward — no migration code is carried for the old layout. If you have a
classification in flight under the old naming, finish it on the previous
revision or rename the file yourself to
`classification_mosaic_autosave_{name}_{n_images}[_{seed}].csv`.

## Fixed: `page`/`grid_pos` describing two grids at once (mosaic tool)
The mosaic records which page and cell each stamp was graded in. Both are only
meaningful for the grid shape in force at the time, but they were computed once,
when a classification CSV was first created, and afterwards only rewritten for
the cells actually rendered.

Whenever a CSV is reopened under a different shape, those two columns silently
end up describing both grids at once. Grading a 12-object deck at 2×3 (6 per
page) and reopening it at 2×2 (4 per page) left eight objects claiming page 1 on
a four-cell page, with `grid_pos` values of 4 and 5 that no 4-cell grid can
produce. Only the pages the user happened to visit were correct.

Under the old naming this was reachable only in the one case where a reshape
kept the same file (`--nrows` alone, no seed). Taking the grid shape out of the
filename — see the entry above — makes *every* reshape reuse the file, so fixing
this was a precondition for that change rather than a nicety.

Both columns are now recomputed from the current grid when an existing CSV is
read, exactly as they already were when one is created, so the file always
describes the shape it is being used with.

The grades themselves were never affected, and nothing downstream reads either
column — `extraction.py` uses only `file_name` and `classification`. This was a
long-standing bug, not a regression from the session-state work above.

## Background markers are drawn, not loaded (mosaic tool)
`.background.png`, `.background_interesting.png` and `.backgrounddark.png` —
a white "L", a white "I", and a solid black square, used to mark a stamp as
a lens candidate / interesting / deactivated — were loaded from three
CWD-relative dotfiles, so outside the repo root they silently resolved to
nothing: no error, just a missing pixmap. They're now drawn at runtime with
`QPainter` instead, so the three tracked PNGs are gone.

While touching this code, fixed a latent bug in `MiniMosaics.deactivate()`:
it passed a bare path string into a method expecting a list, which iterated
over it character-by-character. That happened to produce harmless null
pixmaps with path strings, so it never showed a symptom, but would have
raised `TypeError` outright with the new `QPixmap` objects.

## Shared FITS reading, plus multi-extension FITS (`--mef`)
Both viewers used to hand-roll their own `astropy.io.fits` access independently:
three separate copies of "glob the main band's directory for `*.fits`" (mosaic,
single_viewer, and a third one inside single_viewer's prefetch `FetchThread`),
always reading HDU 0, with single_viewer's `load_fits` never closing the
`HDUList` it opened, and `memmap` handled inconsistently between the two
(mosaic passed `memmap=False` explicitly; single_viewer's main read path left
it at astropy's own default, `True`). All of that now lives in one new module,
`fits_io.py`, that both tools and the prefetch thread call through instead of
touching `astropy` directly — `memmap=False` and closing the file via `with`
are now consistent everywhere. `detect_band_filetype`/`find_band_file` moved
out of `imaging.py` (now astropy-free) into `fits_io.py`, since resolving
"where does this band's data live" is one job whether the answer is a
directory or, as of this change, an HDU extension.

New capability that fell out of unifying it: `--mef` (both tools). Point
`--path` at a directory of multi-extension FITS files — one file per object,
holding every band as an HDU extension (identified by `EXTNAME`) — instead of
one subdirectory per band. `-b`/`-B`/`--rgb-composites` work exactly as
before, except a band name now has to match an extension found in the first
file under `--path` rather than a subdirectory name; that mapping is
discovered automatically (open one sample file, read its extensions) and is
never typed in. Lobby auto-detects an MEF dataset from the path's contents
(`*.fits` files directly under it, not only subdirectories) and lists its
extensions in the band picker exactly the way it lists subdirectories today,
so `--mef` gets appended to the launched viewer's command line without the
user ever naming it.

Mixing the two schemes in one run — some bands from directories, one band an
extension of another band's file — is out of scope: `--mef` applies to the
whole run, and every object's file is assumed to share the same extension
layout as the first one found. New tests: `tests/test_fits_io.py`.

See the CLI `--help` on either tool for the full current argument list.
