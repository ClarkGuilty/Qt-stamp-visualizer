# Known bugs — fixed

Archive. Items move here from [BUGS.md](BUGS.md) once they are fixed *and*
verified, keeping their original number and their fix writeup. Check here before
re-reporting something that looks familiar, and before re-litigating a fix.

**Numbers are never reused.** `CHANGES.md`, `PLAN.md`, `STATUS.md` and two source
comments ([`state.py`](state.py), [`paths.py`](paths.py)) cite these by number,
so item 1 here is the item 1 those references mean.

Found reviewing the working-tree changes against `715261a` (session-state split,
atomic autosaves, `--classifications-dir`, drawn background markers) on
2026-09-14. Verified by running both viewers headless (`QT_QPA_PLATFORM=offscreen`,
`MPLBACKEND=Agg`) against synthetic FITS and PNG datasets, from a directory
outside the checkout.

"New" = introduced by the changes on this branch since `715261a`. "Pre-existing"
= older, but adjacent to them, or newly reachable. Two items found in this pass
(the permissions narrowing and the foreign-write guard's hole) were fixed
before commit and never got an item here; see CHANGES.md for how.

Not bugs, verified working: both viewers run standalone from any CWD; PNG/JPG
classification; a change of `--name` or `--seed` restarting at the first
image/page in both tools; a mosaic reshape keeping the CSV and re-deriving the
page; legacy `.config*.json` migration; `--reset-config` / `--reset-position`;
fork-on-foreign-write; `tests/` (48 tests).

---

## 1. The lobby launches its children with `cwd=REPO_ROOT` — PRE-EXISTING, now inconsistent, FIXED 2026-09-17

[`lobby.py:819`](lobby.py#L819), [`lobby.py:844`](lobby.py#L844),
[`lobby.py:1111`](lobby.py#L1111).

`--classifications-dir` is now resolved to an absolute path against the lobby's
own CWD and passed down, but `-p`/`-o` are passed through verbatim, and the
child runs in `REPO_ROOT`. Three consequences:

* **A relative `--path` breaks.** Reproduced from a directory containing
  `data/`:

  ```
  $ lobby.py --no-gui -m mosaic -p data -b VIS
  Launching: .../mosaic.py -p data ... --classifications-dir /abs/path/Classifications
  Band directories not found under data: ...
  ```

  The GUI usually escapes this because Browse... returns absolute paths, but a
  relative `data_path` in `.config_lobby.json` or on the CLI hits it.

* **Resume positions do not carry between launch paths.** `sessions.json` and
  `.preferences_*.json` are CWD-relative (`state.DEFAULT_CONFIG_DIR = os.curdir`),
  so a lobby-launched viewer writes them to `REPO_ROOT` and a directly-launched
  one writes them to the user's CWD. The CSV is shared (absolute path); the
  position is not. CHANGES.md claims otherwise: "launching the lobby and
  launching a viewer directly, from the same directory, land on the same
  `Classifications/` and can resume each other's CSVs."

* **Under `pip install` / `uvx`, `REPO_ROOT` is site-packages.** The child then
  tries to write `sessions.json`, `.preferences_*.json`, `./.temp/` and
  `./Legacy_survey/` there. Unhandled exception on the first save if it is not
  writable.

Fix: resolve `path` and `output` to absolute in the lobby before building argv
(the same rule `--classifications-dir` already follows), and either stop setting
`cwd=REPO_ROOT` on the children or pass the state directory down explicitly.
PLAN.md item 2b (`paths.py`) subsumes the second half.

Fixed by PLAN.md 2b-ii/2b-iii, landed together (2b-iii had to land with 2b-ii or
the caches would just scatter across whatever CWD each launch happened to use,
instead of being fixed). `-p`/`-o` are now resolved to absolute in the lobby
before argv is built, in all three launch paths (`on_run_clicked`,
`on_stage1_finished`, `run_headless`), and the three
`cwd=REPO_ROOT`/`setWorkingDirectory(REPO_ROOT)` call sites are gone, so a
launched viewer inherits the lobby's own CWD instead. That alone only fixes the
first bullet, though -- the second and third needed 2b-iii's move of user/machine
state off the CWD entirely: preferences, the lobby's own config/predefined-configs,
and the `Legacy_survey`/`PanSTARRS`/`.temp` caches now live under a fixed
per-user config/cache directory (`paths.user_config_dir()`/`user_cache_dir()`,
overridable via `QTSTAMP_CONFIG_DIR`/`QTSTAMP_CACHE_DIR` for tests), migrated
once from wherever they used to sit (CWD, or `REPO_ROOT` for the lobby's own
config) if found there. Verified: this bug's own relative-path repro no longer
errors; the mosaic runs from a non-writable CWD without `PermissionError:
'./.temp'`; a lobby-launched viewer and a directly-launched one from the same
directory now share the same resume position, as CHANGES.md already claimed.

**Amended 2026-09-18 (PLAN.md 2b-v).** The per-user config/cache directory was
the wrong *default*, and is now the fallback instead: settings, presets and
caches live in `./.qtstamp` (the directory the tool was launched from), read
through a state-dir -> per-user -> packaged-defaults chain and written only to
the state dir. This bug stays fixed either way -- the non-writable-CWD repro was
re-run under the new scheme and still passes, because the failure mode it
describes is now handled by giving up on the write rather than by relocating it
(`paths.ensure_dir`, `state._write_json` returning False). `--state-dir`/
`--global` opt back into a shared location.

## 2. `Classifications/README` still documents the old mosaic filenames — NEW, FIXED 2026-09-21

[`Classifications/README`](Classifications/README).

The paragraph about `--classifications-dir` was added, but the filename table
immediately below it still says:

```
  no seed:   classification_mosaic_autosave_{name}_{n_stamps}_{ncols}_99.csv
  with seed: classification_mosaic_autosave_{name}_{n_stamps}_{ncols}_{nrows}_{seed}.csv
```

The grid shape is exactly what this change removed from the name. Should read
`classification_mosaic_autosave_{name}_{n_stamps}[_{seed}].csv`. README.md was
updated correctly; this file was missed.

Fixed. The table now reads `classification_mosaic_autosave_{name}_{n_stamps}.csv`
/ `..._{n_stamps}_{seed}.csv`, with a short paragraph saying *why* the shape is
absent (it is a display choice; reshaping keeps the CSV and recomputes
`page`/`grid_pos`) so the next reshape doesn't look like a bug. Also added the
`sessions.json` line the item-3 fix made necessary -- that file now lives in this
directory too, and the README listed only CSVs. Verified by building a
`MosaicVisualizer` headless against `test_stamps_png_jpg/` from a scratch
directory outside the checkout and reading back `df_name`:
`classification_mosaic_autosave_readmecheck_5.csv` unseeded and
`..._readmecheck_5_7.csv` with `-s 7`, both matching the corrected table.

## 3. The new state files are not gitignored — NEW, FIXED 2026-09-18

[`.gitignore`](.gitignore).

`sessions.json`, `.preferences_single.json`, `.preferences_mosaic.json`,
`.config.json.bak`, `.config_mosaic.json.bak`. The existing `.config*.json`
pattern does not match the `.bak` names, and nothing matches the other three.
First run from the repo root produces five untracked files.

`sessions.json` is also the only one of the set that is not a dotfile, so it is
visible clutter in whatever directory the user launches from.

**Fixed 2026-09-18 (2b-v).** `sessions.json` moved inside the classifications
directory (2b-i) and everything else moved inside `.qtstamp/` (2b-v), so the
whole set is now two ignore entries rather than six, and only `Classifications/`
is visible in a working directory. `.gitignore` gained `.qtstamp/` and keeps the
pre-2b-v patterns for old checkouts. The `.bak` names are still unmatched by
`.config*.json`, but they only appear in a directory that once ran a pre-split
version, so they are left as-is.

## 4. The 1-by-1 "Go to" dialog can crash the tool — PRE-EXISTING, FIXED 2026-09-21

[`single_viewer.py:1542`](single_viewer.py#L1542).

```python
i, ok = QtWidgets.QInputDialog.getInt(self, 'Visual inspection', '',
                                      self.counter + 1, 1, self.COUNTER_MAX + 1)
```

The dialog's maximum is one past the end. Entering 13 of 12 sets `counter = 12`
and `go_to_counter_page` raises `IndexError` on `self.listimage[12]`
(reproduced). Should be `self.COUNTER_MAX`.

Same off-by-one family as the mosaic's `goto`, which this change did fix — and
the old `counter > len(listimage)` startup clamp that used to soften the related
case was removed along with `config_dict['counter']`.

Fixed: the dialog's inclusive maximum is now `self.COUNTER_MAX`, with a comment
saying why (`COUNTER_MAX` is `len(listimage)`, a count rather than a last index,
while the dialog is 1-based and `counter` is 0-based -- the mismatch that made
`+1` look right). Verified headless on a 5-image dataset by driving the real
`goto()` slot with `QInputDialog.getInt` stubbed: the bounds handed to the
dialog are now `min=1 max=5`, entering 5 lands on index 4, and the old
maximum's value (6) still raises `IndexError` through the same path, so the
check is a real one. No unit test added -- the bound lives in a Qt slot, and the
repo's pattern is Qt-free tests over extracted logic rather than driving Qt;
a one-line bound does not justify the extraction. Suite still 134 passing.

## 5. The mosaic's page warning is 0-based, the widget is 1-based — NEW, FIXED 2026-09-21

[`mosaic.py:643`](mosaic.py#L643) says `'Pages go from 0 to {PAGE_MAX-1}'` while
`LabelledIntField` displays and accepts 1-based page numbers ("Page 1 / 3").
Should be `1 to {PAGE_MAX}`. The `>= PAGE_MAX` bound fixed just above it is
correct.

**This entry was wrong about "cosmetic, and currently unreachable behind
`QIntValidator(1, PAGE_MAX)`".** A `QLineEdit` refuses only input its validator
calls *Invalid*; *Intermediate* input stays in the field, because you have to be
able to type "1" on the way to "12". Checked against PySide6 with
`QIntValidator(1, 3)`: `'0'`, `'4'`, `'9'` and `''` are all Intermediate (only
`'04'`, `'-1'` and `'12'` are Invalid). Typing `0` into the real widget and
reading it back gives `getValue() == -1`, so the branch is reached by ordinary
typing, and the user is told pages "go from 0 to 2" in a widget that will not
accept 0 and labels the same page 1.

Found while fixing it: **an empty field crashed the tool.** Clear the box and
press Enter and `getValue()`'s `int('')` raises `ValueError` straight out of the
`goto` slot, which caught nothing. Same reachability argument -- `''` is
Intermediate, so the field can be left empty.

Fixed: the message is now `Pages go from 1 to {PAGE_MAX}` (shared by both
rejection branches, so they cannot drift apart again), `getValue()` is wrapped
for the empty-field case, and that path restores the number to the blank box
rather than leaving the widget reading "Page  / 3". The stray
`print("page: ", self.PAGE_MAX)` in the first branch went with it. The two
numbering schemes now have a comment saying which is which.

Verified headless on a 5-image, 2-per-page deck (3 pages), driving the real slot
through `QTest` keystrokes: `3` and `1` navigate, `4` warns "There are only 3
pages.", `0` warns "Pages go from 1 to 3." (was "from 0 to 2"), and an emptied
field warns the same instead of raising `ValueError`, with the box restored to
`1`. Suite still 134 passing.

## 6. `new_class` can be unbound on a modified click — PRE-EXISTING, FIXED 2026-09-21

[`mosaic.py:318`](mosaic.py#L318), `MiniMosaics.mousePressEvent`.

On an unclassified stamp, the branch assigns `new_class` only for
`Qt.ControlModifier`, `Qt.ShiftModifier` or `Qt.NoModifier`. Ctrl+Shift+click,
Alt+click or a Meta-modified click falls through all of them and reaches
`self.update_df_func(event, self.i, new_class)` with `new_class` unbound →
`UnboundLocalError`. Grades already on disk are unaffected.

Fixed by returning early on an unrecognised modifier, which is the no-op the
suggested `new_class = self.is_a_candidate` default would have produced, without
writing an unchanged grade back through `update_df_func` (it re-saves the CSV
and re-stamps the page's `time` column, so a stray Alt+click would have counted
as inspection time). The `elif` chain now has an explicit `else` with a comment
saying why the combinations reach it -- `modifiers` is compared for equality,
not tested bitwise.

Verified headless by driving the real `mousePressEvent` with real
`QMouseEvent`s over all 7 modifier states (none, Ctrl, Shift, Ctrl+Shift, Alt,
Meta, Ctrl+Alt) x all 3 starting classes, against a copy of the module with the
fix reversed as a control: the control raises `UnboundLocalError` on exactly the
4 unrecognised-modifier clicks on an unclassified stamp, the fixed module raises
nothing, records no `update_df_func` call for those 4, and is unchanged on the
other 17. No unit test added -- this is a Qt event handler, and the repo's
pattern is Qt-free tests over extracted logic. Suite still 134 passing.


## 7. `get_ra_dec` swapped the WCS axis order — PRE-EXISTING, FIXED 2026-09-16

[`fits_io.py:200`](fits_io.py#L200).

```python
sky = w.pixel_to_world_values([w.array_shape[0] // 2], [w.array_shape[1] // 2])
```

`WCS.array_shape` is numpy order `(ny, nx)`; `pixel_to_world_values` takes
WCS-axis order `(x, y)`. The two indices were therefore crossed, so the "centre
of the image" handed to the cutout downloaders was the centre only for square
stamps. Verified against astropy on a synthetic 40x100 TAN header with 0.2"
pixels: the old expression returned `(150.0017, 2.0017)` where the true centre
is `(150.0, 2.0)` — about 6" off in each axis, comfortably enough to mis-centre
a Legacy Survey or PanSTARRS cutout.

Fixed, with a non-square regression test in
[`tests/test_fits_io.py`](tests/test_fits_io.py). Found incidentally during the
item 2b-i paths work, not by looking for it — it predates `fits_io.py` and came
across in the unification from `single_viewer.py`.


## 8. `_empty_store()` drops `legacy_imports` — PRE-EXISTING, FIXED 2026-09-21

[`state.py`](state.py). Found by `/code-review` during 2b-v, 2026-09-18, not
introduced by it.

`load_store` falls back to `_empty_store()` when `sessions.json` is missing *or
unparseable*, and `_empty_store()` returned only `version`/`sessions`. The
once-only guard `migrate_sessions_store` writes into `store['legacy_imports']`
therefore disappeared along with a corrupted store, so the next startup
re-imported the legacy CWD `sessions.json` and could resurrect a position the
user deliberately cleared with `--reset-position`.

Fixed by making the two cases distinguishable. `load_store` now wraps a new
`_load_store`, which returns `(store, intact)` -- `intact` is False only when a
store file *exists* and will not parse, which is exactly the case where the
empty store handed back has silently replaced a record nobody can see any more.
`migrate_sessions_store` treats that as "already imported": it does not import,
and it writes the mark straight back out so the next startup does not ask the
same unanswerable question. The two ways of being wrong are not symmetric --
skipping an import the user never had costs them nothing visible, repeating one
puts back a position they cleared on purpose. The unreadable file is kept as
`sessions.json.corrupt` before it is replaced, since this is the point where it
stops being recoverable (the next `save_session` would have overwritten it
regardless). `_empty_store()` also carries `legacy_imports` now, and
`_load_store` normalises a wrong-typed one, so the shape is the same whichever
branch produced it and the `setdefault` at the use site is no longer
load-bearing.

Reproduced first, at both levels. Unit: import, `forget_session`, corrupt the
store, re-import -- old code returned True and put `a.fits` back, new code
returns False and the position stays cleared. End to end with the real 1-by-1
viewer, headless from a scratch directory outside the checkout, against
`test_stamps_png_jpg/`: a planted pre-2b-i CWD `sessions.json` pointing at
`source_005.png` was imported on the first startup (resumed at `counter=4`),
cleared by `--reset-position`, and after truncating
`Classifications/sessions.json` to `{ truncated` the next two startups both
resumed at `counter=0` with `sessions.json.corrupt` beside the rewritten store,
its `legacy_imports` holding the mark and `sessions` empty. The legacy CWD file
was never moved or deleted throughout, as that migration requires. Four
regression tests in [`tests/test_state.py`](tests/test_state.py) cover the
resurrection, the mark's durability, the ordinary first-run import into a
*missing* store (the guard must not swallow that), the store shape, and
`save_session` round-tripping the record.

## 9. `migrate_legacy_config`'s `classifications_dir=None` default is unusable — PRE-EXISTING, FIXED 2026-09-21

[`state.py`](state.py). Same review pass.

The default reached `sessions_path(None)` and raised `TypeError: expected str,
bytes or os.PathLike object, not NoneType` three frames down. Both real call
sites passed it explicitly, so this was a dead default rather than a live crash.

Made required, which is what the module already says it should be -- the note
above `_config_dir` states that session state has no default anchor, on
purpose, and this was the one signature contradicting it. It moved ahead of
`csv` in the parameter list; every call site ([`mosaic.py:518`](mosaic.py#L518),
[`single_viewer.py:478`](single_viewer.py#L478), and all six in
[`tests/test_state.py`](tests/test_state.py)) already passed it by keyword, so
nothing else changed. Omitting it is now a `TypeError` naming the missing
argument, at the call.

Verified by running the mosaic headless from a scratch directory outside the
checkout with both migrations pending at once: a planted `.config_mosaic.json`
(`scale: sqrt`) and a planted pre-2b-i CWD `sessions.json`. Both ran -- the
legacy config became `.config_mosaic.json.bak` with `scale: sqrt` in
`.qtstamp/preferences_mosaic.json`, and the session entry was imported and
rekeyed to `path: ../stamps` with the `legacy_imports` mark. The 1-by-1 viewer's
call site is covered by the item 8 runs above. One regression test asserts the
argument is required and that nothing is half-migrated when it is missing.

## 10. `obtain_df`'s missing length check crashes the mosaic — PRE-EXISTING, FIXED 2026-09-21

[`mosaic.py`](mosaic.py). Reported as minor ("relies on deprecated behaviour"),
and that was too generous: on the installed NumPy it is a hard crash at startup.

[`single_viewer.py`](single_viewer.py) guards its dataset check with
`len(self.listimage) == len(df)`; the mosaic went straight to
`np.all(self.listimage == df['file_name'].values)`. The claim that NumPy
degrades a ragged comparison to a scalar `False` stopped being true:

```
$ python -c "import numpy as np; print(np.__version__); \
    print(np.all(np.array(list('abcde')) == np.array(list('abc'))))"
2.4.6
ValueError: operands could not be broadcast together with shapes (5,) (3,)
```

So any CSV with this session's exact name but a different row count -- a run
interrupted mid-write, a hand-edited file, a dataset that gained or lost a stamp
-- killed the mosaic before its window existed, instead of being recognised as a
different dataset and forked. Fixed by adding the same `len(...) == len(...)`
short circuit the 1-by-1 tool has, so the ragged comparison is never reached.

Reproduced and verified headless from a scratch workspace outside the checkout,
against `test_stamps_png_jpg/`: a 3-row `classification_mosaic_autosave_m_5_7.csv`
against the 5 real stamps raises `ValueError` at `mosaic.py:860` on the unfixed
code, and on the fixed code prints "Classification file corresponds to a
different dataset." and opens on a `-new_dataset_1` fork.

## 11. Seeded CSV globs collide on the seed prefix — PRE-EXISTING, FIXED 2026-09-21

[`mosaic.py`](mosaic.py), [`single_viewer.py`](single_viewer.py). Both tools
looked for a seeded session's CSV with `{base}_{seed}*.csv`, so seed 7 also
matched seed 70 and seed 78. With three such files present, the `class_file[-2]`
pick lands on a neighbouring seed's file, the dataset check correctly rejects it,
and the tool starts an empty `-new_dataset_1` beside the perfectly good seed-7
CSV the user was actually resuming -- silently, at the cost of the session.

Fixed by globbing `{base}-*.csv` plus `{base}.csv` instead, mirroring what the
unseeded branch already does. This is exactly the separation
[`imaging.next_free_csv_path`](imaging.py) documents and relies on: `-` introduces
a suffix, `_` introduces the seed, so nothing this tool writes is missed by the
narrower pattern.

Reproduced and verified headless from a scratch workspace outside the checkout,
against `test_stamps_png_jpg/`. With `..._5_7.csv` (the real session) plus
`..._5_70.csv` and `..._5_78.csv` (a different dataset), both unfixed tools read
`..._5_70.csv`, declared a different dataset and created `..._5_7-new_dataset_1.csv`;
both fixed tools read `..._5_7.csv` and created nothing.

## 12. Two mosaics in one workspace share `cache/temp` — PRE-EXISTING, FIXED 2026-09-21

[`mosaic.py`](mosaic.py), [`paths.py`](paths.py). PLAN.md item 2b left this open
deliberately: anchoring state to the workspace fixed the *cross*-workspace case
and widened the same-workspace one, and the note there said a per-process scratch
dir was what it would take. That is what this is.

Tile filenames are index-based and every mosaic wipes the scratch directory at
startup and on each page turn, so a second mosaic started in the same workspace
deleted the first one's renders out from under it. New `paths.scratch_dir` gives
each process `<cache>/temp/<pid>/` and removes it via `atexit` on the way out.

Leftovers cannot accumulate: anything under the root that is not a live process's
directory is swept at startup. That covers a hard crash (where `atexit` never
runs) and loose tiles from the flat layout this replaces, including a migrated
`./.temp` -- which is why the legacy migration now targets the root rather than
the per-PID dir, and runs before `scratch_dir`. Liveness is `os.kill(pid, 0)`,
except on Windows where that call *terminates* the process it is asked about;
there the sweep is skipped entirely, costing a leaked directory rather than
someone's editor. A reused PID looks alive and is left alone, so the worst case
is one stale directory surviving one extra run.

Reproduced and verified headless from scratch workspaces outside the checkout.
Two overlapping mosaics in one workspace, first held open while the second starts
and exits, comparing tile inodes across the overlap: unfixed, `SURVIVED_UNTOUCHED:
0 of 50` -- every tile deleted and rewritten; fixed, `50 of 50`, with the two
processes in `temp/60369` and `temp/60388` and an empty `temp/` after both exit.
Crash path: `kill -9` on a running mosaic left `temp/60832` behind, and the next
mosaic swept it and cleaned up after itself. A planted pre-2b-iii `./.temp` is
still migrated out of the CWD. Four tests in [`tests/test_paths.py`](tests/test_paths.py)
cover the per-process naming, a live sibling being left alone, a dead PID being
reaped (forking and reaping a child for a PID that is certainly free, rather than
guessing a large number), and the flat-layout sweep.

## 13. `df.drop(...)` result discarded, so `Unnamed:` columns survive — PRE-EXISTING, FIXED 2026-09-21

[`single_viewer.py`](single_viewer.py). `df.drop(keys_to_drop, axis=1)` with no
`inplace=True` and its return value unused: the columns were collected and then
nothing happened to them. Fixed with `df = df.drop(...)`.

Verified headless from a scratch workspace outside the checkout: reading a CSV
with a trailing empty field, the unfixed viewer reports
`COLUMNS: [..., 'time', 'Unnamed: 9']` and the fixed one `[..., 'time']`.

## 15. A classification label outside the current scheme crashes the 1-by-1 viewer — PRE-EXISTING, FIXED 2026-09-21

[`single_viewer.py`](single_viewer.py), [`lobby.py`](lobby.py). Two parts, one
per how the tool was reached, per the user's own framing of the fix:

**The 1-by-1 viewer itself**, always. `dict_class2button` / `dict_subclass2button`
were built from `args.classifications` (the current scheme) and then indexed
with whatever string the CSV holds, with no `.get` and no membership check --
`self.dict_class2button[grade]` raised `KeyError` at startup (before the window
existed) if the resume position landed on such a row, and inside a slot on
navigation (Qt swallows the traceback and leaves the previous row's button
highlighted). Fixed with a shared `_button_for_grade(grade, dict_grade2button)`
lookup used at both call sites (`__init__`'s initial highlight and
`update_classification_buttoms`/`update_subclassification_buttoms`): it
returns `None` for a label truly absent from the scheme dict (distinct from a
key the dict deliberately maps to `None`, like a major with no subclass
button). The classification itself is untouched, valid data; only the button
lookup was ever the problem.

Not crashing is not enough, though -- a row whose label has no button looks
exactly like an unclassified one, which invites overwriting it. So the viewer
now says so **at launch**, over the whole CSV, rather than only when the user
happens to navigate onto such a row (`_report_unknown_classifications`, fed by
`_unknown_classification_counts`, which counts both columns together). Two
channels, because the user has to see this before grading anything:

* the **terminal** gets a boxed warning block at startup listing every
  undeclared label with its row count, the current majors, and the CSV's path;
* the **bottom of the window** gets a persistent, coloured status-bar notice
  ("Not in scheme: 'B' (2 rows), 'Lens' (1 row)"), added with
  `addPermanentWidget` rather than `showMessage` precisely so it outlives every
  transient message the session later prints over it ("Downloading ...",
  "Last image") -- it is the only thing on screen distinguishing those rows
  from unclassified ones.

Landing on such a row still shows the transient per-row message in the status
bar's message area, naming the row in front of the user; that one no longer
also prints, since the startup block already lists everything.

The same alert also **copies the classification file aside**
(`_backup_classification_csv`), because the alert means exactly that this
session can overwrite labels it cannot show a button for. The first copy is
`<name>.csv.bak`, and every later launch that hits the alert adds the next
number -- `.bak.1`, `.bak.2`, ... (`_backup_paths` walks that sequence to the
first free name). No copy is ever replaced: an earlier one may be the only
remaining record of a label that has since been graded over. A copy is skipped
when the newest one already matches the file byte for byte (`filecmp.cmp`,
`shallow=False`), so relaunching without grading anything does not pile up
identical files.

The naming is deliberately *not* `imaging.next_free_csv_path`, whose
`-marker_N.csv` names exist to be found by the viewers' CSV globs. A backup
needs the opposite -- nothing here ends in `.csv`, so `obtain_df` cannot
resume one by mistake (verified: with three backups sitting beside it, the
viewer still reads the `.csv`). A failed copy (read-only directory) is
reported in the warning block, telling the user to copy the file by hand, and
does not stop the tool. The copies are gitignored as `*.csv.bak*` -- the
existing `*.csv` rule does not cover them, and neither would a plain
`*.csv.bak`, which matches the first copy but not `.bak.1` (gitignore matches
the whole name, so the trailing `*` is load-bearing).

**The lobby**, additionally, since it is the only one of the three tools with a
scheme editor to offer fixing this to. Before launching the 1-by-1 viewer (both
the direct launch and the second stage of the chained workflow), `lobby.py`'s
new `LobbyWindow.resolve_single_classifications_string` predicts the CSV that
launch would resume (`predict_single_csv_path`, mirroring `obtain_df`'s own
naming, the way `predict_mosaic_csv_path` already does for the mosaic), reads
it, and diffs its `classification`/`subclassification` columns against the
current scheme table. For each label the scheme doesn't declare, it asks --
add it to the scheme (`AddUnknownClassificationDialog`, which lets the user
pick a keyboard shortcut, and for a major, optionally a subclass with its own
shortcut, in the same dialog), or ignore it and launch anyway, in which case
the viewer's own message is what the user sees. The headless CLI
(`--no-gui`/`--print-command`) does not run this check -- there is no one to
ask -- and launches straight through to the viewer's own tolerant behavior.
The mosaic is unaffected either way: it stores numeric codes, not scheme
names, so it cannot have this mismatch (item 14).

Verified headless (`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`) from a
scratch workspace outside the checkout, against `test_stamps_png_jpg/` with a
planted CSV: a row classified `'B'` reopened with `--classifications
"A=1;Lens=2;C=3"` no longer raises at startup (counter lands on that row) or on
`next()` into it (previously reproduced as a `KeyError` at both sites), and
leaves `bactivatedclassification` cleanly `None` rather than stale. The
launch-time report was verified on a CSV mixing two undeclared majors and an
undeclared subclass under `--classifications "A=1;C=3"`: the startup block
lists `B (2 rows)`, `Lens (1 row)`, `Merger (1 row)` before the window opens,
and the status bar carries the matching permanent notice -- confirmed by
grabbing the rendered window offscreen, and confirmed to survive a subsequent
`showMessage`, which lands in the separate message area and leaves the notice
alone. A CSV whose labels all match the scheme, and a fresh session with no
CSV at all, both produce no warning and no notice.

The backups were verified through the case they exist for, four launches with
grading in between: launch 1 writes `.bak` (`['B','Empty','B','Lens','A']`);
launch 2, with nothing graded, writes nothing and says the copy already
matches; row 0's `B` is graded over with `A`, and launch 3 writes `.bak.1`
(`['A','Empty','B',...]`); row 2's `B` is graded over with `C`, and launch 4
writes `.bak.2` (`['A','Empty','C',...]`). Every overwritten label survives in
an earlier copy. With three backups beside it the viewer still reads the
`.csv`, never a `.bak`. No copy is written when the scheme matches, and an
unwritable `Classifications/` (with a copy genuinely needed, so the identical
check cannot short-circuit it) produces `Backup FAILED ([Errno 13] Permission
denied: ...)` plus the copy-by-hand instruction, with the viewer still
starting normally. The lobby side was driven directly (constructing
`LobbyWindow` headless, `QMessageBox.question`/`AddUnknownClassificationDialog`
substituted with fakes standing in for the user's choice): a resumed CSV with
an unknown major and an unknown sub produces the expected
`(unknown_majors, unknown_subs)`, choosing "add" for the major inserts it (with
a chosen key and an optional subclass) into `_scheme_table_to_rows()` and the
returned `--classifications` string, and choosing "ignore" for the sub leaves
the scheme untouched. Existing suite still green throughout (143 tests).

---

## 14. `lobby.py` keeps per-session state in a global blob — PRE-EXISTING, FIXED 2026-09-22

[`lobby.py`](lobby.py), [`state.py`](state.py). The deferred follow-up to the
resume-position fix, one level up: `scheme_rows`, `main_band`, `color_bands`,
`rgb_composites` and `mosaic_ncols`/`mosaic_nrows` all lived in the single
`config_lobby.json` alongside `session_name`, which was just another field in
the same flat dict. They therefore persisted across a change of session instead
of travelling with it. 2b-v had moved that file to `<CWD>/.qtstamp/`, making the
blob per *workspace* rather than per user — narrower blast radius, still not per
session.

The sharp end: `classifications_string_from_rows` builds the 1-by-1 viewer's
`--classifications` string from `scheme_rows` **at launch time**, and nothing
compared it against the CSV that session had already written. Reopen an earlier
session after editing the scheme and the viewer came up with a scheme that no
longer matched its own CSV — a renamed major leaving rows labelled with a class
that no longer exists, a reused key silently meaning something different from
what was recorded. That mismatch is also what produced item 15. The mosaic was
never affected: it stores numeric codes, not scheme names.

**The fix: a session record for the lobby, in the store that already exists.**
`state.py` had proven the mechanism in both viewers, so the lobby now keeps its
own entry in the same `sessions.json`, under `SessionId('lobby', path, name,
seed)` and anchored on the classifications directory like every other entry. The
record carries a `config` sub-dict holding exactly the six dataset-scoped keys
above.

Which keys those are was the decision this item turned on. The split is by
lifetime, not by tidiness:

* **Session-scoped**, into the record: the classification scheme and the band
  setup (`scheme_rows`, `main_band`, `color_bands`, `rgb_composites`) plus the
  mosaic grid shape. These describe the *dataset*.
* **Workspace-scoped**, staying in `config_lobby.json`: the identity itself
  (`data_path`, `session_name`, `seed_*`, `classifications_path`) and genuine UI
  preference — `run_mode_index`, `dock_state`, `output_path`,
  `mosaic_printname`, the three `mosaic_*_positive`, `copy_instead_of_symlink`.

`config_lobby.json` **keeps** its copy of the six, with a new meaning: the
template for a session that has no record yet. Carrying a scheme forward into a
new session is wanted and is not the bug; resurrecting it over an older
session's own scheme is.

A record is created **at launch** — every mode, plus again after the chained
run's stage 1, since the stage-2 prompt from item 15 can add scheme rows that
have to be captured. After that it tracks edits: switching identity away from a
session that has a record saves it, and so does closing the window. A session
that was never run leaves nothing behind, so the picker lists real work only.

Supporting changes, each with its own reason:

* **`state._update_session`** — entries are now written by two different
  writers (a viewer's resume position, the lobby's config), and `save_session`
  used to rebuild the entry from scratch, so whichever wrote last would have
  erased the other's half. It now merges. `save_session` reroutes through it,
  behaviour otherwise unchanged. New public API around it:
  `save_session_config`, `load_session_config`, `list_sessions`,
  `session_id_from_entry`.
* **Legacy-key dedup**, found while testing the above and fixed here because
  the merge made it visible: a pre-2b-i, absolute-realpath-keyed entry was read
  through the legacy fallback but never removed, so every save left a stale
  duplicate of the same identity behind, competing for a slot under `prune`.
  `_update_session` now drops the legacy copy once it has absorbed it — the
  same "both keys are one identity" rule `forget_session` already applied. This
  predates item 14; `save_session` had it too.
* **Restore happens *after* the band rescan, not before.** Reproduced and fixed
  during review: `_rescan_bands` rebuilds the colour-band checklist from the
  boxes currently ticked in it, and a restore can only tick a band the list
  already offers. Restoring first ticked the incoming session's bands against
  the *outgoing* path's band list, dropping every band the two datasets did not
  share, and the rescan then read that emptied list back as the answer. Two
  datasets with different bands is exactly the case the record exists for.
* **The identity widgets are disabled during a chained stage 1.** The scheme
  now follows the session name/seed, so letting those change mid-run would swap
  the scheme out from under the stage-2 launch, which builds from the captured
  context.

**The recent-sessions picker** is the user-visible payoff, and what justifies
the mechanism rather than a narrower correctness patch: a "Recent sessions..."
button in the Session dock lists the lobby's saved sessions most-recent-first
(name, data path, seed, last opened), with Open and Forget. The stored path is
relative to the classifications directory, so the list survives the tree being
copied elsewhere, exactly like the viewers' resume positions.

The headless CLI gets the same fix, since `--no-gui -N old_session` had the
identical failure. `config_from_cli` now layers config file → identity
overrides (`--path`/`--name`/`--seed`/`--classifications-dir`) → the session
record → the remaining overrides, so an explicit flag still beats the record
while the record beats the stale workspace blob.

**Verified** headless (`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`) from
scratch workspaces outside the checkout, against generated two-dataset PNG
fixtures with deliberately disjoint band sets:

* the bug itself — session `alpha` with scheme `Lens/Ring/Junk`, scheme
  rewritten to `Cat/Dog` under session `beta`, reopening `alpha` brings back
  `Lens=1;Ring=2;Junk=3`; and `beta` correctly inherited `alpha`'s scheme as a
  template when it was new;
* the no-op guard — `editingFinished` fires on mere focus-out, so an
  identity-change event that is not actually a change must do nothing; an
  unsaved in-progress scheme row survives one. This was the failure mode most
  likely to eat the user's work, so it is pinned;
* the cross-dataset band case, which failed on first run (colour bands came
  back empty) and passes after the ordering fix — colour bands and RGB
  composites both restore per session;
* `closeEvent` creating no record for a session that was never run;
* the picker — listing and ordering, `(unnamed)`/`--` sentinels, the stored
  relative path resolving back to the right absolute one, Forget removing both
  the row and the underlying record;
* the chained-run lockout, including the seed spin box following its checkbox
  rather than the blanket re-enable;
* CLI precedence, all six cases (record beats blob for scheme/band/grid;
  `-b`/`--ncols` beat the record; an unknown name, a different seed, and no
  data path each fall back to the blob without crashing);
* and the last link — that the record reaches the *actual* argv, not just the
  config dict: `--print-command` emits
  `--classifications 'ArchivalLens=1;ArchivalLens:Double=2' ... -b VIS` and
  `-l 4 -m 6` for the mosaic, from the record alone.

Suite: 164 tests, up from 143 — 12 new in `tests/test_state.py` for the new
store API (including the merge property in both orders, and the legacy dedup),
and a new `tests/test_lobby.py` with 9 covering CLI precedence and
`classifications_string_from_rows`.
