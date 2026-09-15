# Known bugs

Found reviewing the working-tree changes against `715261a` (session-state split,
atomic autosaves, `--classifications-dir`, drawn background markers) on
2026-09-14. Verified by running both viewers headless (`QT_QPA_PLATFORM=offscreen`,
`MPLBACKEND=Agg`) against synthetic FITS and PNG datasets, from a directory
outside the checkout.

"New" = introduced by the changes on this branch since `715261a`. "Pre-existing"
= older, but adjacent to them, or newly reachable. Two items found in this pass
(the permissions narrowing and the foreign-write guard's hole) were fixed
before commit; see CHANGES.md for how. What's left below is still open.

Not bugs, verified working: both viewers run standalone from any CWD; PNG/JPG
classification; a change of `--name` or `--seed` restarting at the first
image/page in both tools; a mosaic reshape keeping the CSV and re-deriving the
page; legacy `.config*.json` migration; `--reset-config` / `--reset-position`;
fork-on-foreign-write; `tests/` (48 tests).

---

## 1. The lobby launches its children with `cwd=REPO_ROOT` — PRE-EXISTING, now inconsistent

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

## 2. `Classifications/README` still documents the old mosaic filenames — NEW

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

## 3. The new state files are not gitignored — NEW

[`.gitignore`](.gitignore).

`sessions.json`, `.preferences_single.json`, `.preferences_mosaic.json`,
`.config.json.bak`, `.config_mosaic.json.bak`. The existing `.config*.json`
pattern does not match the `.bak` names, and nothing matches the other three.
First run from the repo root produces five untracked files.

`sessions.json` is also the only one of the set that is not a dotfile, so it is
visible clutter in whatever directory the user launches from.

## 4. The 1-by-1 "Go to" dialog can crash the tool — PRE-EXISTING

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

## 5. The mosaic's page warning is 0-based, the widget is 1-based — NEW

[`mosaic.py:643`](mosaic.py#L643) says `'Pages go from 0 to {PAGE_MAX-1}'` while
`LabelledIntField` displays and accepts 1-based page numbers ("Page 1 / 3").
Should be `1 to {PAGE_MAX}`. Cosmetic, and currently unreachable behind
`QIntValidator(1, PAGE_MAX)`; the `>= PAGE_MAX` bound fixed just above it is
correct.

## 6. `new_class` can be unbound on a modified click — PRE-EXISTING

[`mosaic.py:318`](mosaic.py#L318), `MiniMosaics.mousePressEvent`.

On an unclassified stamp, the branch assigns `new_class` only for
`Qt.ControlModifier`, `Qt.ShiftModifier` or `Qt.NoModifier`. Ctrl+Shift+click,
Alt+click or a Meta-modified click falls through all of them and reaches
`self.update_df_func(event, self.i, new_class)` with `new_class` unbound →
`UnboundLocalError`. Grades already on disk are unaffected.

Fix: default `new_class = self.is_a_candidate` (treat an unrecognised modifier
as a no-op) and return early rather than calling `update_df_func`.

---

## Minor

* **`obtain_df` length check.** [`mosaic.py`](mosaic.py) compares
  `np.all(self.listimage == df['file_name'].values)` without the
  `len(self.listimage) == len(df)` guard [`single_viewer.py`](single_viewer.py)
  has. NumPy degrades a length mismatch to a scalar `False`, so it happens to
  work, but it relies on deprecated behaviour.
* **Seeded globs collide on prefixes.** Both viewers glob
  `..._{n}_{seed}*.csv`, so seed 7 also matches `..._70.csv` and `..._78.csv`.
  Usually harmless (the dataset check rejects them), but with three such files
  present the `class_file[-2]` pick can land on the wrong one and then start an
  empty `-new_dataset_1` file beside a perfectly good seed-7 CSV. Pre-existing.
* **Two mosaics in one directory share `./.temp`.** Each wipes the scratch
  directory at startup (`clean_dir`) and the tile filenames are index-based, so
  a second mosaic pulls the first one's tiles out from under it. Pre-existing;
  PLAN.md item 2b.
* **`df.drop(keys_to_drop, axis=1)` result is discarded** in
  `single_viewer.obtain_df` (no `inplace=True`, return value unused), so
  `Unnamed:` columns are never actually dropped. Harmless today because the
  1-by-1 tool round-trips its index. Pre-existing.
