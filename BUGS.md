# Known bugs

Open items only. Fixed ones move to [BUGS-DONE.md](BUGS-DONE.md) with their fix
writeup, keeping their original number -- numbering is continuous across the two
files and never reused, because `CHANGES.md`, `PLAN.md`, `STATUS.md` and two
source comments cite items by number. Next number is 16.

"New" = introduced by the changes on this branch since `715261a`. "Pre-existing"
= older, but adjacent to them, or newly reachable. Anything GUI-affecting is
verified by running the affected tool headless (`QT_QPA_PLATFORM=offscreen`,
`MPLBACKEND=Agg`) from a directory outside the checkout before it is called
fixed.

Items 14 and 15 arrived 2026-09-21 from PLAN.md's "1b. Known bugs -- remaining"
section, which no longer exists: 14 was the entry that was sitting there, re-checked
against the current tree, and 15 the concrete failure found while re-checking it.
Both are fixed; see [BUGS-DONE.md](BUGS-DONE.md).

Items 16-29 arrived 2026-09-22 from a full-source audit (all ten modules, six
parallel reviewers, every finding re-verified against the source before being
filed here). Nothing in that pass was found by running the tools -- these are
read-and-verify findings, so each one needs its own repro before it is called
fixed, per the headless rule above. Dead code and the duplication the same audit
turned up are not bugs and live in PLAN.md items 2d and 2e.

---

## Major

**16. `_apply_preset` restores bands before rescanning -- item 14's bug, at a
second call site.** `lobby.py:1385-1387` calls `_apply_config_to_widgets()` and
only then `_rescan_bands()`. Item 14 fixed exactly this ordering in
`_on_identity_changed`, whose docstring calls the order "load-bearing", but the
preset path kept the old one. The mechanism is the one item 14 documents:
`_refresh_band_widgets` (`lobby.py:838-840`) reads the checklist back only when
it is non-empty (`self.color_bands_list.count()`), otherwise falling back to
`config_dict['color_bands']`. Applying a preset first ticks the preset's bands
against the *outgoing* path's band list, leaving it non-empty, so the rescan
reads that already-filtered list as the answer and every band the two datasets
do not share is dropped. Reachable because `_preset_snapshot`
(`lobby.py:1374-1377`) strips only `dock_state` and so a user-saved preset does
carry `data_path`; packaged presets are forbidden from carrying it, but only by
a test convention (`tests/test_paths.py`), not by the snapshot code.
`__init__`'s same-looking order (`lobby.py:505-508`) is safe -- the checklist is
still empty there, so the fallback branch runs.

**17. Nothing serialises the read-modify-write on the shared `sessions.json`.**
`state._update_session` (`state.py:270-282`) does load -> mutate -> write with no
cross-process exclusion, and so do `forget_session` and
`migrate_sessions_store`. `imaging.atomic_write` makes each individual write
atomic, which is not the same guarantee. There is no lock of any kind in the
repo (`fcntl`/`flock`/`filelock`: zero hits). The window is real rather than
theoretical: the lobby launches viewers with `QProcess` parented to itself
(`lobby.py:1201-1207`, `1225-1233`), so it stays alive and keeps writing
(`save_session_config`, `lobby.py:1467`) while a viewer writes its position
(`save_session`, `mosaic.py:803`, `single_viewer.py:896`). A stale read written
back drops the other process's entry. Blast radius is resume position, scheme
and band setup -- classification CSVs are written by a different path and are
not at risk. No test covers concurrent writers.

**18. `background_rms_image` throws away the box size its caller computed.**
`mosaic.py:1067-1069` and `single_viewer.py:1297-1299` both open with
`cb=10`, overwriting the parameter. `scale_val` computes a size-dependent
`box_size_vmin` for large stamps (`mosaic.py:1049-1055`,
`single_viewer.py:1320-1322`) precisely to pass it here, so that whole branch is
dead and every stamp gets a fixed 10x10 corner box for its noise estimate.
Display scaling only -- no effect on classifications -- but the two copies have
also drifted (threshold `>173` in the mosaic, `>170` in the 1-by-1 viewer),
which is why the fix belongs with PLAN.md 2e rather than being applied twice.

## Minor

**19. The lobby's workspace config is only persisted on a clean close.**
`save_dict()` has exactly one caller, `closeEvent` (`lobby.py:1438`), while
`_save_session_record()` also runs on every Run (`lobby.py:1164`, `1275`). A
kill or crash after several successful runs therefore keeps the session record
but loses `data_path`, `session_name`, `seed`, `run_mode_index` and
`output_path`. Asymmetry introduced by item 14's split, not covered by its
verification.

**20. `save_session(csv=None)` erases a recorded CSV.** `state.py:286-292`
passes `'csv': None` unconditionally into `_update_session`'s `entry.update`,
so a caller that omits `csv` clears a previously stored value instead of
leaving it alone. Latent: both call sites always pass it today.

**21. `NamedLabel.getValue` reads a nonexistent attribute.** `widgets.py:95-96`
returns `int(self.lineEdit.text())-1`, but the class stores its field as
`self.label` (`widgets.py:83`); `lineEdit` belongs to the sibling
`LabelledIntField`. Latent -- the only `.getValue()` call (`mosaic.py:714`) is
on a `LabelledIntField` -- but it is an `AttributeError` waiting for the next
caller.

**22. `get_value_range_asymmetric`'s box scales with band count.**
`imaging.py:46` takes `np.sqrt(np.prod(x.shape) * 0.01)` over the whole array,
and `xl, yl, _ = np.shape(x)` on the next line proves it is the 3-D composite,
so the "1% of area" box is about `sqrt(nbands)` times larger than it reads and
silently changes size with how many bands a dataset has.

**23. `get_value_range_asymmetric` can build a negative slice start.**
`imaging.py:50-53`: for a stamp smaller than the 8-pixel default box,
`int(xl/2 - box/2)` goes negative (xl=5 gives -1) and the negative index wraps
instead of clamping, so the high percentile is taken over the wrong region.

**24. `get_contrast_bias_reasonable_assumptions` divides without a guard.**
`imaging.py:70-75` divides by `(bkg_level - 1)`, with only a trailing comment
saying it must not be 1. `bkg_level` comes from real image data via
`clip_normalize` + `scale`, so 1.0 is reachable and yields a silent inf/NaN
that propagates into the rendered image.

**25. ds9 children are never reaped.** `single_viewer.py:1263` discards the
`subprocess.Popen` handle, so repeated "Open ds9" clicks accumulate zombies for
the life of the session.

**26. `obtain_df` hardcodes `file_index = -2`.** `single_viewer.py:1518-1541`
picks the second-to-last glob match whenever more than one CSV matches. That is
correct for the two-match case item 11 was about (live file plus one
`-new_dataset_N` fork); with three or more it selects by sort position rather
than by which file is live.

**27. `record_miss` is not atomic.** `workers.py:106-114` writes the `.miss`
marker directly, unlike `_download_to`, which goes through `mkstemp` +
`os.replace`. Two workers racing on one marker can leave invalid JSON. Benign
in practice -- `read_miss` already treats an unparseable marker as absent, so
the cost is one extra retry.

**28. An explicit empty `--state-dir` is silently ignored.** `paths.py:134-139`
and `293-305` test the override for truthiness, so `--state-dir ''` falls
through to the env/anchor default instead of being honoured or rejected.

**29. The lobby's unknown-label check compares bare subclass names.**
`_unknown_classification_labels` (`lobby.py:1035-1056`) matches a CSV's
`subclassification` values against a flat set of known sub-names rather than
(major, subclass) pairs, so a subclass valid under one major reads as known
under any other. Needs a cross-check against how `single_viewer.py` keys its
subclass buttons before it is called a bug or closed as a non-issue -- it is
filed here so the check is not forgotten.
