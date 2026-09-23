"""Tests for state.py -- the session store and preferences split.

state.py is deliberately Qt-free and pure over a directory path, so all of this
runs headless. Written as plain pytest functions, but also runnable directly
(`python tests/test_state.py`) until the suite of item 3 lands.
"""

import json
import os
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Since 2b-v, load_preferences reads along paths.config_search_path(), whose
# middle entry is the *real* per-user config dir. Pin it somewhere that cannot
# exist before importing anything, or a developer who has their own
# preferences_single.json there sees this suite fail for reasons that have
# nothing to do with the code. The few tests that exercise the fallback
# deliberately point it at a tmpdir themselves (see `_env`).
os.environ['QTSTAMP_CONFIG_DIR'] = os.path.join(
    tempfile.gettempdir(), 'qtstamp-test-no-such-user-dir')
os.environ.pop('QTSTAMP_STATE_DIR', None)

import paths
import state


SINGLE = 'single'
LIST = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
POSIX = sys.platform != 'win32'   # elsewhere chmod only toggles the read-only bit


def sid(tool=SINGLE, path='/data/stamps', name='', seed=None):
    return state.SessionId(tool, path, name, seed)


# --- how the files are written -------------------------------------------

def test_store_files_are_written_like_ordinary_files(tmpdir):
    """Same permission rule as the classification CSVs (imaging.atomic_write).

    These went out at 0600 while every other file the tools write followed the
    umask, purely because they were built on `tempfile.mkstemp`.
    """
    if not POSIX:
        return
    probe = os.path.join(tmpdir, 'probe.tmp')
    open(probe, 'w').close()
    expected = stat.S_IMODE(os.stat(probe).st_mode)
    os.remove(probe)

    state.save_preferences(SINGLE, {'colormap': 'viridis'}, config_dir=tmpdir)
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    for path in (state.preferences_path(SINGLE, tmpdir), state.sessions_path(tmpdir)):
        assert stat.S_IMODE(os.stat(path).st_mode) == expected, path


def test_rewriting_the_store_preserves_its_mode(tmpdir):
    if not POSIX:
        return
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    store = state.sessions_path(tmpdir)
    os.chmod(store, 0o640)
    state.save_session(sid(), tmpdir, position_filename='b.fits')
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o640
    assert state.load_session(sid(), tmpdir)['position_filename'] == 'b.fits'


def test_no_scratch_files_are_left_behind(tmpdir):
    state.save_preferences(SINGLE, {'colormap': 'viridis'}, config_dir=tmpdir)
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    assert not [f for f in os.listdir(tmpdir) if f.startswith('.tmp-')]


# --- identity -------------------------------------------------------------

def test_identity_keys_on_tool_path_name_seed():
    """Session key axes: tool, path, name, seed -- and now the anchor itself.

    Identity is stored *relative to the classifications directory* (PLAN.md
    2b-i), so the anchor is effectively part of the key: the same data path
    under two different anchors must produce two different keys. The actual
    point of the step is the opposite direction -- the same data path at the
    same offset from a *moved* anchor must produce the *same* key, which is
    what lets the tool resume after the whole tree is copied elsewhere.
    """
    anchor = '/anchor/Classifications'
    base = state.session_key(sid(), anchor)
    assert state.session_key(sid(tool='mosaic'), anchor) != base
    assert state.session_key(sid(path='/data/other'), anchor) != base
    assert state.session_key(sid(name='vfin'), anchor) != base
    assert state.session_key(sid(seed=812), anchor) != base
    assert state.session_key(sid(), anchor) == base

    # Same data path, a different anchor -> a different key. (The two anchors
    # differ in depth, not just name, so the relative path -- and thus the
    # key -- actually differs; two equally-deep anchors can coincidentally
    # produce the same '../../data/stamps' relative path.)
    assert state.session_key(sid(), '/a/b/c/Classifications') != base

    # The point of the step: the same data at the same *offset* from a moved
    # anchor produces the same key.
    original = sid(path='/anchor/data/stamps')
    moved = sid(path='/elsewhere/data/stamps')
    assert state.session_key(original, '/anchor/Classifications') == \
        state.session_key(moved, '/elsewhere/Classifications')


def test_identity_ignores_grid_shape_and_file_count():
    "Neither appears in SessionId at all -- the position is a filename."
    assert set(state.session_identity(sid(), '/anchor/Classifications')) == \
        {'tool', 'path', 'path_form', 'name', 'seed'}


def test_identity_normalizes_path_and_empty_name():
    anchor = '/anchor/Classifications'
    assert state.session_key(sid(path='/data/stamps/'), anchor) == state.session_key(sid(), anchor)
    assert state.session_key(sid(name=None), anchor) == state.session_key(sid(name=''), anchor)


# --- session store --------------------------------------------------------

def test_save_and_load_roundtrip(tmpdir):
    csv = os.path.join(tmpdir, 'x.csv')
    state.save_session(sid(), tmpdir, position_filename='c.fits', csv=csv)
    entry = state.load_session(sid(), tmpdir)
    assert entry['position_filename'] == 'c.fits'
    # csv is stored relative to the anchor (here, csv sits right inside it).
    assert entry['csv'] == 'x.csv'
    assert entry['last_opened'].endswith('Z')
    # The identity is stored in plain text beside its hash.
    assert entry['name'] == '' and entry['tool'] == SINGLE


def test_sessions_of_different_identities_do_not_collide(tmpdir):
    state.save_session(sid(seed=1), tmpdir, position_filename='a.fits')
    state.save_session(sid(seed=2), tmpdir, position_filename='d.fits')
    assert state.load_session(sid(seed=1), tmpdir)['position_filename'] == 'a.fits'
    assert state.load_session(sid(seed=2), tmpdir)['position_filename'] == 'd.fits'


def test_the_two_viewers_share_one_file_without_clobbering(tmpdir):
    state.save_session(sid(tool='single'), tmpdir, position_filename='a.fits')
    state.save_session(sid(tool='mosaic'), tmpdir, position_filename='c.fits')
    assert state.load_session(sid(tool='single'), tmpdir)['position_filename'] == 'a.fits'
    assert len(state.load_store(tmpdir)['sessions']) == 2


def test_missing_session_is_none(tmpdir):
    assert state.load_session(sid(), tmpdir) is None


def test_forget_session(tmpdir):
    state.save_session(sid(), tmpdir, position_filename='b.fits')
    assert state.forget_session(sid(), tmpdir) is True
    assert state.load_session(sid(), tmpdir) is None
    assert state.forget_session(sid(), tmpdir) is False


def test_corrupt_store_reads_as_empty(tmpdir):
    with open(state.sessions_path(tmpdir), 'w') as f:
        f.write('{not json')
    assert state.load_store(tmpdir) == state._empty_store()
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    assert state.load_session(sid(), tmpdir)['position_filename'] == 'a.fits'


def test_store_writes_leave_no_temp_files(tmpdir):
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    assert os.listdir(tmpdir) == [state.SESSIONS_FILENAME]


def test_prune_keeps_the_most_recently_opened():
    store = {'version': 1, 'sessions': {
        'old': {'last_opened': '2020-01-01T00:00:00Z'},
        'mid': {'last_opened': '2024-01-01T00:00:00Z'},
        'new': {'last_opened': '2026-01-01T00:00:00Z'},
        'never': {},
    }}
    state.prune(store, max_sessions=2)
    assert sorted(store['sessions']) == ['mid', 'new']


def test_prune_is_a_noop_below_the_cap():
    store = {'version': 1, 'sessions': {'a': {'last_opened': 'z'}}}
    state.prune(store, max_sessions=50)
    assert list(store['sessions']) == ['a']


def test_saving_prunes_the_store(tmpdir):
    for i in range(5):
        state.save_session(sid(seed=i), tmpdir, position_filename='a.fits', max_sessions=3)
    assert len(state.load_store(tmpdir)['sessions']) == 3


# --- position resolution --------------------------------------------------
#
# These exercise resolve_position on its own, with hand-built entries and no
# classifications_dir -- the pure-unit-test corner the docstring calls out,
# where only an absolute recorded csv can be compared at all.

def test_resolve_position_finds_the_saved_file():
    assert state.resolve_position({'position_filename': 'c.fits'}, LIST) == 2


def test_resolve_position_falls_back_to_zero():
    assert state.resolve_position(None, LIST) == 0
    assert state.resolve_position({}, LIST) == 0
    assert state.resolve_position({'position_filename': None}, LIST) == 0
    # A file that has since been deleted is a defined fallback, not a crash.
    assert state.resolve_position({'position_filename': 'gone.fits'}, LIST) == 0


def test_resolve_position_survives_files_being_added_or_removed():
    entry = {'position_filename': 'c.fits'}
    assert state.resolve_position(entry, ['a.fits', 'c.fits']) == 1
    assert state.resolve_position(entry, ['new.fits'] + LIST) == 3


def test_resolve_position_rejects_a_different_csv():
    entry = {'position_filename': 'c.fits', 'csv': '/abs/one.csv'}
    assert state.resolve_position(entry, LIST, csv='/abs/one.csv') == 2
    assert state.resolve_position(entry, LIST, csv='/abs/two.csv') == 0


def test_resolve_position_with_no_anchor_rejects_a_relative_recorded_csv():
    "Without classifications_dir, a relative recorded value has nothing to resolve against."
    entry = {'position_filename': 'c.fits', 'csv': 'sub/one.csv'}
    assert state.resolve_position(entry, LIST, csv='/abs/sub/one.csv') == 0


def test_reshaped_mosaic_lands_on_the_page_holding_the_same_object():
    """The point of storing a filename: grid shape stays out of the identity.

    Production always records the CSV alongside the position, so pass one here
    too -- without it this test silently skips the cross-check that decides
    whether a reshape keeps its place at all (see the test below).
    """
    images = [f'{i:03d}.fits' for i in range(100)]
    entry = {'position_filename': images[42], 'csv': '/abs/deck_5.csv'}
    index = state.resolve_position(entry, images, csv='/abs/deck_5.csv')
    assert index // (5 * 8) == 1   # 40-cell pages: object 42 is on page 1
    assert index // (5 * 2) == 4   # 10-cell pages: the same object is on page 4


def test_a_reshape_keeps_the_position_because_the_csv_name_is_shape_free():
    """The precondition on the test above.

    The mosaic's CSV filename carries dataset identity only -- name, image
    count, seed -- so no reshape can rename it, and the cross-check below
    never fires on a reshape alone.
    """
    images = [f'{i:03d}.fits' for i in range(100)]
    csv = '/abs/classification_mosaic_autosave_deck_100.csv'
    entry = {'position_filename': images[42], 'csv': csv}
    assert state.resolve_position(entry, images, csv=csv) == 42


def test_a_changed_dataset_still_resets_the_position():
    """What *does* rename the CSV: name, image count or seed.

    That is a different dataset, so the recorded position belongs to another
    session and is stale by definition.
    """
    images = [f'{i:03d}.fits' for i in range(100)]
    entry = {'position_filename': images[42],
             'csv': '/abs/classification_mosaic_autosave_deck_100.csv'}
    assert state.resolve_position(
        entry, images, csv='/abs/classification_mosaic_autosave_deck_100_7.csv') == 0
    assert state.resolve_position(
        entry, images, csv='/abs/classification_mosaic_autosave_other_100.csv') == 0


# --- preferences ----------------------------------------------------------

DEFAULTS = {'colormap': 'gist_gray', 'scale': 'log', 'autonext': True}


def test_preferences_roundtrip_and_defaults(tmpdir):
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir) == DEFAULTS
    state.save_preferences(SINGLE, dict(DEFAULTS, scale='sqrt'), config_dir=tmpdir)
    loaded = state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)
    assert loaded['scale'] == 'sqrt' and loaded['autonext'] is True


def test_preferences_drop_unknown_keys(tmpdir):
    "Position keys from the old blob must not leak back in as preferences."
    with open(state.preferences_path(SINGLE, tmpdir), 'w') as f:
        json.dump({'scale': 'sqrt', 'counter': 91, 'name': 'vfin'}, f)
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir) == \
        dict(DEFAULTS, scale='sqrt')


def test_preferences_are_per_tool(tmpdir):
    state.save_preferences('single', dict(DEFAULTS, scale='sqrt'), config_dir=tmpdir)
    assert state.load_preferences('mosaic', DEFAULTS, config_dir=tmpdir)['scale'] == 'log'


def test_reset_preferences(tmpdir):
    state.save_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)
    assert state.reset_preferences(SINGLE, config_dir=tmpdir) is True
    assert state.reset_preferences(SINGLE, config_dir=tmpdir) is False


def test_preferences_survive_a_change_of_seed(tmpdir):
    "The whole point of the split: preferences follow the user, not the dataset."
    state.save_preferences(SINGLE, dict(DEFAULTS, colormap='viridis'), config_dir=tmpdir)
    state.save_session(sid(seed=1), tmpdir, position_filename='c.fits')
    # New seed: no saved position ...
    assert state.resolve_position(state.load_session(sid(seed=2), tmpdir), LIST) == 0
    # ... but the colormap is still the user's.
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['colormap'] == 'viridis'


# --- migrate_preferences_file (pre-2b-iii, CWD-relative preferences file) --

def test_migrate_preferences_file_moves_once(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    config_dir = os.path.join(tmpdir, 'config')
    os.makedirs(legacy_dir)
    legacy_file = os.path.join(legacy_dir, '.preferences_single.json')
    with open(legacy_file, 'w') as f:
        json.dump({'colormap': 'viridis'}, f)

    assert state.migrate_preferences_file(
        SINGLE, legacy_dir=legacy_dir, config_dir=config_dir) is True
    assert not os.path.exists(legacy_file)
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=config_dir)['colormap'] == \
        'viridis'


def test_migrate_preferences_file_second_call_is_a_noop(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    config_dir = os.path.join(tmpdir, 'config')
    os.makedirs(legacy_dir)
    legacy_file = os.path.join(legacy_dir, '.preferences_single.json')
    with open(legacy_file, 'w') as f:
        json.dump({'colormap': 'viridis'}, f)

    assert state.migrate_preferences_file(
        SINGLE, legacy_dir=legacy_dir, config_dir=config_dir) is True
    assert state.migrate_preferences_file(
        SINGLE, legacy_dir=legacy_dir, config_dir=config_dir) is False


def test_migrate_preferences_file_never_overwrites_an_existing_target(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    config_dir = os.path.join(tmpdir, 'config')
    os.makedirs(legacy_dir)
    legacy_file = os.path.join(legacy_dir, '.preferences_single.json')
    with open(legacy_file, 'w') as f:
        json.dump({'colormap': 'viridis'}, f)

    # A newer preferences file already exists at the target.
    state.save_preferences(SINGLE, dict(DEFAULTS, colormap='gray'), config_dir=config_dir)

    assert state.migrate_preferences_file(
        SINGLE, legacy_dir=legacy_dir, config_dir=config_dir) is False
    assert os.path.exists(legacy_file)  # untouched
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=config_dir)['colormap'] == \
        'gray'


# --- migration (single legacy blob -> preferences + one session entry) ----

def write_legacy(tmpdir, payload, filename='.config.json'):
    path = os.path.join(tmpdir, filename)
    with open(path, 'w') as f:
        json.dump(payload, f)
    return path


def test_migration_splits_the_old_blob(tmpdir):
    legacy = write_legacy(tmpdir, {'name': 'vfin', 'counter': 2, 'scale': 'sqrt',
                                   'colormap': 'viridis'})
    csv = os.path.join(tmpdir, 'x.csv')
    migrated = state.migrate_legacy_config(
        legacy, sid(name='vfin'), DEFAULTS, LIST,
        lambda d: d['counter'] if d.get('name') == 'vfin' else None,
        csv=csv, classifications_dir=tmpdir, config_dir=tmpdir)
    assert migrated is True
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'sqrt'
    assert state.load_session(sid(name='vfin'), tmpdir)['position_filename'] == 'c.fits'
    # Kept, not deleted.
    assert not os.path.exists(legacy) and os.path.exists(legacy + '.bak')


def test_migration_drops_a_position_it_cannot_vouch_for(tmpdir):
    legacy = write_legacy(tmpdir, {'name': 'other', 'counter': 2, 'scale': 'sqrt'})
    state.migrate_legacy_config(legacy, sid(name='vfin'), DEFAULTS, LIST,
                                lambda d: None, classifications_dir=tmpdir, config_dir=tmpdir)
    assert state.load_session(sid(name='vfin'), tmpdir) is None
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'sqrt'


def test_migration_ignores_an_out_of_range_index(tmpdir):
    legacy = write_legacy(tmpdir, {'name': '', 'counter': 99})
    state.migrate_legacy_config(legacy, sid(), DEFAULTS, LIST, lambda d: d['counter'],
                                classifications_dir=tmpdir, config_dir=tmpdir)
    assert state.load_session(sid(), tmpdir) is None


def test_migration_runs_only_once(tmpdir):
    state.save_preferences(SINGLE, dict(DEFAULTS, scale='cbrt'), config_dir=tmpdir)
    legacy = write_legacy(tmpdir, {'name': '', 'counter': 2, 'scale': 'sqrt'})
    assert state.migrate_legacy_config(legacy, sid(), DEFAULTS, LIST, lambda d: d['counter'],
                                       classifications_dir=tmpdir, config_dir=tmpdir) is False
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'cbrt'
    assert os.path.exists(legacy)  # untouched


def test_migration_without_a_legacy_file_is_a_noop(tmpdir):
    assert state.migrate_legacy_config(os.path.join(tmpdir, 'nope.json'), sid(), DEFAULTS,
                                       LIST, lambda d: 0,
                                       classifications_dir=tmpdir, config_dir=tmpdir) is False
    assert not os.path.exists(state.preferences_path(SINGLE, tmpdir))


def test_migration_does_not_overwrite_a_freshly_imported_position(tmpdir):
    """migrate_sessions_store runs first at startup; this must not clobber its result.

    A pre-split single-blob config and a pre-split CWD sessions.json can both
    exist for the same identity (the tool aged through both formats). The
    session store import is the one that should win.
    """
    session = sid(name='vfin')
    state.save_session(session, tmpdir, position_filename='d.fits')
    legacy = write_legacy(tmpdir, {'name': 'vfin', 'counter': 2, 'scale': 'sqrt'})
    state.migrate_legacy_config(legacy, session, DEFAULTS, LIST,
                                lambda d: d['counter'],
                                classifications_dir=tmpdir, config_dir=tmpdir)
    assert state.load_session(session, tmpdir)['position_filename'] == 'd.fits'


# --- the requirement itself: copying the tool and its data elsewhere ------

def test_resuming_after_copying_the_whole_tree(tmpdir):
    """Proven end to end: this is the one test that would have caught the bug.

    Copy the tool's data and classifications directory together to a new
    absolute path, keeping their position relative to each other, and the
    saved index must still be found -- not silently reset to 0.
    """
    root = os.path.join(tmpdir, 'root')
    data = os.path.join(root, 'data')
    classifications = os.path.join(root, 'Classifications')
    os.makedirs(data)
    os.makedirs(classifications)
    csv = os.path.join(classifications, 'x.csv')
    open(csv, 'w').close()

    state.save_session(sid(path=data), classifications, position_filename='c.fits', csv=csv)

    root2 = os.path.join(tmpdir, 'root2')
    shutil.copytree(root, root2)
    data2 = os.path.join(root2, 'data')
    classifications2 = os.path.join(root2, 'Classifications')
    csv2 = os.path.join(classifications2, 'x.csv')

    entry = state.load_session(sid(path=data2), classifications2)
    assert entry is not None
    index = state.resolve_position(entry, LIST, csv=csv2, classifications_dir=classifications2)
    assert index == 2   # LIST.index('c.fits') -- not 0


def test_resuming_with_data_outside_the_classifications_dir(tmpdir):
    "The anchor need not be an ancestor of the data -- a `..`-heavy relative path."
    root = os.path.join(tmpdir, 'root')
    data = os.path.join(root, 'data')
    classifications = os.path.join(root, 'out', 'Classifications')
    os.makedirs(data)
    os.makedirs(classifications)
    csv = os.path.join(classifications, 'x.csv')
    open(csv, 'w').close()

    state.save_session(sid(path=data), classifications, position_filename='b.fits', csv=csv)

    root2 = os.path.join(tmpdir, 'root2')
    shutil.copytree(root, root2)
    data2 = os.path.join(root2, 'data')
    classifications2 = os.path.join(root2, 'out', 'Classifications')
    csv2 = os.path.join(classifications2, 'x.csv')

    entry = state.load_session(sid(path=data2), classifications2)
    assert entry is not None
    assert state.resolve_position(entry, LIST, csv=csv2, classifications_dir=classifications2) == 1


def test_a_different_relative_offset_does_not_resume(tmpdir):
    "A moved tree where the data ends up at a *different* offset is a miss, not a guess."
    anchor = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(anchor)
    os.makedirs(data)
    state.save_session(sid(path=data), anchor, position_filename='c.fits')

    moved_data = os.path.join(tmpdir, 'elsewhere', 'data')
    os.makedirs(moved_data)
    assert state.load_session(sid(path=moved_data), anchor) is None


def test_csv_is_stored_relative_to_the_anchor(tmpdir):
    classifications = os.path.join(tmpdir, 'Classifications')
    csv_dir = os.path.join(classifications, 'sub')
    os.makedirs(csv_dir)
    csv = os.path.join(csv_dir, 'x.csv')
    open(csv, 'w').close()
    state.save_session(sid(), classifications, position_filename='a.fits', csv=csv)

    with open(state.sessions_path(classifications)) as f:
        raw = json.load(f)
    stored_csv = next(iter(raw['sessions'].values()))['csv']
    assert not os.path.isabs(stored_csv)
    assert stored_csv == 'sub/x.csv'


def test_path_is_stored_relative_and_slash_separated(tmpdir):
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data', 'stamps')
    os.makedirs(classifications)
    os.makedirs(data)
    state.save_session(sid(path=data), classifications, position_filename='a.fits')

    with open(state.sessions_path(classifications)) as f:
        raw = json.load(f)
    entry = next(iter(raw['sessions'].values()))
    assert entry['path_form'] == 'relative'
    assert entry['path'] == '../data/stamps'
    assert '\\' not in entry['path']


# --- legacy key fallback ----------------------------------------------------

def test_legacy_key_fallback_is_honoured(tmpdir):
    """A store written by the pre-2b-i code still resumes at the original location.

    Hand-build a store shaped exactly like the old code would have written it
    -- four keys, no path_form, realpath identity, absolute csv -- and check
    that both load_session and resolve_position still accept it.
    """
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(classifications)
    os.makedirs(data)
    csv = os.path.join(classifications, 'x.csv')
    open(csv, 'w').close()

    session = sid(path=data)
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='c.fits',
                        csv=os.path.realpath(csv), last_opened='2020-01-01T00:00:00Z')
    with open(state.sessions_path(classifications), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    entry = state.load_session(session, classifications)
    assert entry is not None and entry['position_filename'] == 'c.fits'
    assert state.resolve_position(entry, LIST, csv=csv, classifications_dir=classifications) == 2


def test_forget_session_drops_a_legacy_keyed_entry(tmpdir):
    classifications = os.path.join(tmpdir, 'Classifications')
    os.makedirs(classifications)
    session = sid()
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='a.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(state.sessions_path(classifications), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    assert state.forget_session(session, classifications) is True
    assert state.load_session(session, classifications) is None
    assert state.forget_session(session, classifications) is False


# --- migrate_sessions_store (pre-2b-i, CWD-relative sessions.json) --------

def test_migrate_sessions_store_imports_and_rekeys(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(data)

    session = sid(path=data)
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='c.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is True
    entry = state.load_session(session, classifications)
    assert entry is not None
    assert entry['position_filename'] == 'c.fits'
    assert entry['path_form'] == 'relative'


def test_migrate_sessions_store_default_legacy_dir_is_the_cwd_not_config_dir(tmpdir):
    """Regression: the default `legacy_dir` must be the CWD, not `DEFAULT_CONFIG_DIR`.

    `DEFAULT_CONFIG_DIR` now means the per-user config directory (2b-iii, for
    preferences) -- a real move. The pre-2b-i `sessions.json`, though, never
    lived anywhere but the CWD, and still doesn't: mosaic.py and
    single_viewer.py call `migrate_sessions_store(self.classifications_dir)`
    with no `legacy_dir` at all and mean "check the CWD". Tying the two
    defaults together (as a naive `legacy_dir = DEFAULT_CONFIG_DIR if ...`
    would) would silently stop that real call site from ever finding a legacy
    store again, with every other test in this file passing `legacy_dir`
    explicitly and so never exercising the default at all.
    """
    cwd_dir = os.path.join(tmpdir, 'cwd')
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(cwd_dir)
    os.makedirs(data)

    session = sid(path=data)
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='c.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(cwd_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    original_cwd = os.getcwd()
    os.chdir(cwd_dir)
    try:
        # No legacy_dir passed -- exactly how mosaic.py/single_viewer.py call this.
        assert state.migrate_sessions_store(classifications) is True
    finally:
        os.chdir(original_cwd)

    entry = state.load_session(session, classifications)
    assert entry is not None
    assert entry['position_filename'] == 'c.fits'


def test_migrate_sessions_store_is_a_noop_when_anchor_is_the_legacy_dir(tmpdir):
    "The overwhelmingly common case: classifications dir == CWD == legacy dir."
    state.save_session(sid(), tmpdir, position_filename='a.fits')
    assert state.migrate_sessions_store(tmpdir, legacy_dir=tmpdir) is False


def test_migrate_sessions_store_never_overwrites_a_newer_entry(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')
    os.makedirs(classifications)

    session = sid()
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='old.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    # A newer entry already exists at the current key in the target store.
    state.save_session(session, classifications, position_filename='new.fits')

    state.migrate_sessions_store(classifications, legacy_dir=legacy_dir)
    assert state.load_session(session, classifications)['position_filename'] == 'new.fits'


def test_migrate_sessions_store_runs_only_once(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')

    session = sid()
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='a.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is True
    assert state.forget_session(session, classifications) is True

    # A second migration must not resurrect the position --reset-position just cleared.
    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is False
    assert state.load_session(session, classifications) is None


def test_migrate_sessions_store_does_not_resurrect_through_a_corrupt_store(tmpdir):
    """BUGS.md item 8: the once-only record lives in the store it protects.

    `load_store` falls back to an empty store when `sessions.json` is missing
    *or* unparseable, so a corrupt store used to take `legacy_imports` with it
    and the next startup re-imported the CWD file -- putting back a position
    the user had deliberately cleared with `--reset-position`.
    """
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')

    session = sid()
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='a.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is True
    assert state.forget_session(session, classifications) is True

    target = state.sessions_path(classifications)
    with open(target, 'w') as f:
        f.write('{ truncated')

    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is False
    assert state.load_session(session, classifications) is None
    # The unreadable file is kept rather than silently overwritten, ...
    assert os.path.exists(target + '.corrupt')
    # ... and the mark is on disk, so a third startup does not ask again.
    assert legacy_key not in state.load_store(classifications)['sessions']
    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is False
    assert state.load_session(session, classifications) is None


def test_migrate_sessions_store_still_imports_into_a_missing_store(tmpdir):
    "The corrupt-store guard must not swallow the ordinary first-run import."
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')
    os.makedirs(classifications)
    assert not os.path.exists(state.sessions_path(classifications))

    session = sid()
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='a.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {state._legacy_session_key(session): legacy_entry}}, f)

    assert state.migrate_sessions_store(classifications, legacy_dir=legacy_dir) is True
    assert state.load_session(session, classifications)['position_filename'] == 'a.fits'


def test_empty_store_carries_the_legacy_imports_record(tmpdir):
    "Same shape whichever branch of _load_store produced it (BUGS.md item 8)."
    assert state._empty_store()['legacy_imports'] == []
    assert state.load_store(tmpdir)['legacy_imports'] == []

    # A store on disk whose record is present, or the wrong type, or absent.
    path = state.sessions_path(tmpdir)
    for stored, expected in ((['/x/sessions.json'], ['/x/sessions.json']), ('nonsense', []),
                             (None, [])):
        payload = {'version': 1, 'sessions': {}}
        if stored is not None:
            payload['legacy_imports'] = stored
        with open(path, 'w') as f:
            json.dump(payload, f)
        assert state.load_store(tmpdir)['legacy_imports'] == expected


def test_save_session_keeps_the_legacy_imports_record(tmpdir):
    "Read-modify-write must not drop the guard on an ordinary save."
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')
    session = sid()
    with open(os.path.join(legacy_dir, state.SESSIONS_FILENAME), 'w') as f:
        json.dump({'version': 1, 'sessions': {}}, f)

    state.migrate_sessions_store(classifications, legacy_dir=legacy_dir)
    state.save_session(session, classifications, position_filename='a.fits')
    assert state.load_store(classifications)['legacy_imports'] == [
        os.path.join(paths.absolute(legacy_dir), state.SESSIONS_FILENAME)]


def test_migrate_legacy_config_requires_a_classifications_dir(tmpdir):
    """BUGS.md item 9: the `classifications_dir=None` default was unusable.

    Session state has no default anchor, so the default reached
    `sessions_path(None)` and raised TypeError. Both call sites always passed
    it; the dead default is now gone, and omitting it is a TypeError at the
    call, not three frames down.
    """
    legacy = write_legacy(tmpdir, {'name': '', 'counter': 1})
    try:
        state.migrate_legacy_config(legacy, sid(), DEFAULTS, LIST, lambda d: d['counter'],
                                    config_dir=tmpdir)
    except TypeError as exc:
        assert 'classifications_dir' in str(exc)
    else:
        assert False, 'expected a TypeError naming the missing argument'
    assert os.path.exists(legacy)  # nothing half-migrated


def test_migrate_sessions_store_leaves_the_legacy_file_in_place(tmpdir):
    legacy_dir = os.path.join(tmpdir, 'legacy')
    os.makedirs(legacy_dir)
    classifications = os.path.join(tmpdir, 'Classifications')

    session = sid()
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='a.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    legacy_file = os.path.join(legacy_dir, state.SESSIONS_FILENAME)
    with open(legacy_file, 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    state.migrate_sessions_store(classifications, legacy_dir=legacy_dir)
    assert os.path.exists(legacy_file)


# --- session config (BUGS.md item 14: lobby per-session state) ------------

def test_save_and_load_session_config_roundtrip(tmpdir):
    config = {'scheme_rows': [{'key': 'a'}], 'main_band': 'r'}
    state.save_session_config(sid(), tmpdir, config)
    assert state.load_session_config(sid(), tmpdir) == config
    # An identity that has never saved a config at all.
    assert state.load_session_config(sid(name='never'), tmpdir) is None
    # An identity that has only ever had a position saved, no config.
    state.save_session(sid(tool='mosaic'), tmpdir, position_filename='a.fits')
    assert state.load_session_config(sid(tool='mosaic'), tmpdir) is None


def test_save_session_config_merges_with_an_existing_position(tmpdir):
    """The whole reason `_update_session` exists.

    Before it, `save_session` rebuilt the entry from scratch, so a config save
    after a position save (or vice versa -- see the next test) would have
    erased whichever half was already on disk.
    """
    state.save_session(sid(), tmpdir, position_filename='b.fits')
    state.save_session_config(sid(), tmpdir, {'main_band': 'g'})
    entry = state.load_session(sid(), tmpdir)
    assert entry['position_filename'] == 'b.fits'
    assert entry['config'] == {'main_band': 'g'}


def test_save_session_then_config_also_merges_in_the_other_order(tmpdir):
    "Reverse order: a later position save must not erase an earlier config."
    state.save_session_config(sid(), tmpdir, {'main_band': 'g'})
    state.save_session(sid(), tmpdir, position_filename='b.fits')
    entry = state.load_session(sid(), tmpdir)
    assert entry['position_filename'] == 'b.fits'
    assert entry['config'] == {'main_band': 'g'}


def test_save_session_config_is_scoped_by_identity(tmpdir):
    "Different tool, name, or seed each get their own config."
    state.save_session_config(sid(tool='lobby'), tmpdir, {'main_band': 'g'})
    state.save_session_config(sid(tool='single'), tmpdir, {'main_band': 'r'})
    state.save_session_config(sid(tool='lobby', name='vfin'), tmpdir, {'main_band': 'i'})
    state.save_session_config(sid(tool='lobby', seed=7), tmpdir, {'main_band': 'z'})

    assert state.load_session_config(sid(tool='lobby'), tmpdir) == {'main_band': 'g'}
    assert state.load_session_config(sid(tool='single'), tmpdir) == {'main_band': 'r'}
    assert state.load_session_config(sid(tool='lobby', name='vfin'), tmpdir) == \
        {'main_band': 'i'}
    assert state.load_session_config(sid(tool='lobby', seed=7), tmpdir) == {'main_band': 'z'}


def test_load_session_config_returns_a_copy(tmpdir):
    "Mutating what comes back must not reach what is on disk."
    state.save_session_config(sid(), tmpdir, {'scheme_rows': [1, 2, 3]})
    loaded = state.load_session_config(sid(), tmpdir)
    loaded['scheme_rows'].append(4)
    loaded['extra'] = True
    assert state.load_session_config(sid(), tmpdir) == {'scheme_rows': [1, 2, 3]}


def test_list_sessions_orders_most_recently_opened_first(tmpdir):
    "Also: an entry with no last_opened sorts last, and every entry carries its key."
    original_now = state._now
    try:
        state._now = lambda: '2020-01-01T00:00:00Z'
        state.save_session(sid(name='old'), tmpdir, position_filename='a.fits')
        state._now = lambda: '2026-01-01T00:00:00Z'
        state.save_session(sid(name='new'), tmpdir, position_filename='b.fits')
    finally:
        state._now = original_now

    # An entry with no last_opened at all, added directly to the store.
    store = state.load_store(tmpdir)
    store['sessions']['no-timestamp'] = dict(
        state.session_identity(sid(name='stale'), tmpdir), position_filename='c.fits')
    with open(state.sessions_path(tmpdir), 'w') as f:
        json.dump(store, f)

    listed = state.list_sessions(tmpdir)
    assert [e['name'] for e in listed] == ['new', 'old', 'stale']
    assert all('key' in e for e in listed)
    assert listed[0]['key'] == state.session_key(sid(name='new'), tmpdir)


def test_list_sessions_filters_by_tool(tmpdir):
    state.save_session(sid(tool='single'), tmpdir, position_filename='a.fits')
    state.save_session(sid(tool='mosaic'), tmpdir, position_filename='b.fits')
    assert [e['tool'] for e in state.list_sessions(tmpdir, tool='mosaic')] == ['mosaic']


def test_list_sessions_on_an_empty_or_missing_store_is_empty(tmpdir):
    assert state.list_sessions(tmpdir) == []
    assert state.list_sessions(os.path.join(tmpdir, 'nonexistent')) == []


def test_session_id_from_entry_roundtrips_through_list_sessions(tmpdir):
    """The portable case the module docstring is about: data beside the anchor.

    `path` in the stored entry is relative to the classifications dir, so
    `session_id_from_entry` has to resolve it back before it is usable -- a
    naive implementation that handed the stored (relative) value straight to
    `SessionId` would break exactly this case.
    """
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(classifications)
    os.makedirs(data)

    session = sid(path=data, tool='lobby', name='vfin', seed=3)
    config = {'main_band': 'r'}
    state.save_session_config(session, classifications, config)

    listed = state.list_sessions(classifications, tool='lobby')
    assert len(listed) == 1
    entry = listed[0]
    assert entry['path_form'] == 'relative'

    recovered = state.session_id_from_entry(entry, classifications)
    assert recovered.tool == session.tool
    assert recovered.name == session.name
    assert recovered.seed == session.seed
    assert paths.absolute(recovered.path) == paths.absolute(session.path)
    assert state.load_session_config(recovered, classifications) == config


def test_session_id_from_entry_rejects_junk(tmpdir):
    assert state.session_id_from_entry(None, tmpdir) is None
    assert state.session_id_from_entry('not a dict', tmpdir) is None
    assert state.session_id_from_entry({'path': '/data/stamps'}, tmpdir) is None  # no tool
    assert state.session_id_from_entry({'tool': 'single'}, tmpdir) is None        # no path


def test_save_session_config_rewrites_a_legacy_keyed_entry(tmpdir):
    """Same legacy pickup as `save_session` (see test_legacy_key_fallback_is_honoured).

    A legacy-keyed entry is found the way `load_session` finds it, merged into
    an entry under the current key, and the legacy copy is then dropped -- one
    identity, one record. Before `_update_session`, a pre-2b-i entry survived
    every save as a stale duplicate competing for a slot under `prune`.
    """
    classifications = os.path.join(tmpdir, 'Classifications')
    data = os.path.join(tmpdir, 'data')
    os.makedirs(classifications)
    os.makedirs(data)

    session = sid(path=data)
    legacy_key = state._legacy_session_key(session)
    legacy_entry = dict(state._legacy_session_identity(session), position_filename='c.fits',
                        csv=None, last_opened='2020-01-01T00:00:00Z')
    with open(state.sessions_path(classifications), 'w') as f:
        json.dump({'version': 1, 'sessions': {legacy_key: legacy_entry}}, f)

    state.save_session_config(session, classifications, {'main_band': 'g'})

    sessions = state.load_store(classifications)['sessions']
    current_key = state.session_key(session, classifications)
    assert current_key in sessions
    entry = sessions[current_key]
    assert entry['config'] == {'main_band': 'g'}
    assert entry['position_filename'] == 'c.fits'   # the legacy position survives the merge

    # The stale legacy-keyed copy is gone -- everything it held is in `entry`.
    assert legacy_key not in sessions
    assert len(sessions) == 1


def test_saving_config_prunes_the_store(tmpdir):
    for i in range(5):
        state.save_session_config(sid(seed=i), tmpdir, {'main_band': 'g'}, max_sessions=3)
    assert len(state.load_store(tmpdir)['sessions']) == 3


# --- local-first preferences (2b-v) ----------------------------------------

class _env:
    "Set env vars for the duration of a with-block; test_state.py has no monkeypatch."

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.saved[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc):
        for name, old in self.saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def test_preferences_fall_back_to_the_user_dir_but_save_lands_locally(tmpdir):
    """The user dir *seeds* a fresh workspace; it never becomes the write target.

    This is the whole shape of 2b-v: someone with preferences from a 2b-iii-era
    run keeps them on first open, and the moment they change one, the change
    belongs to this workspace alone -- the shared copy other workspaces read
    must come back unmodified.
    """
    user_dir = os.path.join(tmpdir, 'user')
    local = os.path.join(tmpdir, 'workspace', '.qtstamp')
    os.makedirs(user_dir)
    user_file = os.path.join(user_dir, 'preferences_single.json')
    with open(user_file, 'w') as f:
        json.dump({'colormap': 'viridis', 'scale': 'log'}, f)

    with _env(QTSTAMP_CONFIG_DIR=user_dir, QTSTAMP_STATE_DIR=None):
        assert state.load_preferences(SINGLE, DEFAULTS, config_dir=local)['colormap'] == 'viridis'
        assert state.save_preferences(SINGLE, dict(DEFAULTS, colormap='gray'),
                                      config_dir=local) is True
        assert os.path.exists(os.path.join(local, 'preferences_single.json'))
        assert state.load_preferences(SINGLE, DEFAULTS, config_dir=local)['colormap'] == 'gray'

    with open(user_file) as f:
        assert json.load(f)['colormap'] == 'viridis'


def test_reset_preferences_removes_only_the_local_copy(tmpdir):
    "After a reset the workspace falls back to the seed, exactly like a fresh one."
    user_dir = os.path.join(tmpdir, 'user')
    local = os.path.join(tmpdir, 'workspace', '.qtstamp')
    os.makedirs(user_dir)
    with open(os.path.join(user_dir, 'preferences_single.json'), 'w') as f:
        json.dump({'colormap': 'viridis'}, f)

    with _env(QTSTAMP_CONFIG_DIR=user_dir, QTSTAMP_STATE_DIR=None):
        state.save_preferences(SINGLE, dict(DEFAULTS, colormap='gray'), config_dir=local)
        assert state.reset_preferences(SINGLE, config_dir=local) is True
        assert state.load_preferences(SINGLE, DEFAULTS, config_dir=local)['colormap'] == 'viridis'
        assert os.path.exists(os.path.join(user_dir, 'preferences_single.json'))


def test_save_preferences_gives_up_quietly_on_a_read_only_state_dir(tmpdir):
    """Local-or-nothing: no crash, and no fallback write to somewhere unasked-for.

    The user's rule for 2b-v -- losing settings persistence is acceptable,
    silently relocating their settings is not.
    """
    if os.geteuid() == 0:
        return
    workspace = os.path.join(tmpdir, 'ro')
    user_dir = os.path.join(tmpdir, 'user')
    os.makedirs(workspace)
    os.chmod(workspace, 0o500)
    local = os.path.join(workspace, '.qtstamp')
    try:
        with _env(QTSTAMP_CONFIG_DIR=user_dir, QTSTAMP_STATE_DIR=None):
            assert state.save_preferences(SINGLE, dict(DEFAULTS, colormap='gray'),
                                          config_dir=local) is False
            # Nothing was written anywhere else to compensate.
            assert not os.path.exists(user_dir)
            assert state.load_preferences(SINGLE, DEFAULTS, config_dir=local) == DEFAULTS
    finally:
        os.chmod(workspace, 0o700)


def test_migrate_preferences_file_picks_up_the_dotted_name_in_the_state_dir(tmpdir):
    "2b-v dropped the leading dot; a state dir written by 2b-iii still has it."
    local = os.path.join(tmpdir, '.qtstamp')
    os.makedirs(local)
    dotted = os.path.join(local, '.preferences_single.json')
    with open(dotted, 'w') as f:
        json.dump({'colormap': 'viridis'}, f)

    with _env(QTSTAMP_CONFIG_DIR=os.path.join(tmpdir, 'nowhere'), QTSTAMP_STATE_DIR=None):
        assert state.migrate_preferences_file(SINGLE, legacy_dir=tmpdir, config_dir=local) is True
        assert not os.path.exists(dotted)
        assert state.load_preferences(SINGLE, DEFAULTS, config_dir=local)['colormap'] == 'viridis'


if __name__ == '__main__':
    import inspect
    import tempfile
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = 0
    for name, func in tests:
        try:
            if 'tmpdir' in inspect.signature(func).parameters:
                with tempfile.TemporaryDirectory() as d:
                    func(d)
            else:
                func()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
