"""Tests for state.py -- the session store and preferences split.

state.py is deliberately Qt-free and pure over a directory path, so all of this
runs headless. Written as plain pytest functions, but also runnable directly
(`python tests/test_state.py`) until the suite of item 3 lands.
"""

import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    state.save_session(sid(), position_filename='a.fits', config_dir=tmpdir)
    for path in (state.preferences_path(SINGLE, tmpdir), state.sessions_path(tmpdir)):
        assert stat.S_IMODE(os.stat(path).st_mode) == expected, path


def test_rewriting_the_store_preserves_its_mode(tmpdir):
    if not POSIX:
        return
    state.save_session(sid(), position_filename='a.fits', config_dir=tmpdir)
    store = state.sessions_path(tmpdir)
    os.chmod(store, 0o640)
    state.save_session(sid(), position_filename='b.fits', config_dir=tmpdir)
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o640
    assert state.load_session(sid(), tmpdir)['position_filename'] == 'b.fits'


def test_no_scratch_files_are_left_behind(tmpdir):
    state.save_preferences(SINGLE, {'colormap': 'viridis'}, config_dir=tmpdir)
    state.save_session(sid(), position_filename='a.fits', config_dir=tmpdir)
    assert not [f for f in os.listdir(tmpdir) if f.startswith('.tmp-')]


# --- identity -------------------------------------------------------------

def test_identity_keys_on_tool_path_name_seed():
    base = state.session_key(sid())
    assert state.session_key(sid(tool='mosaic')) != base
    assert state.session_key(sid(path='/data/other')) != base
    assert state.session_key(sid(name='vfin')) != base
    assert state.session_key(sid(seed=812)) != base
    assert state.session_key(sid()) == base


def test_identity_ignores_grid_shape_and_file_count():
    "Neither appears in SessionId at all -- the position is a filename."
    assert set(state.session_identity(sid())) == {'tool', 'path', 'name', 'seed'}


def test_identity_normalizes_path_and_empty_name():
    assert state.session_key(sid(path='/data/stamps/')) == state.session_key(sid())
    assert state.session_key(sid(name=None)) == state.session_key(sid(name=''))


# --- session store --------------------------------------------------------

def test_save_and_load_roundtrip(tmpdir):
    state.save_session(sid(), position_filename='c.fits', csv='/abs/x.csv', config_dir=tmpdir)
    entry = state.load_session(sid(), config_dir=tmpdir)
    assert entry['position_filename'] == 'c.fits'
    assert entry['csv'] == '/abs/x.csv'
    assert entry['last_opened'].endswith('Z')
    # The identity is stored in plain text beside its hash.
    assert entry['name'] == '' and entry['tool'] == SINGLE


def test_sessions_of_different_identities_do_not_collide(tmpdir):
    state.save_session(sid(seed=1), position_filename='a.fits', config_dir=tmpdir)
    state.save_session(sid(seed=2), position_filename='d.fits', config_dir=tmpdir)
    assert state.load_session(sid(seed=1), config_dir=tmpdir)['position_filename'] == 'a.fits'
    assert state.load_session(sid(seed=2), config_dir=tmpdir)['position_filename'] == 'd.fits'


def test_the_two_viewers_share_one_file_without_clobbering(tmpdir):
    state.save_session(sid(tool='single'), position_filename='a.fits', config_dir=tmpdir)
    state.save_session(sid(tool='mosaic'), position_filename='c.fits', config_dir=tmpdir)
    assert state.load_session(sid(tool='single'), config_dir=tmpdir)['position_filename'] == 'a.fits'
    assert len(state.load_store(config_dir=tmpdir)['sessions']) == 2


def test_missing_session_is_none(tmpdir):
    assert state.load_session(sid(), config_dir=tmpdir) is None


def test_forget_session(tmpdir):
    state.save_session(sid(), position_filename='b.fits', config_dir=tmpdir)
    assert state.forget_session(sid(), config_dir=tmpdir) is True
    assert state.load_session(sid(), config_dir=tmpdir) is None
    assert state.forget_session(sid(), config_dir=tmpdir) is False


def test_corrupt_store_reads_as_empty(tmpdir):
    with open(state.sessions_path(tmpdir), 'w') as f:
        f.write('{not json')
    assert state.load_store(config_dir=tmpdir) == {'version': state.STORE_VERSION, 'sessions': {}}
    state.save_session(sid(), position_filename='a.fits', config_dir=tmpdir)
    assert state.load_session(sid(), config_dir=tmpdir)['position_filename'] == 'a.fits'


def test_store_writes_leave_no_temp_files(tmpdir):
    state.save_session(sid(), position_filename='a.fits', config_dir=tmpdir)
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
        state.save_session(sid(seed=i), position_filename='a.fits',
                           config_dir=tmpdir, max_sessions=3)
    assert len(state.load_store(config_dir=tmpdir)['sessions']) == 3


# --- position resolution --------------------------------------------------

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
    state.save_session(sid(seed=1), position_filename='c.fits', config_dir=tmpdir)
    # New seed: no saved position ...
    assert state.resolve_position(state.load_session(sid(seed=2), config_dir=tmpdir), LIST) == 0
    # ... but the colormap is still the user's.
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['colormap'] == 'viridis'


# --- migration ------------------------------------------------------------

def write_legacy(tmpdir, payload, filename='.config.json'):
    path = os.path.join(tmpdir, filename)
    with open(path, 'w') as f:
        json.dump(payload, f)
    return path


def test_migration_splits_the_old_blob(tmpdir):
    legacy = write_legacy(tmpdir, {'name': 'vfin', 'counter': 2, 'scale': 'sqrt',
                                   'colormap': 'viridis'})
    migrated = state.migrate_legacy_config(
        legacy, sid(name='vfin'), DEFAULTS, LIST,
        lambda d: d['counter'] if d.get('name') == 'vfin' else None,
        csv='/abs/x.csv', config_dir=tmpdir)
    assert migrated is True
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'sqrt'
    assert state.load_session(sid(name='vfin'), config_dir=tmpdir)['position_filename'] == 'c.fits'
    # Kept, not deleted.
    assert not os.path.exists(legacy) and os.path.exists(legacy + '.bak')


def test_migration_drops_a_position_it_cannot_vouch_for(tmpdir):
    legacy = write_legacy(tmpdir, {'name': 'other', 'counter': 2, 'scale': 'sqrt'})
    state.migrate_legacy_config(legacy, sid(name='vfin'), DEFAULTS, LIST,
                                lambda d: None, config_dir=tmpdir)
    assert state.load_session(sid(name='vfin'), config_dir=tmpdir) is None
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'sqrt'


def test_migration_ignores_an_out_of_range_index(tmpdir):
    legacy = write_legacy(tmpdir, {'name': '', 'counter': 99})
    state.migrate_legacy_config(legacy, sid(), DEFAULTS, LIST, lambda d: d['counter'],
                                config_dir=tmpdir)
    assert state.load_session(sid(), config_dir=tmpdir) is None


def test_migration_runs_only_once(tmpdir):
    state.save_preferences(SINGLE, dict(DEFAULTS, scale='cbrt'), config_dir=tmpdir)
    legacy = write_legacy(tmpdir, {'name': '', 'counter': 2, 'scale': 'sqrt'})
    assert state.migrate_legacy_config(legacy, sid(), DEFAULTS, LIST, lambda d: d['counter'],
                                       config_dir=tmpdir) is False
    assert state.load_preferences(SINGLE, DEFAULTS, config_dir=tmpdir)['scale'] == 'cbrt'
    assert os.path.exists(legacy)  # untouched


def test_migration_without_a_legacy_file_is_a_noop(tmpdir):
    assert state.migrate_legacy_config(os.path.join(tmpdir, 'nope.json'), sid(), DEFAULTS,
                                       LIST, lambda d: 0, config_dir=tmpdir) is False
    assert not os.path.exists(state.preferences_path(SINGLE, tmpdir))


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
