"""Tests for paths.py -- the portable path arithmetic behind session identity.

paths.py is pure os.path arithmetic (no Qt, no astropy), so this runs headless.
Plain pytest functions, also runnable directly (`python tests/test_paths.py`).
"""

import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths


# --- relative_to / resolve_against round trips -----------------------------

def test_data_inside_the_anchor_round_trips(tmpdir):
    anchor = join(tmpdir, 'Classifications')
    data = join(tmpdir, 'Classifications', 'stamps')
    os.makedirs(data)
    value, form = paths.relative_to(data, anchor)
    assert form == paths.RELATIVE
    assert value == 'stamps'
    assert paths.resolve_against(value, anchor) == paths.absolute(data)


def test_data_beside_the_anchor_round_trips(tmpdir):
    "A `..`-heavy relative path: data is a sibling of the classifications dir."
    anchor = join(tmpdir, 'Classifications')
    data = join(tmpdir, 'data')
    os.makedirs(anchor)
    os.makedirs(data)
    value, form = paths.relative_to(data, anchor)
    assert form == paths.RELATIVE
    assert value.startswith('../')
    assert paths.resolve_against(value, anchor) == paths.absolute(data)


def test_data_far_above_the_anchor_round_trips(tmpdir):
    anchor = join(tmpdir, 'a', 'b', 'Classifications')
    data = join(tmpdir, 'data')
    os.makedirs(anchor)
    os.makedirs(data)
    value, form = paths.relative_to(data, anchor)
    assert form == paths.RELATIVE
    assert value.count('../') >= 2
    assert paths.resolve_against(value, anchor) == paths.absolute(data)


def test_the_anchor_itself_round_trips(tmpdir):
    anchor = join(tmpdir, 'Classifications')
    os.makedirs(anchor)
    value, form = paths.relative_to(anchor, anchor)
    assert form == paths.RELATIVE
    assert value == '.'
    assert paths.resolve_against(value, anchor) == paths.absolute(anchor)


def test_relative_value_uses_forward_slashes(tmpdir):
    anchor = join(tmpdir, 'Classifications')
    data = join(tmpdir, 'a', 'b', 'stamps')
    os.makedirs(anchor)
    os.makedirs(data)
    value, form = paths.relative_to(data, anchor)
    assert form == paths.RELATIVE
    assert value == '../a/b/stamps'
    assert '\\' not in value


def test_resolve_against_an_absolute_stored_value_passes_through(tmpdir):
    data = join(tmpdir, 'somewhere', 'else')
    os.makedirs(data)
    # An absolute, portable value (as a legacy or cross-drive entry would store).
    stored = paths.to_portable(paths.absolute(data))
    assert paths.resolve_against(stored, join(tmpdir, 'Classifications')) == paths.absolute(data)


# --- resolve_classifications_dir -------------------------------------------

def test_resolve_classifications_dir_default_is_cwd_classifications(tmpdir):
    cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        assert paths.resolve_classifications_dir(None) == \
            os.path.join(os.path.abspath(tmpdir), 'Classifications')
    finally:
        os.chdir(cwd)


def test_resolve_classifications_dir_relative_argument_resolves_against_cwd(tmpdir):
    cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        assert paths.resolve_classifications_dir('MyDir') == \
            os.path.join(os.path.abspath(tmpdir), 'MyDir')
    finally:
        os.chdir(cwd)


def test_resolve_classifications_dir_expands_user():
    home = os.path.expanduser('~')
    assert paths.resolve_classifications_dir('~/SomeDir') == \
        os.path.abspath(os.path.join(home, 'SomeDir'))


def test_resolve_classifications_dir_absolute_argument_passes_through(tmpdir):
    target = join(tmpdir, 'Elsewhere')
    assert paths.resolve_classifications_dir(target) == os.path.abspath(target)


# --- user_config_dir / user_cache_dir --------------------------------------

def _clear_dir_env(monkeypatch):
    for var in ('QTSTAMP_CONFIG_DIR', 'QTSTAMP_CACHE_DIR', 'XDG_CONFIG_HOME',
                'XDG_CACHE_HOME', 'APPDATA', 'LOCALAPPDATA'):
        monkeypatch.delenv(var, raising=False)


def test_user_config_dir_env_override_wins_outright(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    override = join(tmpdir, 'my-config')
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', override)
    # A conflicting XDG value is also set, to prove the override wins outright
    # rather than merely being consulted first on this platform.
    monkeypatch.setenv('XDG_CONFIG_HOME', join(tmpdir, 'not-this-one'))
    monkeypatch.setattr(paths.sys, 'platform', 'linux')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    assert paths.user_config_dir() == paths.absolute(override)


def test_user_cache_dir_env_override_wins_outright(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    override = join(tmpdir, 'my-cache')
    monkeypatch.setenv('QTSTAMP_CACHE_DIR', override)
    monkeypatch.setenv('XDG_CACHE_HOME', join(tmpdir, 'not-this-one'))
    monkeypatch.setattr(paths.sys, 'platform', 'linux')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    assert paths.user_cache_dir() == paths.absolute(override)


def test_user_config_dir_posix_uses_xdg_config_home(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'linux')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmpdir))
    assert paths.user_config_dir() == join(tmpdir, 'qtstamp')


def test_user_config_dir_posix_falls_back_to_dot_config(monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'linux')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    assert paths.user_config_dir() == os.path.join(
        os.path.expanduser('~/.config'), 'qtstamp')


def test_user_cache_dir_posix_uses_xdg_cache_home(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'linux')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmpdir))
    assert paths.user_cache_dir() == join(tmpdir, 'qtstamp')


def test_user_config_dir_macos_uses_application_support(monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'darwin')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    assert paths.user_config_dir() == os.path.join(
        os.path.expanduser('~/Library/Application Support'), 'qtstamp')


def test_user_cache_dir_macos_uses_caches(monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'darwin')
    monkeypatch.setattr(paths.os, 'name', 'posix')
    assert paths.user_cache_dir() == os.path.join(
        os.path.expanduser('~/Library/Caches'), 'qtstamp')


def test_user_config_dir_windows_uses_appdata(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'win32')
    monkeypatch.setattr(paths.os, 'name', 'nt')
    monkeypatch.setenv('APPDATA', str(tmpdir))
    assert paths.user_config_dir() == join(tmpdir, 'qtstamp')


def test_user_cache_dir_windows_uses_localappdata(tmpdir, monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.setattr(paths.sys, 'platform', 'win32')
    monkeypatch.setattr(paths.os, 'name', 'nt')
    monkeypatch.setenv('LOCALAPPDATA', str(tmpdir))
    assert paths.user_cache_dir() == join(tmpdir, 'qtstamp')


# --- migrate_once -----------------------------------------------------------

def test_migrate_once_moves_a_file(tmpdir):
    legacy = join(tmpdir, 'legacy.json')
    target = join(tmpdir, 'sub', 'target.json')
    with open(legacy, 'w') as f:
        f.write('{}')
    assert paths.migrate_once(legacy, target) is True
    assert not os.path.exists(legacy)
    assert os.path.exists(target)


def test_migrate_once_moves_a_directory_with_contents(tmpdir):
    legacy = join(tmpdir, 'legacy_dir')
    target = join(tmpdir, 'new_dir')
    os.makedirs(legacy)
    with open(join(legacy, 'inner.txt'), 'w') as f:
        f.write('hello')
    assert paths.migrate_once(legacy, target) is True
    assert not os.path.exists(legacy)
    with open(join(target, 'inner.txt')) as f:
        assert f.read() == 'hello'


def test_migrate_once_is_a_noop_when_legacy_is_missing(tmpdir):
    legacy = join(tmpdir, 'nope')
    target = join(tmpdir, 'target')
    assert paths.migrate_once(legacy, target) is False
    assert not os.path.exists(target)


def test_migrate_once_is_a_noop_when_target_already_exists(tmpdir):
    legacy = join(tmpdir, 'legacy.json')
    target = join(tmpdir, 'target.json')
    with open(legacy, 'w') as f:
        f.write('legacy-content')
    with open(target, 'w') as f:
        f.write('newer-content')
    assert paths.migrate_once(legacy, target) is False
    # Legacy is left untouched, not deleted.
    assert os.path.exists(legacy)
    with open(target) as f:
        assert f.read() == 'newer-content'


def test_migrate_once_is_a_noop_when_legacy_equals_target(tmpdir):
    same = join(tmpdir, 'same.json')
    with open(same, 'w') as f:
        f.write('{}')
    assert paths.migrate_once(same, same) is False
    assert os.path.exists(same)


def test_migrate_once_returns_false_instead_of_raising_when_the_target_is_unwritable(tmpdir):
    """Every migrate_once call site runs at import or in __init__, before any
    window exists, so a raise here is a hard crash with no UI to report it."""
    if os.geteuid() == 0:
        return
    legacy = join(tmpdir, 'legacy.json')
    with open(legacy, 'w') as f:
        f.write('{}')
    ro = join(tmpdir, 'ro')
    os.makedirs(ro)
    os.chmod(ro, 0o500)
    try:
        assert paths.migrate_once(legacy, join(ro, 'target.json')) is False
        assert os.path.exists(legacy)  # left where it was, for the next writable run
    finally:
        os.chmod(ro, 0o700)


# --- state_dir / search path / ensure_dir / cache_dir (2b-v, local-first) ---

def _clear_state_env(monkeypatch):
    _clear_dir_env(monkeypatch)
    monkeypatch.delenv(paths.STATE_DIR_ENV, raising=False)


def test_state_dir_defaults_to_dot_qtstamp_in_the_cwd(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    assert paths.state_dir() == join(paths.absolute(tmpdir), paths.LOCAL_STATE_DIRNAME)


def test_state_dir_does_not_create_anything(tmpdir, monkeypatch):
    "Resolving is pure -- --print-command must not leave a .qtstamp behind."
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    paths.state_dir()
    assert os.listdir(tmpdir) == []


def test_state_dir_override_beats_the_env_which_beats_the_cwd(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    monkeypatch.setenv(paths.STATE_DIR_ENV, join(tmpdir, 'from-env'))
    assert paths.state_dir() == join(paths.absolute(tmpdir), 'from-env')
    assert paths.state_dir(override=join(tmpdir, 'explicit')) == \
        join(paths.absolute(tmpdir), 'explicit')


def test_config_search_path_is_local_then_user_then_packaged(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', join(tmpdir, 'user'))
    assert paths.config_search_path() == [
        join(paths.absolute(tmpdir), paths.LOCAL_STATE_DIRNAME),
        join(paths.absolute(tmpdir), 'user'),
        paths.packaged_defaults_dir(),
    ]


def test_config_search_path_dedupes_when_global_makes_two_entries_equal(tmpdir, monkeypatch):
    "--global passes user_config_dir() as the override; the path must not list it twice."
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', join(tmpdir, 'user'))
    search = paths.config_search_path(override=paths.user_config_dir())
    assert search == [join(paths.absolute(tmpdir), 'user'), paths.packaged_defaults_dir()]


def test_find_config_returns_the_first_hit_and_none_when_nowhere(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    user = join(tmpdir, 'user')
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', user)
    assert paths.find_config('thing.json') is None

    os.makedirs(user)
    with open(join(user, 'thing.json'), 'w') as f:
        f.write('{"where": "user"}')
    assert paths.find_config('thing.json') == join(user, 'thing.json')

    local = paths.state_dir()
    os.makedirs(local)
    with open(join(local, 'thing.json'), 'w') as f:
        f.write('{"where": "local"}')
    assert paths.find_config('thing.json') == join(local, 'thing.json')


def test_packaged_presets_are_discoverable_and_are_valid_json():
    "The shipped defaults are the last resort of an empty workspace -- they must parse."
    import glob
    import json
    found = glob.glob(join(paths.packaged_defaults_dir(), paths.PRESETS_SUBDIR, '*.json'))
    assert found, 'no presets shipped in qtstamp_defaults/presets'
    for path in found:
        with open(path) as f:
            preset = json.load(f)
        assert isinstance(preset, dict)
        # A shipped preset is merged on top of the user's config, so it must not
        # carry anything that would stomp their paths or session.
        assert not ({'data_path', 'output_path', 'classifications_path', 'session_name',
                     'dock_state'} & set(preset))


def test_ensure_dir_creates_and_reports_true(tmpdir):
    target = join(tmpdir, 'a', 'b')
    assert paths.ensure_dir(target) is True
    assert os.path.isdir(target)


def test_ensure_dir_returns_false_instead_of_raising_on_a_read_only_parent(tmpdir):
    "The local-or-nothing rule: an unwritable workspace must not raise out of here."
    if os.geteuid() == 0:
        return  # root ignores the mode bits, so there is nothing to test
    parent = join(tmpdir, 'ro')
    os.makedirs(parent)
    os.chmod(parent, 0o500)
    try:
        assert paths.ensure_dir(join(parent, 'child')) is False
    finally:
        os.chmod(parent, 0o700)


def test_ensure_dir_returns_false_for_an_existing_unwritable_dir(tmpdir):
    "makedirs(exist_ok=True) succeeds here -- only the access probe catches it."
    if os.geteuid() == 0:
        return
    target = join(tmpdir, 'ro')
    os.makedirs(target)
    os.chmod(target, 0o500)
    try:
        assert paths.ensure_dir(target) is False
    finally:
        os.chmod(target, 0o700)


def test_cache_dir_is_local_and_created(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.chdir(tmpdir)
    expected = join(paths.state_dir(), paths.CACHE_SUBDIR)
    assert paths.cache_dir() == expected
    assert os.path.isdir(expected)


def test_cache_dir_falls_back_to_the_user_cache_dir_when_local_is_unwritable(tmpdir, monkeypatch):
    """The one documented exception to local-or-nothing.

    Settings give up silently; a cache does not, because the mosaic has nowhere
    to render its scratch tiles and the run dies rather than losing a preference.
    """
    if os.geteuid() == 0:
        return
    _clear_state_env(monkeypatch)
    workspace = join(tmpdir, 'ro')
    os.makedirs(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('QTSTAMP_CACHE_DIR', join(tmpdir, 'user-cache'))
    os.chmod(workspace, 0o500)
    try:
        assert paths.cache_dir() == join(paths.absolute(tmpdir), 'user-cache')
    finally:
        os.chmod(workspace, 0o700)


def test_state_dir_does_not_fall_back_when_local_is_unwritable(tmpdir, monkeypatch):
    "Config has no fallback: it still points local, and the write is what fails."
    if os.geteuid() == 0:
        return
    _clear_state_env(monkeypatch)
    workspace = join(tmpdir, 'ro')
    os.makedirs(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', join(tmpdir, 'user-config'))
    os.chmod(workspace, 0o500)
    try:
        assert paths.state_dir() == join(paths.absolute(workspace), paths.LOCAL_STATE_DIRNAME)
    finally:
        os.chmod(workspace, 0o700)


# --- --state-dir / --global argparse wiring --------------------------------

def _parse(argv):
    import argparse
    p = argparse.ArgumentParser()
    paths.add_state_dir_args(p)
    return p.parse_args(argv)


def test_no_flags_means_local(monkeypatch):
    _clear_state_env(monkeypatch)
    assert paths.resolve_state_dir_override(_parse([])) is None


def test_global_resolves_to_the_user_config_dir(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', join(tmpdir, 'user'))
    assert paths.resolve_state_dir_override(_parse(['--global'])) == paths.user_config_dir()


def test_explicit_state_dir_wins_over_global(tmpdir, monkeypatch):
    _clear_state_env(monkeypatch)
    monkeypatch.setenv('QTSTAMP_CONFIG_DIR', join(tmpdir, 'user'))
    args = _parse(['--global', '--state-dir', join(tmpdir, 'explicit')])
    assert paths.resolve_state_dir_override(args) == join(paths.absolute(tmpdir), 'explicit')


# --- scratch_dir: one scratch directory per process ------------------------

def _dead_pid():
    """A PID that is certainly not running: fork a child and reap it.

    Picking a large number and hoping is flaky -- it may well be in use.
    """
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_scratch_dir_is_per_process(tmpdir):
    path = paths.scratch_dir(tmpdir)
    assert os.path.isdir(path)
    assert os.path.basename(path) == str(os.getpid())
    assert os.path.dirname(path) == join(tmpdir, 'temp')


def test_scratch_dir_leaves_a_live_process_alone(tmpdir):
    "The bug: a second mosaic wiped the first one's tiles."
    other = join(tmpdir, 'temp', str(os.getppid()))
    os.makedirs(other)
    with open(join(other, 'tile.png'), 'w') as f:
        f.write('x')
    paths.scratch_dir(tmpdir)
    assert os.path.exists(join(other, 'tile.png'))


def test_scratch_dir_reaps_a_dead_process(tmpdir):
    stale = join(tmpdir, 'temp', str(_dead_pid()))
    os.makedirs(stale)
    with open(join(stale, 'tile.png'), 'w') as f:
        f.write('x')
    paths.scratch_dir(tmpdir)
    assert not os.path.exists(stale)


def test_scratch_dir_sweeps_the_flat_layout(tmpdir):
    "Loose tiles at the root are pre-per-process leftovers, or a migrated `.temp`."
    os.makedirs(join(tmpdir, 'temp'))
    loose = join(tmpdir, 'temp', '1linear0.0100.0.png')
    with open(loose, 'w') as f:
        f.write('x')
    paths.scratch_dir(tmpdir)
    assert not os.path.exists(loose)


class _MiniMonkeyPatch:
    "Just enough of pytest's monkeypatch fixture to run these tests without pytest."

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((setattr, obj, name, getattr(obj, name, None), hasattr(obj, name)))
        setattr(obj, name, value)

    def setenv(self, name, value):
        had = name in os.environ
        old = os.environ.get(name)
        self._undo.append(('env', name, old, had))
        os.environ[name] = value

    def chdir(self, path):
        self._undo.append(('cwd', os.getcwd()))
        os.chdir(path)

    def delenv(self, name, raising=True):
        had = name in os.environ
        old = os.environ.get(name)
        if not had and raising:
            raise KeyError(name)
        self._undo.append(('env', name, old, had))
        os.environ.pop(name, None)

    def undo(self):
        for entry in reversed(self._undo):
            if entry[0] == 'cwd':
                os.chdir(entry[1])
            elif entry[0] == 'env':
                _, name, old, had = entry
                if had:
                    os.environ[name] = old
                else:
                    os.environ.pop(name, None)
            else:
                _, obj, name, old, had = entry
                if had:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)


if __name__ == '__main__':
    import inspect
    import tempfile
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = 0
    for name, func in tests:
        params = inspect.signature(func).parameters
        mp = _MiniMonkeyPatch() if 'monkeypatch' in params else None
        try:
            args = []
            cm = tempfile.TemporaryDirectory() if 'tmpdir' in params else None
            d = cm.__enter__() if cm else None
            try:
                if 'tmpdir' in params:
                    args.append(d)
                if 'monkeypatch' in params:
                    args.append(mp)
                func(*args)
            finally:
                if cm:
                    cm.__exit__(None, None, None)
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            if mp:
                mp.undo()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
