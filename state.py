"""Persisted viewer state: per-session resume position, per-tool preferences.

Both viewers used to keep everything in a single config file per tool
(`.config.json`, `.config_mosaic.json`), guarded by one weak check: the saved
`--name` string. That conflated two kinds of state with different lifetimes:

* **where the user is in a dataset** belongs to the *dataset*, and must not
  survive a change of `--path`, `--name` or `--seed`;
* **how the user likes the tool to look** belongs to the *user*, and must
  survive all of those.

This module keeps them apart. Preferences live in one file per tool, in the
workspace's own state dir (`paths.state_dir()`, `<CWD>/.qtstamp` by default),
read through `paths.find_config` so a per-user or packaged copy can seed a
fresh workspace, and written back only to the state dir. That is local-first on
purpose (PLAN.md 2b-v): two datasets side by side get two sets of preferences,
and copying a workspace takes its settings along. Session state
lives in a single `sessions.json` inside the *classifications directory* --
the anchor every `--classifications-dir` run already resolves and creates --
keyed by a short hash of the identity ``(tool, path, name, seed)``, with
`path` stored **relative to that anchor**, not as a realpath. That is what
lets "copy the tool and its data to another machine, keeping the data at the
same position relative to the classifications directory" resume where it left
off: the key is computed the same way regardless of where the anchor itself
sits on disk. See `paths.py` for the portable-path arithmetic and
`migrate_sessions_store` for carrying a pre-split, CWD-relative store forward.
The identity is stored in plain text beside its hash so the store stays
readable, which is what `list_sessions` and `session_id_from_entry` turn into
the lobby's recent-sessions picker.

An entry holds whatever its writers put there and nothing is exclusive to one
of them: the viewers record a resume position (`save_session`), and the lobby
records the part of its own config that belongs to the dataset rather than to
the user -- the classification scheme and the band setup -- under `config`
(`save_session_config`, BUGS.md item 14). Both go through `_update_session`,
which merges rather than rebuilds, so neither erases the other's half.

Grid shape and file count are deliberately *not* part of the identity: the
position is stored as a **filename**, not an index, so a reshaped mosaic
re-derives its page from where that object now falls, and adding or removing
files no longer invalidates anything. A filename that has since disappeared is
a defined fallback -- start at 0 -- rather than an out-of-range index.

Everything here is a pure function over directory paths: no Qt, no globals.
The imports are `imaging.atomic_write` and `paths` (itself Qt- and
astropy-free), so the session store and the classification CSVs are written --
and keep their permissions -- the same way.
"""

import hashlib
import json
import os
from collections import namedtuple
from datetime import datetime, timezone

import paths
from imaging import atomic_write

STORE_VERSION = 1
SESSIONS_FILENAME = 'sessions.json'
MAX_SESSIONS = 50

PREFERENCES_TEMPLATE = 'preferences_{tool}.json'
LEGACY_PREFERENCES_TEMPLATE = '.preferences_{tool}.json'

# Preferences are local-first (PLAN.md 2b-v, reversing 2b-iii): they are read
# from the workspace's own state dir, then the per-user directory, then the
# defaults shipped with the tool -- and written *only* to the state dir.
# `config_dir=None` means "the state dir for the current working directory";
# every public function here takes it so the viewers can pass `--state-dir`
# straight through. Session state is anchored to the classifications directory
# instead (see the module docstring); there is no default for it, on purpose.


def _config_dir(config_dir=None):
    "The single directory preferences are written to."
    return paths.state_dir(override=config_dir)


class SessionId(namedtuple('SessionId', 'tool path name seed')):
    """What makes two runs "the same classifying session".

    `seed` stays in here on purpose: a new seed is a new pass in a new order,
    and resuming on the same object would strand the user mid-deck with
    unclassified stamps on both sides.
    """

    __slots__ = ()


def sessions_path(classifications_dir):
    "Path of the shared session store, inside the classifications directory."
    return os.path.join(paths.absolute(classifications_dir), SESSIONS_FILENAME)


def preferences_path(tool, config_dir=None):
    "Where this tool's preferences are *written* -- always the state dir."
    return os.path.join(_config_dir(config_dir), PREFERENCES_TEMPLATE.format(tool=tool))


def find_preferences(tool, config_dir=None):
    """Where this tool's preferences are *read* from, or None if nowhere has them.

    Searches state dir -> user config dir -> packaged defaults, so an existing
    per-user file written before 2b-v still seeds a fresh workspace. It seeds
    only: the next `save_preferences` lands in the state dir, and the file that
    was read stays untouched, because it may be seeding other workspaces too.
    """
    return paths.find_config(PREFERENCES_TEMPLATE.format(tool=tool), override=config_dir,
                             legacy=LEGACY_PREFERENCES_TEMPLATE.format(tool=tool))


def session_identity(session_id, classifications_dir):
    "The identity payload stored in plain text next to its hash."
    value, form = paths.relative_to(session_id.path, classifications_dir)
    return {
        'tool': session_id.tool,
        'path': value,
        'path_form': form,          # 'relative' | 'absolute'
        'name': session_id.name or '',
        'seed': None if session_id.seed is None else int(session_id.seed),
    }


def _legacy_session_identity(session_id):
    "The pre-2b-i identity shape, reproduced verbatim for the legacy-key fallback."
    return {
        'tool': session_id.tool,
        'path': os.path.realpath(os.path.expanduser(session_id.path)),
        'name': session_id.name or '',
        'seed': None if session_id.seed is None else int(session_id.seed),
    }


def session_key(session_id, classifications_dir):
    "Short, stable hash of the identity; the key into the session store."
    blob = json.dumps(session_identity(session_id, classifications_dir),
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def _legacy_session_key(session_id):
    "The key a pre-2b-i store would have computed for this identity."
    blob = json.dumps(_legacy_session_identity(session_id), sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _empty_store():
    """A store with every key the rest of the module expects to find.

    `legacy_imports` is in here rather than conjured by `setdefault` at its one
    use site so the shape is the same whichever branch of `_load_store`
    produced it -- see BUGS.md item 8, which was that record going missing.
    """
    return {'version': STORE_VERSION, 'sessions': {}, 'legacy_imports': []}


def _read_json(path):
    "Parsed JSON object at `path`, or None if it is missing or unusable."
    try:
        with open(path) as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_json(path, payload):
    """Write `payload` to `path` atomically, preserving its permissions.

    The two viewers wrote to *separate* files before, so consolidating them
    into one store introduces a clobber risk that did not exist: a temp file in
    the same directory plus os.replace means a reader never sees a half-written
    store, and a crash mid-write leaves the previous one intact.

    The classifications directory is not guaranteed to exist yet at every entry
    point (notably the module-level `--reset-position` path, which can run
    before the tool's own `os.makedirs(self.classifications_dir, ...)`), so
    make sure it does before handing off to `atomic_write`.

    Best-effort since 2b-v: returns True if the write happened, False if the
    target directory is unusable. Local-first state means the tool now writes
    into whatever directory the user launched it from, which is not guaranteed
    to be writable -- and the rule the user set is "write locally or not at
    all". Losing persistence is the accepted cost; taking the run down with it
    (BUGS.md item 1's `PermissionError`) is not.
    """
    if not paths.ensure_dir(os.path.dirname(path) or os.curdir):
        return False
    try:
        atomic_write(path, lambda f: json.dump(payload, f, ensure_ascii=False, indent=4),
                     suffix='.json')
    except OSError:
        return False
    return True


def _load_store(classifications_dir):
    """`(store, intact)` -- the store, plus whether it really came off disk.

    `intact` is False only when a store file *exists* and could not be read as
    one. That is not the same as a missing store: the empty store handed back
    in its place has silently replaced state nobody can see any more, including
    the `legacy_imports` record that makes `migrate_sessions_store` once-only
    (BUGS.md item 8). Callers that just read sessions do not care; the
    migration does.
    """
    path = sessions_path(classifications_dir)
    store = _read_json(path)
    if store is None or not isinstance(store.get('sessions'), dict):
        return _empty_store(), not os.path.exists(path)
    store['version'] = STORE_VERSION
    if not isinstance(store.get('legacy_imports'), list):
        store['legacy_imports'] = []
    return store, True


def load_store(classifications_dir):
    "The whole session store; an empty one if it is missing or corrupt."
    return _load_store(classifications_dir)[0]


def prune(store, max_sessions=MAX_SESSIONS):
    "Drop the least recently opened sessions, in place, so the store stays bounded."
    sessions = store.setdefault('sessions', {})
    excess = len(sessions) - max_sessions
    if excess <= 0:
        return store
    # Entries with no timestamp sort first, i.e. they are dropped first.
    oldest = sorted(sessions.items(), key=lambda kv: (kv[1].get('last_opened') or '', kv[0]))
    for key, _ in oldest[:excess]:
        del sessions[key]
    return store


def load_session(session_id, classifications_dir):
    """This identity's saved session entry, or None if it has never been saved.

    Tries the current (anchor-relative) key first, then the legacy
    absolute-realpath key -- so a store that has not been migrated yet still
    resumes on the machine and directory it was written on.
    """
    sessions = load_store(classifications_dir)['sessions']
    entry = sessions.get(session_key(session_id, classifications_dir))
    if not isinstance(entry, dict):
        entry = sessions.get(_legacy_session_key(session_id))
    return entry if isinstance(entry, dict) else None


def _update_session(session_id, classifications_dir, fields, max_sessions=MAX_SESSIONS):
    """Merge `fields` into this identity's entry, creating it if it is new.

    Read-modify-write over the whole store, and over the *entry* too: a writer
    that only knows about some of an entry's keys must not drop the rest. Two
    different writers share one entry now -- the viewers record a position, the
    lobby records its session-scoped config (BUGS.md item 14) -- and before this
    existed, `save_session` rebuilt the entry from scratch, so whichever of them
    wrote last erased the other's half. A legacy-keyed entry is picked up the
    same way `load_session` does and rewritten under the current key -- and the
    legacy copy is then dropped, since everything it held has just been carried
    into the new entry and two keys for one identity would otherwise both sit
    there competing for a slot under `prune`. That is the same "both keys are
    one identity" rule `forget_session` already applies; before this, a
    pre-2b-i entry survived every save as a stale duplicate.
    """
    store = load_store(classifications_dir)  # both viewers and the lobby share this file
    key = session_key(session_id, classifications_dir)
    existing = store['sessions'].get(key)
    if not isinstance(existing, dict):
        legacy_key = _legacy_session_key(session_id)
        existing = store['sessions'].pop(legacy_key, None)
    entry = dict(existing) if isinstance(existing, dict) else {}
    entry.update(session_identity(session_id, classifications_dir))
    entry.update(fields)
    entry['last_opened'] = _now()
    store['sessions'][key] = entry
    prune(store, max_sessions)
    _write_json(sessions_path(classifications_dir), store)
    return entry


def save_session(session_id, classifications_dir, position_filename=None, csv=None,
                 max_sessions=MAX_SESSIONS):
    "Record where this session is, and which CSV that position belongs to."
    return _update_session(session_id, classifications_dir, {
        'position_filename': position_filename,
        'csv': paths.relative_to(csv, classifications_dir)[0] if csv else None,
    }, max_sessions)


def save_session_config(session_id, classifications_dir, config, max_sessions=MAX_SESSIONS):
    """Record this session's own copy of its tool config, under `config`.

    The lobby's use (BUGS.md item 14): `scheme_rows` and the band setup describe
    the *dataset*, not the user, so they have to travel with the session rather
    than sit in the one workspace-wide `config_lobby.json` that every session
    shares. Stored as an opaque dict -- `state.py` stays Qt-free and knows
    nothing about what a scheme row is.
    """
    return _update_session(session_id, classifications_dir,
                           {'config': dict(config)}, max_sessions)


def load_session_config(session_id, classifications_dir):
    "This session's saved tool config, or None if it has never saved one."
    entry = load_session(session_id, classifications_dir) or {}
    config = entry.get('config')
    return dict(config) if isinstance(config, dict) else None


def list_sessions(classifications_dir, tool=None):
    """Saved sessions, most recently opened first -- the recent-sessions picker.

    Each entry is a copy carrying its store key under `key`, so a caller can
    show the list and then act on one without recomputing the hash. Entries with
    no timestamp sort last: they predate `last_opened`, so "recent" is exactly
    what is not known about them. `tool` filters to one tool's records.
    """
    entries = []
    for key, entry in load_store(classifications_dir)['sessions'].items():
        if not isinstance(entry, dict):
            continue
        if tool is not None and entry.get('tool') != tool:
            continue
        entries.append(dict(entry, key=key))
    # '' sorts before any timestamp, so reversing puts the unstamped ones last.
    entries.sort(key=lambda e: (e.get('last_opened') or '', e.get('key') or ''), reverse=True)
    return entries


def session_id_from_entry(entry, classifications_dir):
    """The `SessionId` a stored entry came from, or None if it is unusable.

    The inverse of `session_identity`: `path` is stored relative to the anchor
    (see the module docstring), so it has to be resolved back against it before
    it can be handed to `load_session`/`forget_session`, which re-derive the
    relative form themselves. A legacy entry has no `path_form` and its `path`
    is already absolute, so `resolve_against` passes it through unchanged.
    """
    if not isinstance(entry, dict):
        return None
    tool, path = entry.get('tool'), entry.get('path')
    if not tool or not path:
        return None
    seed = entry.get('seed')
    return SessionId(tool, paths.resolve_against(path, classifications_dir),
                     entry.get('name') or '', None if seed is None else int(seed))


def forget_session(session_id, classifications_dir):
    """Drop this identity's saved position -- current key and legacy key alike.

    True if either was present. Dropping only the current key would leave a
    legacy-keyed entry behind for `load_session` to resurrect, so
    `--reset-position` would appear to work and then the position would come
    back on the next launch.
    """
    store = load_store(classifications_dir)
    current_hit = store['sessions'].pop(session_key(session_id, classifications_dir), None) is not None
    legacy_hit = store['sessions'].pop(_legacy_session_key(session_id), None) is not None
    if not (current_hit or legacy_hit):
        return False
    _write_json(sessions_path(classifications_dir), store)
    return True


def resolve_position(entry, listimage, csv=None, classifications_dir=None):
    """Index into `listimage` to resume at -- 0 whenever the entry can't be trusted.

    Every way of being untrustworthy lands on the same defined answer instead
    of an out-of-range index: no entry, no saved filename, a file that is gone,
    or a position recorded against a different classification CSV (stale by
    definition -- `obtain_df` resolved somewhere else, so the two disagree
    about which session is open).

    The recorded `csv` may be relative (to `classifications_dir`) or, for a
    legacy entry, absolute. Without a `classifications_dir` (pure unit tests,
    and only there) a relative recorded value has nothing to resolve against
    and is treated as a mismatch rather than guessed against the CWD; an
    absolute recorded value still compares fine either way.
    """
    if not entry:
        return 0
    recorded_csv = entry.get('csv')
    if csv and recorded_csv:
        if classifications_dir is not None:
            if paths.resolve_against(recorded_csv, classifications_dir) != paths.absolute(csv):
                return 0
        elif not os.path.isabs(recorded_csv) or paths.absolute(recorded_csv) != paths.absolute(csv):
            return 0
    filename = entry.get('position_filename')
    if not filename:
        return 0
    try:
        return list(listimage).index(filename)
    except ValueError:
        return 0


def load_preferences(tool, defaults, config_dir=None):
    "This tool's preferences, filled in from `defaults`. Unknown keys are dropped."
    preferences = dict(defaults)
    found = find_preferences(tool, config_dir)
    saved = _read_json(found) if found else None
    if saved:
        preferences.update({k: v for k, v in saved.items() if k in defaults})
    return preferences


def save_preferences(tool, preferences, config_dir=None):
    """Persist this tool's preferences in the state dir. True if it was written.

    False means the state dir is not writable and the user has lost settings
    persistence for this workspace -- deliberately not a fallback write to the
    per-user directory, which is a place they did not point at.
    """
    return _write_json(preferences_path(tool, config_dir), dict(preferences))


def reset_preferences(tool, config_dir=None):
    """Delete this tool's preferences file. True if there was one to delete.

    Only the state dir's copy: a per-user or packaged file further along the
    search path is not ours to remove, and after this the tool falls back to it
    exactly as a fresh workspace would.
    """
    path = preferences_path(tool, config_dir)
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def migrate_preferences_file(tool, legacy_dir=None, config_dir=None):
    """Move a pre-2b-v preferences file into the state dir, once.

    Two legacy spellings, both local, both from before the state dir existed:
    `.preferences_{tool}.json` sitting loose in `legacy_dir` (the CWD, where
    preferences lived before 2b-iii), and the same name inside the state dir
    itself (2b-v dropped the leading dot -- redundant inside an already-hidden
    directory, and setuptools skips dotfiles, which the packaged defaults need).

    Deliberately does *not* migrate out of `paths.user_config_dir()`. That copy
    is the fallback other workspaces read; moving it here would strand them.
    Thin wrapper around `paths.migrate_once`, whose no-op-when-legacy-is-target
    guard covers the case where the state dir already is the legacy dir.
    """
    target = preferences_path(tool, config_dir)
    legacy_name = LEGACY_PREFERENCES_TEMPLATE.format(tool=tool)
    moved = paths.migrate_once(
        os.path.join(paths.absolute(legacy_dir or os.curdir), legacy_name), target)
    return paths.migrate_once(
        os.path.join(os.path.dirname(target), legacy_name), target) or moved


def migrate_sessions_store(classifications_dir, legacy_dir=None):
    """Import a pre-2b-i, CWD-relative `sessions.json` into the anchor's store.

    `legacy_dir` defaults to the current working directory -- where the store
    used to live before session state moved to the classifications directory
    (2b-i). It is the bare CWD, *not* `paths.state_dir()`: the pre-2b-i
    `sessions.json` was a loose file in the CWD and predates `.qtstamp/`
    entirely, so pointing this at the state dir would look right and find
    nothing. The two have been near-misses of each other through three
    reshuffles now (2b-iii made the preferences default the per-user dir, 2b-v
    made it the state dir, and this one never moved); they are a different
    thing that keeps coinciding, not one default doing double duty. Tying them
    together silently breaks this migration's real call sites (mosaic.py and
    single_viewer.py, which call it with no `legacy_dir` and mean the CWD) --
    see the regression test in tests/test_state.py.

    Copies rather than moves or deletes the legacy file. The reasoning is
    load-bearing, not cosmetic: the CWD store is a single *global* file that
    can hold entries for several different `--classifications-dir` runs, so
    renaming or deleting it during the first anchor's migration would strand
    every other anchor's position -- the exact failure this step exists to
    prevent. Retiring the legacy file belongs to a later step that moves the
    rest of the CWD state anyway (PLAN.md 2b-iii/iv).

    Runs at most once per (legacy file, anchor) pair: the import is recorded
    in `store['legacy_imports']`, so a deliberate `--reset-position` is not
    undone by the next startup resurrecting the old entry. Never overwrites a
    key that already exists in the target store.

    An unreadable target store is treated as "already imported" rather than as
    a fresh one (BUGS.md item 8). The record that makes this once-only lives
    *in* the store, so a corrupt store is exactly the case where we cannot tell
    whether the import already ran -- and the two ways of being wrong are not
    symmetric: skipping an import the user never had costs them nothing they
    can see, while repeating one resurrects a position they deliberately
    cleared. The mark is written straight back out so the next startup does not
    ask the same unanswerable question, and the unreadable file is kept as
    `sessions.json.corrupt` first, since this is the point where it stops being
    recoverable.
    """
    legacy_dir = os.curdir if legacy_dir is None else legacy_dir
    legacy_file = os.path.join(paths.absolute(legacy_dir), SESSIONS_FILENAME)
    target = sessions_path(classifications_dir)
    if paths.absolute(legacy_file) == paths.absolute(target):
        return False  # the classifications dir *is* the CWD -- must be a no-op

    legacy_store = _read_json(legacy_file)
    if legacy_store is None or not isinstance(legacy_store.get('sessions'), dict):
        return False

    store, intact = _load_store(classifications_dir)
    imports = store['legacy_imports']
    legacy_key = paths.absolute(legacy_file)
    if legacy_key in imports:
        return False  # already imported once -- never resurrect a forgotten entry
    if not intact:
        try:
            os.replace(target, target + '.corrupt')
        except OSError:
            pass
        store['legacy_imports'] = [legacy_key]
        _write_json(target, store)
        return False

    for legacy_entry in legacy_store['sessions'].values():
        if not isinstance(legacy_entry, dict):
            continue
        tool = legacy_entry.get('tool')
        path = legacy_entry.get('path')
        if not tool or not path:
            continue
        sid = SessionId(tool, path, legacy_entry.get('name', ''), legacy_entry.get('seed'))
        new_key = session_key(sid, classifications_dir)
        if new_key in store['sessions']:
            continue  # never clobber a newer entry
        new_entry = dict(legacy_entry)
        new_entry.update(session_identity(sid, classifications_dir))
        old_csv = legacy_entry.get('csv')
        new_entry['csv'] = paths.relative_to(old_csv, classifications_dir)[0] if old_csv else None
        store['sessions'][new_key] = new_entry

    imports.append(legacy_key)
    store['legacy_imports'] = sorted(set(imports))
    prune(store)
    os.makedirs(os.path.dirname(target) or os.curdir, exist_ok=True)
    _write_json(target, store)
    return True


def migrate_legacy_config(legacy_path, session_id, defaults, listimage, legacy_index,
                          classifications_dir, csv=None, config_dir=None):
    """Split one old single-blob config into preferences plus one session entry.

    `legacy_index` is called with the old dict and returns the index into
    `listimage` it was pointing at, or None to drop the position (the old file
    only ever guarded it with a `--name` comparison, so that is all the caller
    can honestly check).

    `classifications_dir` is required. It used to default to None, which had no
    working meaning -- session state has no default anchor (see the note above
    `_config_dir`), so it reached `sessions_path(None)` and raised `TypeError`
    on the first call that took it up on the offer. Both real call sites always
    passed it by keyword, so this was a dead default rather than a live crash;
    BUGS.md item 9.

    The original is renamed to `<legacy_path>.bak` rather than deleted, and
    nothing happens at all once this tool has a preferences file -- migration
    runs exactly once, and never overwrites newer state. A freshly imported
    position (via `migrate_sessions_store`, which runs first at startup) is
    likewise never overwritten: the `save_session` call below is skipped if
    this identity already has an entry.
    """
    tool = session_id.tool
    if os.path.exists(preferences_path(tool, config_dir)):
        return False
    legacy = _read_json(legacy_path)
    if legacy is None:
        return False

    save_preferences(tool, {k: v for k, v in legacy.items() if k in defaults}, config_dir)

    index = legacy_index(legacy)
    if index is not None and 0 <= index < len(listimage):
        if load_session(session_id, classifications_dir) is None:
            save_session(session_id, classifications_dir,
                        position_filename=listimage[index], csv=csv)

    os.replace(legacy_path, legacy_path + '.bak')
    return True
