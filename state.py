"""Persisted viewer state: per-session resume position, per-tool preferences.

Both viewers used to keep everything in a single config file per tool
(`.config.json`, `.config_mosaic.json`), guarded by one weak check: the saved
`--name` string. That conflated two kinds of state with different lifetimes:

* **where the user is in a dataset** belongs to the *dataset*, and must not
  survive a change of `--path`, `--name` or `--seed`;
* **how the user likes the tool to look** belongs to the *user*, and must
  survive all of those.

This module keeps them apart. Preferences live in one file per tool; session
state lives in a single `sessions.json` keyed by a short hash of the identity
``(tool, realpath(path), name, seed)``, with the identity stored in plain text
beside it so the store stays readable (and so a recent-sessions picker is
nearly free to build on top of it).

Grid shape and file count are deliberately *not* part of the identity: the
position is stored as a **filename**, not an index, so a reshaped mosaic
re-derives its page from where that object now falls, and adding or removing
files no longer invalidates anything. A filename that has since disappeared is
a defined fallback -- start at 0 -- rather than an out-of-range index.

Everything here is a pure function over a directory path: no Qt, no globals.
The one import is `imaging.atomic_write`, so that the session store and the
classification CSVs are written -- and keep their permissions -- the same way.
"""

import hashlib
import json
import os
from collections import namedtuple
from datetime import datetime, timezone

from imaging import atomic_write

STORE_VERSION = 1
SESSIONS_FILENAME = 'sessions.json'
MAX_SESSIONS = 50

# Today's location: the config files were CWD-relative. Item 2b of PLAN.md moves
# every config/cache directory at once, and then only this default changes.
DEFAULT_CONFIG_DIR = os.curdir


class SessionId(namedtuple('SessionId', 'tool path name seed')):
    """What makes two runs "the same classifying session".

    `seed` stays in here on purpose: a new seed is a new pass in a new order,
    and resuming on the same object would strand the user mid-deck with
    unclassified stamps on both sides.
    """

    __slots__ = ()


def _config_dir(config_dir=None):
    return os.path.abspath(os.path.expanduser(config_dir or DEFAULT_CONFIG_DIR))


def sessions_path(config_dir=None):
    "Path of the shared session store."
    return os.path.join(_config_dir(config_dir), SESSIONS_FILENAME)


def preferences_path(tool, config_dir=None):
    "Path of one tool's preferences file."
    return os.path.join(_config_dir(config_dir), f'.preferences_{tool}.json')


def session_identity(session_id):
    "The identity payload stored in plain text next to its hash."
    return {
        'tool': session_id.tool,
        'path': os.path.realpath(os.path.expanduser(session_id.path)),
        'name': session_id.name or '',
        'seed': None if session_id.seed is None else int(session_id.seed),
    }


def session_key(session_id):
    "Short, stable hash of the identity; the key into the session store."
    blob = json.dumps(session_identity(session_id), sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _empty_store():
    return {'version': STORE_VERSION, 'sessions': {}}


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
    """
    atomic_write(path, lambda f: json.dump(payload, f, ensure_ascii=False, indent=4),
                 suffix='.json')


def load_store(config_dir=None):
    "The whole session store; an empty one if it is missing or corrupt."
    store = _read_json(sessions_path(config_dir))
    if store is None or not isinstance(store.get('sessions'), dict):
        return _empty_store()
    store['version'] = STORE_VERSION
    return store


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


def load_session(session_id, config_dir=None):
    "This identity's saved session entry, or None if it has never been saved."
    entry = load_store(config_dir)['sessions'].get(session_key(session_id))
    return entry if isinstance(entry, dict) else None


def save_session(session_id, position_filename=None, csv=None, config_dir=None,
                 max_sessions=MAX_SESSIONS):
    "Record where this session is, and which CSV that position belongs to."
    store = load_store(config_dir)  # read-modify-write: both viewers share this file
    entry = session_identity(session_id)
    entry['position_filename'] = position_filename
    entry['csv'] = os.path.abspath(csv) if csv else None
    entry['last_opened'] = _now()
    store['sessions'][session_key(session_id)] = entry
    prune(store, max_sessions)
    _write_json(sessions_path(config_dir), store)
    return entry


def forget_session(session_id, config_dir=None):
    "Drop this identity's saved position. True if there was one to drop."
    store = load_store(config_dir)
    if store['sessions'].pop(session_key(session_id), None) is None:
        return False
    _write_json(sessions_path(config_dir), store)
    return True


def resolve_position(entry, listimage, csv=None):
    """Index into `listimage` to resume at -- 0 whenever the entry can't be trusted.

    Every way of being untrustworthy lands on the same defined answer instead
    of an out-of-range index: no entry, no saved filename, a file that is gone,
    or a position recorded against a different classification CSV (stale by
    definition -- `obtain_df` resolved somewhere else, so the two disagree
    about which session is open).
    """
    if not entry:
        return 0
    recorded_csv = entry.get('csv')
    if csv and recorded_csv and os.path.abspath(csv) != recorded_csv:
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
    saved = _read_json(preferences_path(tool, config_dir))
    if saved:
        preferences.update({k: v for k, v in saved.items() if k in defaults})
    return preferences


def save_preferences(tool, preferences, config_dir=None):
    _write_json(preferences_path(tool, config_dir), dict(preferences))


def reset_preferences(tool, config_dir=None):
    "Delete this tool's preferences file. True if there was one to delete."
    path = preferences_path(tool, config_dir)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def migrate_legacy_config(legacy_path, session_id, defaults, listimage,
                          legacy_index, csv=None, config_dir=None):
    """Split one old single-blob config into preferences plus one session entry.

    `legacy_index` is called with the old dict and returns the index into
    `listimage` it was pointing at, or None to drop the position (the old file
    only ever guarded it with a `--name` comparison, so that is all the caller
    can honestly check).

    The original is renamed to `<legacy_path>.bak` rather than deleted, and
    nothing happens at all once this tool has a preferences file -- migration
    runs exactly once, and never overwrites newer state.
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
        save_session(session_id, position_filename=listimage[index], csv=csv,
                     config_dir=config_dir)

    os.replace(legacy_path, legacy_path + '.bak')
    return True
