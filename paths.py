# This Python file uses the following encoding: utf-8
"""Portable path arithmetic for the classifications directory and session state.

PLAN.md item 2b makes the classifications directory (`--classifications-dir`)
the anchor session state is stored under and keyed against, so that copying
the tool and its data to another machine or another directory -- as long as
the data keeps the same position relative to the classifications directory --
lets the tool resume where it left off. Everything in this module is pure
`os.path` arithmetic: no Qt, no astropy, no import from any other module in
this repo, so that `state.py` can import it without creating a cycle
(`state.py` already imports `imaging.atomic_write`).
"""

import atexit
import os
import shutil
import sys

DEFAULT_CLASSIFICATIONS_DIRNAME = 'Classifications'
RELATIVE = 'relative'
ABSOLUTE = 'absolute'


def resolve_classifications_dir(path=None):
    """Absolute path of the directory the classification CSVs live in.

    `path` may be absolute or relative (resolved against the current working
    directory) and may start with `~`; None or '' means the default,
    ./Classifications. Callers that are about to write create it themselves --
    this stays a pure function so that merely resolving a path (e.g. for
    lobby's --print-command) never touches the disk.
    """
    return os.path.abspath(os.path.expanduser(path or DEFAULT_CLASSIFICATIONS_DIRNAME))


def absolute(path):
    "realpath(expanduser(path)) -- the one normalisation both ends of a relpath use."
    return os.path.realpath(os.path.expanduser(path))


def to_portable(path):
    "Native separators -> '/', so a key written on Windows equals one written on POSIX."
    path = path.replace(os.sep, '/')
    if os.altsep:
        path = path.replace(os.altsep, '/')
    return path


def from_portable(path):
    "Inverse of to_portable: '/' -> the running platform's separator."
    if os.sep != '/':
        return path.replace('/', os.sep)
    return path


def relative_to(path, anchor):
    """`path` expressed relative to `anchor` -- returns `(value, form)`.

    Both ends are normalised through `absolute` (realpath, not abspath) before
    the relationship is computed: mixing the two is the bug that breaks the
    round trip on macOS (`/tmp` -> `/private/tmp`) and anywhere the data or
    the classifications dir is reached through a symlink. `..` segments in the
    result are expected and correct -- the data directory is not required to
    live inside the classifications directory. Falls back to an absolute,
    portable path only where `relpath` cannot express the relationship at all
    (different drives on Windows; POSIX never raises here).
    """
    target = absolute(path)
    try:
        rel = os.path.relpath(target, absolute(anchor))
    except ValueError:
        return to_portable(target), ABSOLUTE
    return to_portable(rel), RELATIVE


def resolve_against(value, anchor):
    "Inverse of relative_to: turn a stored (portable) value back into an absolute path."
    native = from_portable(value)
    if os.path.isabs(native):
        return absolute(native)
    return absolute(os.path.join(absolute(anchor), native))


def _env_override(var):
    v = os.environ.get(var)
    return absolute(v) if v else None


# --- the state directory ----------------------------------------------------
#
# Step 2b-v reverses part of 2b-iii. Config is *local first*: everything a
# workspace persists lives in one directory, `<CWD>/.qtstamp`, so that copying
# or deleting a workspace takes its settings with it and two datasets side by
# side don't share one global blob. The per-user directories below are still
# here, but demoted to a read-only fallback: they are consulted when the local
# state dir has no answer, and are never written to unless the user explicitly
# asks with `--state-dir`/`--global`.
#
# The anchor is the CWD, deliberately *not* `--classifications-dir`. That flag
# anchors dataset state (`sessions.json`, the CSVs) and 2b-i's copy-the-tree
# guarantee depends on it meaning exactly that; overloading it with "and also
# where your colormap preference lives" would make the config location move
# silently the first time someone passes the flag.

LOCAL_STATE_DIRNAME = '.qtstamp'
CACHE_SUBDIR = 'cache'
PRESETS_SUBDIR = 'presets'
STATE_DIR_ENV = 'QTSTAMP_STATE_DIR'


def packaged_defaults_dir():
    """Read-only defaults shipped with the tool (`qtstamp_defaults/`).

    Last entry on the config search path, so a brand-new workspace starts from
    something reasonable instead of from nothing. Nothing ever writes here --
    under `pip install` it is inside site-packages.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qtstamp_defaults')


def state_dir(anchor=None, override=None):
    """The one directory this workspace's state is written to.

    `override` is `--state-dir` (or `--global`, which passes `user_config_dir()`)
    -- the explicit opt-in that is the *only* way anything lands outside the
    workspace. `QTSTAMP_STATE_DIR` says the same thing through the environment,
    which is how the lobby tells the viewers it launches to agree with it.
    Otherwise it is `<anchor>/.qtstamp`, with `anchor` defaulting to the
    directory the program was launched from.

    Pure arithmetic: it does not create anything. Callers about to write go
    through `ensure_dir` and accept that it can say no.
    """
    if override:
        return absolute(override)
    from_env = _env_override(STATE_DIR_ENV)
    if from_env:
        return from_env
    return os.path.join(absolute(anchor or os.curdir), LOCAL_STATE_DIRNAME)


def config_search_path(anchor=None, override=None):
    """Directories to look for a config file in, most specific first.

    state dir -> per-user config dir -> packaged defaults. Deduplicated, since
    `--global` (and `QTSTAMP_CONFIG_DIR` in tests) can make the first two the
    same directory. Only the first entry is ever written to.
    """
    ordered = [state_dir(anchor, override), user_config_dir(), packaged_defaults_dir()]
    seen, out = set(), []
    for d in ordered:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_config(relname, anchor=None, override=None, legacy=None):
    """Path of the first existing `relname` along the search path, or None.

    `legacy` is an older spelling of the same file, tried *within each
    directory* right after `relname` -- not after the whole search path. That
    ordering matters: a workspace's own legacy-named file has to beat a
    modern-named one in the per-user fallback, or a user who has both would
    silently start reading someone else's settings. It exists because 2b-iii
    wrote `.preferences_mosaic.json`/`.config_lobby.json` into the per-user
    directory and 2b-v dropped the leading dot; those files are never migrated
    out of there (other workspaces still read them), so they have to stay
    findable under their old names indefinitely.
    """
    for d in config_search_path(anchor, override):
        for name in (relname, legacy):
            if not name:
                continue
            candidate = os.path.join(d, name)
            if os.path.exists(candidate):
                return candidate
    return None


def ensure_dir(path):
    """Create `path` and report whether it is actually usable. Never raises.

    The local-first rule is "write locally or not at all": a workspace on
    read-only media loses settings persistence, it does not lose the run. Every
    write site asks this first and degrades quietly when the answer is False,
    rather than crashing (the class of bug BUGS.md item 1 was) or silently
    writing somewhere the user never pointed at.

    `os.access` after `makedirs` rather than `makedirs` alone: an existing
    directory with no write bit raises nothing until the first write inside it.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK | os.X_OK)


def cache_dir(anchor=None, override=None):
    """Directory for re-derivable data: survey cutouts and scratch renders.

    Local like everything else -- `<state dir>/cache` -- but with one
    deliberate exception to "local or nothing": if the local state dir is not
    usable, this falls back to `user_cache_dir()` instead of giving up. A lost
    preference costs the user a preference; a lost scratch directory costs them
    a mosaic that cannot render at all. Caches are re-downloadable and carry no
    user intent, so putting them somewhere else is a recoverable surprise in a
    way that silently rewriting settings would not be.
    """
    if override is None:
        env = _env_override('QTSTAMP_CACHE_DIR')
        if env:
            return env
    local = os.path.join(state_dir(anchor, override), CACHE_SUBDIR)
    if ensure_dir(local):
        return local
    return user_cache_dir()


def _pid_alive(pid):
    """Whether `pid` is a live process. Conservatively True when unsure.

    Never on Windows, where `os.kill(pid, 0)` does not probe -- it calls
    TerminateProcess and would kill the very process we are asking about.
    Reaping is skipped entirely there; the cost is a leaked scratch directory
    after a crash, not a killed editor.
    """
    if sys.platform == 'win32':
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True         # PermissionError: alive, just someone else's
    return True


def scratch_dir(cache_root):
    """A per-process scratch directory under `<cache_root>/temp`.

    Two mosaics started in one workspace used to share `cache/temp`: tile
    filenames are index-based and each mosaic wipes the directory at startup
    and on every page turn, so the second one pulled the first one's renders
    out from under it. Each process gets its own subdirectory instead, named
    for its PID and removed on exit.

    Anything under the root that is not a live process's directory is swept on
    the way in: a crashed run's leftovers (where `atexit` never ran), or loose
    tiles from the flat layout this replaced. Nothing accumulates. A PID that
    has been reused looks alive and is left alone, so the worst case is one
    stale directory surviving one extra run.
    """
    root = os.path.join(cache_root, 'temp')
    os.makedirs(root, exist_ok=True)
    for name in os.listdir(root):
        entry = os.path.join(root, name)
        if name.isdigit() and os.path.isdir(entry) and _pid_alive(int(name)):
            continue
        if os.path.isdir(entry):
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                os.remove(entry)
            except OSError:
                pass
    path = os.path.join(root, str(os.getpid()))
    os.makedirs(path, exist_ok=True)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def add_state_dir_args(parser):
    """Register `--state-dir`/`--global` on an argparse parser.

    The same two flags on all three tools, defined once so they cannot drift
    apart: the lobby passes its own choice down to the viewer it launches, and
    a viewer started by hand has to understand the identical spelling.
    """
    group = parser.add_argument_group("state location")
    group.add_argument("--state-dir", metavar="PATH", default=None,
                       help="Directory holding this workspace's settings, presets and caches "
                            f"(default: ./{LOCAL_STATE_DIRNAME}, i.e. local to wherever you "
                            "started the tool).")
    group.add_argument("--global", dest="global_state", action="store_true",
                       help="Shorthand for --state-dir pointing at the per-user config "
                            f"directory ({user_config_dir()}). Settings are then shared by "
                            "every workspace instead of staying local.")
    return group


def resolve_state_dir_override(args):
    """`--state-dir`/`--global` reduced to one value: a path, or None for local.

    Explicit `--state-dir` wins over `--global` rather than conflicting with
    it, so `--global --state-dir X` means X -- the more specific flag is the
    one the user typed last in spirit.
    """
    override = getattr(args, 'state_dir', None)
    if override:
        return absolute(override)
    if getattr(args, 'global_state', False):
        return user_config_dir()
    return None


def user_config_dir():
    """Per-user, per-machine config directory -- the *fallback* since 2b-v, read
    when the local state dir has no answer and written only via `--global`.
    QTSTAMP_CONFIG_DIR overrides it (used by tests to avoid touching the real
    user profile)."""
    override = _env_override('QTSTAMP_CONFIG_DIR')
    if override:
        return override
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    elif os.name == 'nt':
        base = os.environ.get('APPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    return os.path.join(base, 'qtstamp')


def user_cache_dir():
    "Same shape as user_config_dir, QTSTAMP_CACHE_DIR override, Caches/.cache branch."
    override = _env_override('QTSTAMP_CACHE_DIR')
    if override:
        return override
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Caches')
    elif os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'qtstamp')


def migrate_once(legacy_path, target_path):
    """Move a file or directory from a pre-2b-iii location to its new one, at most once.

    No-op if `legacy_path` doesn't exist, if `target_path` already exists (never clobber
    newer state), or if the two resolve to the same place. shutil.move, not os.replace --
    a directory move must survive crossing filesystems (repo checkout on a different mount
    than $HOME). Returns True iff it actually moved something, so callers can log it.
    """
    if not os.path.exists(legacy_path):
        return False
    if absolute(legacy_path) == absolute(target_path):
        return False
    if os.path.exists(target_path):
        return False
    try:
        os.makedirs(os.path.dirname(target_path) or os.curdir, exist_ok=True)
        shutil.move(legacy_path, target_path)
    except OSError:
        # Same rule as every other write since 2b-v: a workspace we cannot write
        # to loses persistence, not the run. This one matters more than it looks
        # -- every migration call site runs at import or in __init__, before any
        # window exists, so an exception here is an immediate hard crash with no
        # UI to report it. A failed migration just leaves the legacy file where
        # it is, and the next writable run picks it up.
        return False
    return True
