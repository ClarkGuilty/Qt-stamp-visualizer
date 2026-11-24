# Settings and files

This page explains where the tools keep their settings, presets and caches, and lists
every file they write. The flags themselves are described in [usage.md](usage.md).

## Where settings live

**Everything is local by default.** All three tools keep their settings, presets and
caches in a single `.qtstamp/` directory inside **whatever directory you launched them
from** — not in a hidden per-user location. Two datasets worked on from two directories
get two independent sets of settings, copying a working directory takes its settings
along, and deleting `.qtstamp/` resets that workspace and nothing else.

```
<where you launched the tool>/
├── Classifications/            <- your work product (see below)
└── .qtstamp/
    ├── config_lobby.json               preferences_mosaic.json
    ├── preferences_single.json
    ├── presets/                        named lobby presets
    └── cache/
        ├── Legacy_survey/  PanSTARRS/  downloaded cutouts
        └── temp/                       mosaic scratch renders
```

**Reading falls back, writing does not.** A settings file is looked for in `.qtstamp/`
first, then in the per-user config directory (`$XDG_CONFIG_HOME/qtstamp`,
`~/Library/Application Support/qtstamp`, `%APPDATA%\qtstamp`), then in the defaults
shipped with the tool. The first hit wins, so an existing per-user config from an older
version still seeds a new working directory. But saving *only ever* writes to
`.qtstamp/`: the copy further down the chain is left alone, because other working
directories may be reading it too.

If `.qtstamp/` cannot be created — a read-only directory, say — the tools run normally
and simply do not remember anything between runs. They will not fall back to writing
somewhere you did not point them at. (Caches are the one exception: they fall back to the
per-user cache directory, since a mosaic with nowhere to render its scratch tiles cannot
run at all.)

To opt out of local-first, every tool takes:

| Flag | Effect |
| --- | --- |
| `--state-dir PATH` | Use `PATH` instead of `./.qtstamp` — read *and* write. |
| `--global` | Shorthand for `--state-dir <per-user config directory>`: settings shared by every working directory, the way versions before this one behaved. |

`QTSTAMP_STATE_DIR` does the same through the environment, and is how the lobby tells the
viewers it launches to agree with it. The `Classifications/` directory is **not** part of
this — it is your work product, has its own `--classifications-dir` flag, and stays where
you point it.

## Files the tools write

| Path | Contents |
| --- | --- |
| `Classifications/` | Classification CSVs, auto-saved as you work. Named after the session, the number of images and the seed — the same scheme in both viewers, and deliberately not the mosaic's grid shape, so reshaping the grid doesn't fork your classification. Defaults to `./Classifications` relative to wherever you launch the tool, and is created on demand; override with `--classifications-dir` (all three tools, same flag). |
| `Classifications/sessions.json` | One entry per `(tool, path, name, seed)`. For the two viewers: where you are in each dataset — positions are stored as filenames, so they survive a mosaic reshape or files being added and removed (`--reset-position` clears the current one). For the lobby: that session's own classification scheme, band setup and mosaic grid shape, which is what the **Recent sessions...** picker lists. Capped at 50 sessions, least-recently-opened dropped first. Lives beside the CSVs, and records the data path *relative* to them — that is what lets you copy the whole tree elsewhere and resume where you left off. |
| `.qtstamp/config_lobby.json` | Saved lobby settings that belong to the workspace rather than to one session: data path, run mode, output path, extraction options, panel layout. Also the scheme and band setup you last used, as the starting point for a session that has no record of its own yet. |
| `.qtstamp/preferences_mosaic.json`, `.qtstamp/preferences_single.json` | Saved viewer preferences — colormap, scale, panels, toggles. They survive a change of `--path`/`--name`/`--seed` (`--reset-config` clears them). |
| `.qtstamp/presets/` | Named lobby presets. |
| `.qtstamp/cache/Legacy_survey/`, `.qtstamp/cache/PanSTARRS/` | Downloaded cutout caches (`--clean` empties both). |
| `.qtstamp/cache/temp/` | Mosaic scratch renders. Safe to delete at any time. |
| `.config_mosaic.json.bak`, `.config.json.bak` | The pre-split config files, kept after they are migrated. Safe to delete. |

Classification CSVs are named `classification_mosaic_autosave_{name}_{n_stamps}[_{seed}].csv`
(mosaic) and `classification_single_{name}_{n_stamps}[_{seed}].csv` (1-by-1), the seed part
only when you pass one. When a name is already taken, the file that moves aside says why:

- `-new_dataset_N`: the name was taken by a *different* dataset. Pass `-N name` to give your
  sessions distinct names and avoid this.
- `-simultaneous_N`: another viewer is grading this same session right now. Its file is left
  untouched and this viewer's grades go here, so neither set is lost; reconcile them afterwards.

Both count up (`_1`, `_2`, ...). Files with the older `-(N)` suffix are still found, read and
resumed; that suffix is no longer written, since parentheses need quoting in a shell.

Upgrading from an older version: a `.config_lobby.json`, `.predefined_configs/`,
`.preferences_*.json`, `Legacy_survey/`, `PanSTARRS/` or `.temp/` left in your working
directory is moved into `.qtstamp/` on first run. Copies in the per-user directory are
read but never moved, since other working directories may still be using them.
