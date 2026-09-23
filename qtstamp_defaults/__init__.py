"""Defaults shipped with the tool -- the last entry on `paths.config_search_path`.

A package rather than a bare directory so that `pip install` carries the JSON
along (`[tool.setuptools.package-data]` in pyproject.toml) and so the files are
addressable under `site-packages` the same way they are in a checkout. Nothing
ever writes here: a workspace that picks one of these up and changes it saves
the result to its own `.qtstamp/`, leaving this copy alone.

Contents are deliberately partial. `presets/*.json` hold only the band and
classification-scheme keys, never paths or session names, because a preset is
merged on top of whatever the user already has (`lobby._apply_preset`) -- a
shipped preset that carried `data_path` would stomp it. The full defaults for
everything else stay in code (`lobby.DEFAULT_CONFIG`, and each viewer's
preference defaults); duplicating them here would only create two copies to
drift apart.
"""
