# This Python file uses the following encoding: utf-8
"""The viewers' command line, defined once.

mosaic.py and single_viewer.py take the same dataset arguments and normalise
them the same way, and lobby.py writes those arguments -- plus the 1-by-1
viewer's --classifications string -- for the viewers to read. Readers and
writer share this module so they cannot drift apart, the way the lobby's own
copy of the viewers' CSV naming did (BUGS.md 39, 40).

Qt-free, and nothing here runs at import: each viewer parses its arguments in
`main()`, so a test can import a viewer and hand it an argv of its own.
"""

import os
import sys
from collections import namedtuple
from os.path import join

import pandas as pd

from . import fits_io
from . import paths
from . import state

# What a viewer looks for when -b/-B/--rgb-composites are not given (see
# resolve_bands for what it does when they are not there). The lobby omits -b when
# no main band is set, so it needs the first of these too.
DEFAULT_MAIN_BAND = 'VIS'
DEFAULT_COLOR_BANDS = 'Y,J,H'
DEFAULT_RGB_COMPOSITES = 'H,Y,I;H,J,Y'
FLAT_BAND = fits_io.FLAT_BAND      # -b . : the images directly under --path


def add_dataset_args(parser):
    "Registers the arguments both viewers take: the dataset, its bands, and where its state lives."
    parser.add_argument('-p', "--path", help="Path to the images to inspect.",
                        default="Color_stamps_to_inspect")
    parser.add_argument('-N', "--name", help="Name of the classifying session.",
                        default=None)
    parser.add_argument('-s', "--seed", help="Seed used to shuffle the images.", type=int,
                        default=None)
    parser.add_argument('-b', "--main_band", help='The default band: the panel shown first and '
                        'pre-selected. Example: "VIS". No band needs every object: an object is '
                        "listed when any band has it, and a band it lacks shows as \"no data\". "
                        f"Default: {DEFAULT_MAIN_BAND} if --path has it, else the first -B band "
                        "(pass -b '' to ask for that), else the first band found under --path.",
                        default=None)
    parser.add_argument('-B', "--color_bands", help='Comma-separated bands to make individually selectable as '
                        'their own panel (in addition to the RGB composites). Example: "Y,J,H". '
                        f'Default: those of {DEFAULT_COLOR_BANDS} that --path has; with no band '
                        'argument at all and none of the default bands there, every other '
                        'band found under --path.',
                        default=None)
    parser.add_argument('--rgb-composites',
                        help='RGB composites: semicolon-separated R,G,B band-name triples (any directory name '
                        "under --path can be a band -- not just VIS/Y/J/H/I). A composite's own name/label is "
                        'simply its comma-joined member list. Example: "H,Y,I;H,J,Y". '
                        'A composite is made for an object only when all three bands have a FITS stamp '
                        'of the same dimensions for it; otherwise its panel reads "no data". '
                        f'Default: those of {DEFAULT_RGB_COMPOSITES} whose bands --path has.',
                        default=None)
    parser.add_argument("--mef", help="Treat --path as a directory of multi-extension FITS "
                        "files -- one file per object, with bands as HDU extensions "
                        "(identified by EXTNAME) inside it, instead of one subdirectory per "
                        "band. Bands are looked up by EXTNAME in each object's own file, so "
                        "files may differ; a band a file lacks shows as a placeholder. Names "
                        "passed to -b/-B/--rgb-composites must match an EXTNAME some file has.",
                        action="store_true", default=False)
    parser.add_argument("--classifications-dir", metavar="PATH",
                        help="Directory for the autosaved classification CSVs. Absolute, or "
                        "relative to the directory you launch from; created if it does not "
                        "exist. Default: ./Classifications",
                        default=None)
    parser.add_argument("--reset-config", help="Forgets the saved preferences (colormap, scale, "
                        "panels and the like) during startup. Resume positions are kept; use "
                        "--reset-position for those.",
                        action="store_true", default=False)
    parser.add_argument("--reset-position", help="Forgets the saved resume position for this "
                        "--path/--name/--seed, so the session restarts at the beginning. "
                        "Preferences are kept.",
                        action="store_true", default=False)
    paths.add_state_dir_args(parser)


def split_bands(text):
    """'Y, J,,H' -> ['Y', 'J', 'H']. Empty entries are dropped: -B '' means "no
    color bands", not one band named '' -- which would pass the band-directory
    check, since it resolves to --path itself, and then fail at image load."""
    return [b.strip() for b in text.split(',') if b.strip()]


def resolve_main_band(main_band, color_bands):
    """`(main band, color bands)` as a viewer uses them. -b '' means "no main band":
    PNG/JPG stamps carry no WCS, which is most of what a main band is for, so the
    first color band takes its other job -- the object list and the default panel --
    and the rest of the tool never sees an empty main band. The main band comes
    back '' only when there is no color band to take it. The main band is never
    also a color band: it would show twice in the panel pickers (BUGS.md 42)."""
    main_band = (main_band or '').strip()
    color_bands = list(color_bands)
    if not main_band and color_bands:
        main_band = color_bands[0]
    return main_band, [b for b in color_bands if b != main_band]


def parse_rgb_composites(text):
    """'H,Y,I;H,J,Y' -> [('H', 'Y', 'I'), ('H', 'J', 'Y')], in order, empty entries
    skipped. Raises ValueError on an entry that is not an R,G,B triple."""
    composites = []
    for entry in text.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        members = tuple(b.strip() for b in entry.split(','))
        if len(members) != 3:
            raise ValueError(f'--rgb-composites entry "{entry}" must have exactly 3 '
                             'comma-separated bands (R,G,B).')
        composites.append(members)
    return composites


def resolve_bands(path, main_band, color_bands, composites, mef, note=print):
    """`(main_band, color_bands, composites)` to run with, each given as its argument's
    text or None when the argument was not passed. Raises ValueError, with the message
    to exit on, when no usable setup comes out.

    What was passed is taken as it is. What was not is the default *if the data has
    it*, so a dataset that is not laid out as VIS/Y/J/H/I opens without naming its
    bands:

    - With no band argument at all and no default main band under `path`, the bands
      are whatever is there: every subdirectory holding stamps, the first as the
      default band; or, when the stamps sit directly under `path`, that one flat
      directory (`FLAT_BAND`). No composites.
    - Otherwise an unpassed -B keeps the default color bands that exist, an unpassed
      --rgb-composites the default composites whose three bands exist, and an
      unpassed -b is the default main band if it exists, else the first color band.

    Under `mef` bands are extensions inside the files, which are not opened here:
    unpassed arguments are simply the defaults, checked when the window opens."""
    passed_any = not (main_band is None and color_bands is None and composites is None)
    colors = split_bands(DEFAULT_COLOR_BANDS if color_bands is None else color_bands)
    triples = parse_rgb_composites(DEFAULT_RGB_COMPOSITES if composites is None else composites)
    if mef:
        main, colors = resolve_main_band(DEFAULT_MAIN_BAND if main_band is None else main_band, colors)
        if not main:
            raise ValueError('-b "" (no main band) needs at least one -B color band to take its place.')
        return main, colors, triples

    def exists(band):
        return os.path.isdir(join(path, band))

    if not passed_any and not exists(DEFAULT_MAIN_BAND):
        found = fits_io.list_band_dirs(path)
        if found:
            note(f"No band arguments and no {DEFAULT_MAIN_BAND} under {path}: using the bands "
                 f"found there -- {', '.join(found)} ({found[0]} shown first).")
            return found[0], found[1:], []
        if fits_io.has_stamps(path):
            note(f"No band directories under {path}: showing the images directly in it.")
            return FLAT_BAND, [], []
        raise ValueError(f"No FITS, PNG, or JPG stamps found under {path}, nor in a subdirectory of it.")

    if color_bands is None:
        colors = [b for b in colors if exists(b)]
    if composites is None:
        triples = [t for t in triples if all(exists(b) for b in t)]
    if main_band is None:
        main, colors = resolve_main_band(DEFAULT_MAIN_BAND if exists(DEFAULT_MAIN_BAND) else '', colors)
        if not main:
            raise ValueError(f"No default band: -b was not passed, there is no {DEFAULT_MAIN_BAND} "
                             f"under {path}, and no -B band to take its place. Pass -b.")
    else:
        main, colors = resolve_main_band(main_band, colors)
        if not main:
            raise ValueError('-b "" (no main band) needs at least one -B color band to take its place.')
    return main, colors, triples


def finish_dataset_args(parser, args):
    """Normalises the parsed dataset arguments in place, into what both viewers use.

    `main_band`, `color_bands` (a list) and the composites come from `resolve_bands`,
    which fills in the ones not passed from what --path holds; `composite_bands`
    (ordered keys, "H,Y,I") and `composite_band_members` (key -> (r, g, b)) are the
    composites; `state_dir_override` comes from --state-dir/--global. Exits, the way
    argparse does, on arguments that cannot work -- including a band that was asked
    for and has no directory under --path.
    """
    args.state_dir_override = paths.resolve_state_dir_override(args)
    try:
        args.main_band, args.color_bands, composites = resolve_bands(
            args.path, args.main_band, args.color_bands, args.rgb_composites, args.mef)
    except ValueError as e:
        parser.error(str(e))
    args.composite_bands = [','.join(members) for members in composites]
    args.composite_band_members = {','.join(members): members for members in composites}

    if not args.mef:
        # Under --mef there are no per-band subdirectories to check here -- band names
        # are validated against the files' extensions instead, once the window opens them.
        _exit_on_missing_band_dirs(args)


def _exit_on_missing_band_dirs(args):
    referenced = ({args.main_band} | set(args.color_bands) |
                  {b for members in args.composite_band_members.values() for b in members})
    missing = [b for b in referenced if not os.path.isdir(join(args.path, b))]
    if not missing:
        return
    available = fits_io.list_band_dirs(args.path)
    print(f"Band director{'y' if len(missing) == 1 else 'ies'} not found under {args.path}: "
          f"{', '.join(sorted(missing))}")
    print(f"Available subdirectories: {', '.join(available) if available else '(none found)'}")
    if fits_io.has_stamps(args.path):
        print(f"There are images directly in {args.path}: leave out -b/-B/--rgb-composites "
              f"to show those, or pass -b {fits_io.FLAT_BAND}")
    sys.exit(1)


def apply_reset_flags(args, tool, legacy_config_file):
    """--reset-config and --reset-position, run at startup before the window opens.

    `legacy_config_file` is the tool's pre-split config (state.migrate_legacy_config):
    it is moved aside too, or the migration would restore what was just reset."""
    if args.reset_config:
        state.migrate_preferences_file(tool, config_dir=args.state_dir_override)
        if state.reset_preferences(tool, config_dir=args.state_dir_override):
            print("Preferences reset.")
        if os.path.exists(legacy_config_file):
            os.replace(legacy_config_file, legacy_config_file + '.bak')

    if args.reset_position:
        anchor = paths.resolve_classifications_dir(args.classifications_dir)
        state.migrate_sessions_store(anchor)
        if state.forget_session(state.SessionId(tool, args.path, args.name or '', args.seed),
                                anchor):
            print("Resume position reset.")


# --- the 1-by-1 viewer's --classifications ------------------------------------
#
# Semicolon-separated entries. MAJOR=KEY is a major-class button; MAJOR:SUB=KEY a
# subclass button under MAJOR; MAJOR1,MAJOR2:SUB=KEY a subclass shared by several
# majors. An empty SUB is a major-class button. The lobby writes this string from
# its scheme table and the viewer reads it back.

SchemeEntry = namedtuple('SchemeEntry', 'major sub key')
SchemeEntry.__doc__ = ("One --classifications entry. `major` is the text before ':' "
                       "('A', or 'A,B' for a shared subclass); `sub` is '' for a major-class button.")


def split_majors(text):
    "A subclass's majors ('A' or 'A,B') as a list of names, in order, no repeats."
    return list(dict.fromkeys(m.strip() for m in (text or '').split(',') if m.strip()))


def format_classifications(entries):
    "SchemeEntry list -> the --classifications string."
    return ';'.join(f"{e.major}:{e.sub}={e.key}" if e.sub else f"{e.major}={e.key}"
                    for e in entries)


def parse_classifications(text):
    "The --classifications string -> SchemeEntry list, in declared order."
    entries = []
    for entry in text.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        rest, _, key = entry.rpartition('=')
        major, _, sub = rest.partition(':')
        entries.append(SchemeEntry(major.strip(), sub.strip(), key.strip()))
    return entries


def scheme_buttons(entries):
    """`(major_classes, subclasses)` for the 1-by-1 viewer's buttons, in declared
    order: `[(major, key)]` and `[(majors, sub, key)]`, `majors` a tuple."""
    major_classes = [(e.major, e.key) for e in entries if not e.sub]
    subclasses = [(tuple(split_majors(e.major)) or ('',), e.sub, e.key)
                  for e in entries if e.sub]
    return major_classes, subclasses


def duplicate_keys(entries):
    "{key: [labels]} for every keyboard shortcut given to more than one button."
    seen = {}
    for e in entries:
        seen.setdefault(e.key, []).append(f"{e.major}:{e.sub}" if e.sub else e.major)
    return {key: labels for key, labels in seen.items() if len(labels) > 1}


# What an ungraded row holds in the CSV, so never a class: a button with one of these
# names writes it, and the row then reads as ungraded.
RESERVED_LABELS = ('None', 'Empty')


def reserved_labels(entries):
    "The majors and subclasses in `entries` named with a RESERVED_LABELS word, in order."
    names = [e.sub if e.sub else e.major for e in entries]
    return list(dict.fromkeys(n for n in names if n in RESERVED_LABELS))


def unknown_labels(df, entries):
    """{column: {label: rows}}: the labels in a classification CSV's `classification`
    and `subclassification` columns that the scheme `entries` makes no button for.
    A major's button writes its own name into subclassification too, so majors are
    known subclasses. Blank cells, 'None' and 'Empty' are not labels. Compared as
    stored, since a button lights up only for its exact name."""
    majors = {e.major for e in entries if not e.sub}
    known = {'classification': majors,
             'subclassification': majors | {e.sub for e in entries if e.sub}}
    unknown = {}
    for column, names in known.items():
        counts = unknown[column] = {}
        if column not in df.columns:
            continue
        for label in df[column]:
            if pd.isna(label) or label in names or str(label).strip() in ('', *RESERVED_LABELS):
                continue
            counts[label] = counts.get(label, 0) + 1
    return unknown


def duplicate_subclasses(entries):
    """{sub: count} for every subclass declared more than once. One button per
    subclass name is what lights up on a resumed row -- the CSV stores only the
    name -- so a second entry would take the first one's place."""
    counts = {}
    for e in entries:
        if e.sub:
            counts[e.sub] = counts.get(e.sub, 0) + 1
    return {sub: n for sub, n in counts.items() if n > 1}
