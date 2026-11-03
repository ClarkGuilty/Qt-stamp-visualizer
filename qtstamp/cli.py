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

from . import paths
from . import state

# What a viewer uses when -b/-B/--rgb-composites are not given. The lobby omits
# -b when no main band is set, so it needs the first of these too.
DEFAULT_MAIN_BAND = 'VIS'
DEFAULT_COLOR_BANDS = 'Y,J,H'
DEFAULT_RGB_COMPOSITES = 'H,Y,I;H,J,Y'


def add_dataset_args(parser):
    "Registers the arguments both viewers take: the dataset, its bands, and where its state lives."
    parser.add_argument('-p', "--path", help="Path to the images to inspect.",
                        default="Color_stamps_to_inspect")
    parser.add_argument('-N', "--name", help="Name of the classifying session.",
                        default=None)
    parser.add_argument('-s', "--seed", help="Seed used to shuffle the images.", type=int,
                        default=None)
    parser.add_argument('-b', "--main_band", help='High resolution band. Example: "VIS". This is also the '
                        "tool's default band: the panel that's always individually shown and pre-selected. "
                        "Pass -b '' for no main band (PNG/JPG sets, which have no WCS): the first -B "
                        "band then takes its place.",
                        default=DEFAULT_MAIN_BAND)
    parser.add_argument('-B', "--color_bands", help='Comma-separated bands to make individually selectable as '
                        'their own panel (in addition to the RGB composites). Example: "Y,J,H"',
                        default=DEFAULT_COLOR_BANDS)
    parser.add_argument('--rgb-composites',
                        help='RGB composites: semicolon-separated R,G,B band-name triples (any directory name '
                        "under --path can be a band -- not just VIS/Y/J/H/I). A composite's own name/label is "
                        'simply its comma-joined member list. Example: "H,Y,I;H,J,Y". '
                        'All three bands in a composite must have the exact same image dimensions '
                        '(and ideally the same zero-point) -- mismatched bands cannot be stacked into one RGB image.',
                        default=DEFAULT_RGB_COMPOSITES)
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


def finish_dataset_args(parser, args):
    """Normalises the parsed dataset arguments in place, into what both viewers use.

    `color_bands` becomes a list; `main_band` is filled in when -b '' asked for
    none; `composite_bands` (ordered keys, "H,Y,I") and `composite_band_members`
    (key -> (r, g, b)) come from --rgb-composites; `state_dir_override` from
    --state-dir/--global. Exits, the way argparse does, on arguments that cannot
    work -- including a band with no directory under --path.
    """
    args.state_dir_override = paths.resolve_state_dir_override(args)
    args.main_band, args.color_bands = resolve_main_band(args.main_band,
                                                         split_bands(args.color_bands))
    if not args.main_band:
        parser.error('-b "" (no main band) needs at least one -B color band to take its place.')

    try:
        composites = parse_rgb_composites(args.rgb_composites)
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
    available = sorted(d for d in os.listdir(args.path) if os.path.isdir(join(args.path, d))) \
        if os.path.isdir(args.path) else []
    print(f"Band director{'y' if len(missing) == 1 else 'ies'} not found under {args.path}: "
          f"{', '.join(sorted(missing))}")
    print(f"Available subdirectories: {', '.join(available) if available else '(none found)'}")
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


def duplicate_subclasses(entries):
    """{sub: count} for every subclass declared more than once. One button per
    subclass name is what lights up on a resumed row -- the CSV stores only the
    name -- so a second entry would take the first one's place."""
    counts = {}
    for e in entries:
        if e.sub:
            counts[e.sub] = counts.get(e.sub, 0) + 1
    return {sub: n for sub, n in counts.items() if n > 1}
