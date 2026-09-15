"""Tests for imaging.ClassificationWriter -- how classification CSVs are saved.

These files are the only irreplaceable thing the tools produce, and they are
rewritten in full on every grade (a page at a time in the mosaic, an object at a
time in the 1-by-1 tool). So the properties under test here are durability
properties, not formatting ones: a save is all-or-nothing, and a save never
destroys a file this process did not write.

Qt-free, so it runs headless. Plain pytest functions, also runnable directly
(`python tests/test_classification_writer.py`).
"""

import fnmatch
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from imaging import (ClassificationWriter, atomic_to_csv, next_free_csv_path,
                     NEW_DATASET_FORK, SIMULTANEOUS_FORK)

POSIX = sys.platform != 'win32'   # elsewhere chmod only toggles the read-only bit


def frame(n=3, grade=1):
    return pd.DataFrame({'file_name': [f'{i:03d}.fits' for i in range(n)],
                         'classification': [grade] * n})


def temps_in(directory):
    return [f for f in os.listdir(directory) if f.startswith('.tmp-')]


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def mode_of_a_plain_new_file(directory):
    """What an ordinary `open()` in `directory` produces right now.

    Asserting against this rather than a literal 0644 keeps the permission
    tests honest under any umask, including whatever CI happens to run with.
    """
    probe = os.path.join(directory, 'probe.tmp')
    open(probe, 'w').close()
    try:
        return mode(probe)
    finally:
        os.remove(probe)


# --- atomic_to_csv --------------------------------------------------------

def test_write_produces_a_complete_file(tmpdir):
    path = os.path.join(tmpdir, 'c.csv')
    atomic_to_csv(frame(), path, index=False)
    assert len(pd.read_csv(path)) == 3
    assert not temps_in(tmpdir)


def test_a_failed_write_leaves_the_previous_file_untouched(tmpdir):
    "The whole point: an interrupted save must not cost the user their work."
    path = os.path.join(tmpdir, 'c.csv')
    atomic_to_csv(frame(n=5), path, index=False)
    before = open(path).read()

    class Exploding(pd.DataFrame):
        def to_csv(self, *args, **kwargs):
            raise RuntimeError('disk full')

    try:
        atomic_to_csv(Exploding(frame(n=2)), path, index=False)
    except RuntimeError:
        pass
    assert open(path).read() == before
    assert not temps_in(tmpdir), 'a failed write must not leave scratch files'


def test_write_creates_missing_directories(tmpdir):
    path = os.path.join(tmpdir, 'nested', 'deeper', 'c.csv')
    atomic_to_csv(frame(), path, index=False)
    assert os.path.exists(path)


def test_a_new_file_gets_the_same_mode_as_a_plain_write(tmpdir):
    """The atomic write must not be more restrictive than `df.to_csv(path)`.

    Writing via `tempfile.mkstemp` created the temp file at 0600 and
    `os.replace` carried that onto the target, so every classification came out
    owner-only regardless of the user's umask.
    """
    if not POSIX:
        return
    path = os.path.join(tmpdir, 'c.csv')
    atomic_to_csv(frame(), path, index=False)
    assert mode(path) == mode_of_a_plain_new_file(tmpdir)


def test_an_existing_files_mode_is_preserved(tmpdir):
    "An autosave must not change how a file is shared -- in either direction."
    if not POSIX:
        return
    shared = os.path.join(tmpdir, 'shared.csv')
    atomic_to_csv(frame(), shared, index=False)
    os.chmod(shared, 0o664)
    atomic_to_csv(frame(grade=2), shared, index=False)
    assert mode(shared) == 0o664, 'group read was stripped off a shared CSV'

    private = os.path.join(tmpdir, 'private.csv')
    atomic_to_csv(frame(), private, index=False)
    os.chmod(private, 0o600)
    atomic_to_csv(frame(grade=2), private, index=False)
    assert mode(private) == 0o600, 'a deliberately private CSV was widened'


def test_a_failed_write_leaves_the_mode_untouched(tmpdir):
    if not POSIX:
        return
    path = os.path.join(tmpdir, 'c.csv')
    atomic_to_csv(frame(), path, index=False)
    os.chmod(path, 0o640)

    class Exploding(pd.DataFrame):
        def to_csv(self, *args, **kwargs):
            raise RuntimeError('disk full')

    try:
        atomic_to_csv(Exploding(frame()), path, index=False)
    except RuntimeError:
        pass
    assert mode(path) == 0o640
    assert not temps_in(tmpdir)


def test_csv_bytes_are_unchanged_by_the_atomic_path(tmpdir):
    "Writing through a handle instead of a path must not alter the output."
    atomic = os.path.join(tmpdir, 'atomic.csv')
    plain = os.path.join(tmpdir, 'plain.csv')
    atomic_to_csv(frame(n=4), atomic, index=False)
    frame(n=4).to_csv(plain, index=False)
    assert open(atomic, 'rb').read() == open(plain, 'rb').read()


# --- next_free_csv_path ---------------------------------------------------

def test_next_free_path_skips_existing(tmpdir):
    "Each marker numbers itself from 1 and walks up past the names already taken."
    path = os.path.join(tmpdir, 'c.csv')
    assert next_free_csv_path(path, SIMULTANEOUS_FORK) == \
        os.path.join(tmpdir, 'c-simultaneous_1.csv')
    assert next_free_csv_path(path, NEW_DATASET_FORK) == \
        os.path.join(tmpdir, 'c-new_dataset_1.csv')

    open(path, 'w').close()
    open(os.path.join(tmpdir, 'c-simultaneous_1.csv'), 'w').close()
    assert next_free_csv_path(path, SIMULTANEOUS_FORK) == \
        os.path.join(tmpdir, 'c-simultaneous_2.csv')
    assert next_free_csv_path(path, NEW_DATASET_FORK) == \
        os.path.join(tmpdir, 'c-new_dataset_1.csv'), 'the markers number independently'


def test_next_free_path_does_not_stack_markers(tmpdir):
    "Forking a fork replaces its marker rather than appending another one."
    simultaneous = os.path.join(tmpdir, 'c-simultaneous_1.csv')
    open(simultaneous, 'w').close()
    assert next_free_csv_path(simultaneous, SIMULTANEOUS_FORK) == \
        os.path.join(tmpdir, 'c-simultaneous_2.csv'), \
        'must not give c-simultaneous_1-simultaneous_2.csv'
    assert next_free_csv_path(simultaneous, NEW_DATASET_FORK) == \
        os.path.join(tmpdir, 'c-new_dataset_1.csv'), \
        'one de-stacking rule covers both markers, in either direction'


def test_a_legacy_paren_name_forks_to_a_named_marker(tmpdir):
    """`-(N)` was what both markers used to be called, and those files still exist.

    They are still globbed, read and resumed, so a fork of one has to land on a
    named marker rather than stacking onto the old suffix -- otherwise a user
    who started a classification before the rename ends up with
    `c-(1)-simultaneous_1.csv`, which is both ugly and still unquotable.
    """
    legacy = os.path.join(tmpdir, 'c-(1).csv')
    open(legacy, 'w').close()
    assert next_free_csv_path(legacy, SIMULTANEOUS_FORK) == \
        os.path.join(tmpdir, 'c-simultaneous_1.csv')


def test_fork_names_carry_no_parentheses(tmpdir):
    "The reason for the rename: `-(1)` is a syntax error in bash and zsh unquoted."
    path = os.path.join(tmpdir, 'c.csv')
    for marker in (SIMULTANEOUS_FORK, NEW_DATASET_FORK):
        assert not set('()') & set(next_free_csv_path(path, marker))


def test_fork_markers_are_still_found_by_the_viewers_glob():
    """Pins the marker spelling: a stray leading `_` would hide a fork from everyone.

    `mosaic.py`'s seedless discovery globs `{base}-*.csv` and subtracts
    `{base}_*.csv` to keep names that carry a seed out of the running, since `_`
    is the seed marker there. A fork spelled `__simultaneous_N` would read as a
    seeded name and vanish from relaunch and from the lobby's auto-detect --
    silently leaving one viewer's grades behind with nothing left pointing at
    them, which is exactly the failure this naming choice avoids.
    """
    base = 'classification_mosaic_autosave_x_12'
    for marker in (SIMULTANEOUS_FORK, NEW_DATASET_FORK):
        forked = '{}-{}_1.csv'.format(base, marker)
        assert fnmatch.fnmatch(forked, '{}-*.csv'.format(base))
        assert not fnmatch.fnmatch(forked, '{}_*.csv'.format(base))


# --- ClassificationWriter -------------------------------------------------

def test_repeated_saves_keep_the_same_file(tmpdir):
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    assert {writer.save(frame()) for _ in range(5)} == {path}
    assert os.listdir(tmpdir) == ['c.csv']


def test_a_foreign_write_is_never_clobbered(tmpdir):
    """A second viewer on the same session must not lose its grades.

    Both processes hold the same filename; without this the one that saves
    second silently replaces the other's work on every click.
    """
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    writer.save(frame(grade=1))

    foreign = frame(n=9, grade=7)
    foreign.to_csv(path, index=False)      # another process writes it

    forked = writer.save(frame(grade=2))
    assert forked == os.path.join(tmpdir, 'c-simultaneous_1.csv')
    assert len(pd.read_csv(path)) == 9, 'the other process kept its file'
    assert int(pd.read_csv(path)['classification'][0]) == 7
    assert int(pd.read_csv(forked)['classification'][0]) == 2


def test_a_forked_file_gets_the_umask_default(tmpdir):
    "A fork is a new file, so it is created like one rather than inheriting."
    if not POSIX:
        return
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    writer.save(frame())
    os.chmod(path, 0o600)
    frame(n=9).to_csv(path, index=False)   # another process writes it

    forked = writer.save(frame(grade=2))
    assert forked != path
    assert mode(forked) == mode_of_a_plain_new_file(tmpdir)


def test_the_writer_stays_on_the_forked_file(tmpdir):
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    writer.save(frame())
    frame(n=9).to_csv(path, index=False)
    forked = writer.save(frame())
    assert writer.save(frame()) == forked, 'must not fork again every save'


def test_a_brand_new_file_is_not_treated_as_foreign(tmpdir):
    "Nothing on disk yet is the normal first-save case, not a conflict."
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    assert writer.save(frame()) == path


def test_a_file_that_appeared_since_startup_is_not_clobbered(tmpdir):
    """The BUGS.md repro: two viewers racing to create the same brand-new session.

    Both writers start with nothing on disk, so neither has a fingerprint to
    compare against. Guarding on *our own* fingerprint alone let the second
    save through unchallenged -- silently replacing the first viewer's grades
    with no fork and no warning, which is exactly what a user loses if this
    regresses.
    """
    path = os.path.join(tmpdir, 'c.csv')
    first = ClassificationWriter(path, index=False)
    second = ClassificationWriter(path, index=False)   # both see nothing at startup

    assert first.save(frame(grade=1)) == path
    forked = second.save(frame(grade=2))
    assert forked == os.path.join(tmpdir, 'c-simultaneous_1.csv')

    assert int(pd.read_csv(path)['classification'][0]) == 1, \
        "the first viewer's grades were overwritten"
    assert int(pd.read_csv(forked)['classification'][0]) == 2

    assert second.save(frame(grade=2)) == forked, 'must not re-fork every save'


def test_a_third_viewer_takes_the_next_simultaneous_number(tmpdir):
    "Three viewers open on one new session must leave all three sets of grades on disk."
    path = os.path.join(tmpdir, 'c.csv')
    writers = [ClassificationWriter(path, index=False) for _ in range(3)]   # all see nothing
    paths = [w.save(frame(grade=n)) for n, w in enumerate(writers, start=1)]
    assert paths == [path,
                     os.path.join(tmpdir, 'c-simultaneous_1.csv'),
                     os.path.join(tmpdir, 'c-simultaneous_2.csv')]
    assert all(os.path.exists(p) for p in paths)


def test_a_deleted_target_is_recreated_rather_than_forked(tmpdir):
    """A target that vanished has nothing left to protect, so it is simply rewritten.

    Pins the `current is not None` half of the guard: BUGS.md's literal
    suggested fix, dropping `is not None` outright, would fork this save to
    `-simultaneous_1` even though `c.csv` itself sits free again -- stranding
    the user's own grades in a fork of their own file for no reason.
    """
    path = os.path.join(tmpdir, 'c.csv')
    writer = ClassificationWriter(path, index=False)
    writer.save(frame())
    os.remove(path)
    assert writer.save(frame(grade=2)) == path
    assert os.listdir(tmpdir) == ['c.csv']


def test_index_kwargs_are_honoured(tmpdir):
    "The 1-by-1 tool saves with the index; the mosaic without it."
    with_index = os.path.join(tmpdir, 'with.csv')
    ClassificationWriter(with_index).save(frame())
    assert list(pd.read_csv(with_index).columns)[0] == 'Unnamed: 0'

    without = os.path.join(tmpdir, 'without.csv')
    ClassificationWriter(without, index=False).save(frame())
    assert list(pd.read_csv(without).columns) == ['file_name', 'classification']


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
