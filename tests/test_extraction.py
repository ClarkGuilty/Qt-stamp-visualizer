"""Tests for extraction.py: the positives of a classification CSV, linked or copied
into a fresh tree for the chained workflow's 1-by-1 stage.

Qt-free. Plain pytest functions, also runnable directly (`python tests/test_extraction.py`).
"""

import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtstamp import extraction


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(os.path.basename(path))
    return path


def _csv(path, rows):
    "rows: (file_name, classification). Written as the mosaic writes it, numeric codes."
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('file_name,classification,grid_pos,page\n')
        for name, code in rows:
            f.write(f'{name},{code},0,0\n')
    return path


ROWS = [('a.fits', 1), ('b.fits', 0), ('c.fits', 1)]


def test_band_directories_are_mirrored(tmpdir):
    src = join(tmpdir, 'src')
    for band in ('VIS', 'H'):
        for name in ('a.fits', 'b.fits', 'c.fits'):
            if (band, name) != ('H', 'c.fits'):
                _touch(join(src, band, name))
    result = extraction.extract(_csv(join(tmpdir, 'm.csv'), ROWS), src, join(tmpdir, 'out'), {1})
    assert result.bands_processed == ['H', 'VIS']
    assert sorted(os.listdir(join(tmpdir, 'out', 'VIS'))) == ['a.fits', 'c.fits']
    assert sorted(os.listdir(join(tmpdir, 'out', 'H'))) == ['a.fits']
    assert result.missing == [('H', 'c.fits')]
    assert os.path.islink(join(tmpdir, 'out', 'VIS', 'a.fits'))


def test_mef_files_land_directly_in_the_output(tmpdir):
    "BUGS 41: an MEF dataset has no band directories, so nothing was extracted."
    src = join(tmpdir, 'src')
    for name in ('a.fits', 'b.fits', 'c.fits'):
        _touch(join(src, name))
    out = join(tmpdir, 'out')
    result = extraction.extract(_csv(join(tmpdir, 'm.csv'), ROWS), src, out, {1},
                                use_symlink=False)
    assert sorted(os.listdir(out)) == ['a.fits', 'c.fits']
    assert (result.copied, result.missing, result.bands_processed) == (2, [], ['(MEF)'])
    with open(join(out, 'a.fits')) as f:
        assert f.read() == 'a.fits'


def test_output_on_the_source_is_refused_before_touching_it(tmpdir):
    "Every destination would be its own source: deleted, then linked to itself."
    for layout in ('mef', 'bands'):
        src = join(tmpdir, layout)
        stamp = _touch(join(src, 'a.fits') if layout == 'mef' else join(src, 'VIS', 'a.fits'))
        try:
            extraction.extract(_csv(join(tmpdir, 'm.csv'), ROWS), src, src + os.sep, {1})
        except ValueError as e:
            assert 'data path' in str(e)
        else:
            raise AssertionError(f'{layout}: extracted onto its own source')
        assert os.path.isfile(stamp) and not os.path.islink(stamp)


def test_string_codes_from_the_single_viewer(tmpdir):
    path = join(tmpdir, 's.csv')
    with open(path, 'w') as f:
        f.write(',file_name,classification\n0,a.png,A\n1,b.png,B\n2,c.png,Empty\n')
    assert extraction.select_positive_filenames(path, {'A', 'B'}) == ['a.png', 'b.png']


def test_objects_are_matched_by_stem_across_formats(tmpdir):
    "The CSV names the object's file in one band; another band may hold it as a JPG."
    src = join(tmpdir, 'src')
    for name in ('a.fits', 'b.fits', 'c.fits'):
        _touch(join(src, 'VIS', name))
    for name in ('a.JPG', 'b.JPG', 'c.png'):
        _touch(join(src, 'rgb', name))
    _touch(join(src, 'Classifications', 'old.csv'))     # not a band: no stamps in it
    _touch(join(src, '.qtstamp', 'cache', 'x.png'))
    result = extraction.extract(_csv(join(tmpdir, 'm.csv'), ROWS), src, join(tmpdir, 'out'), {1})
    assert result.bands_processed == ['VIS', 'rgb'] and result.missing == []
    assert sorted(os.listdir(join(tmpdir, 'out', 'VIS'))) == ['a.fits', 'c.fits']
    assert sorted(os.listdir(join(tmpdir, 'out', 'rgb'))) == ['a.JPG', 'c.png']
    assert sorted(os.listdir(join(tmpdir, 'out'))) == ['VIS', 'rgb']


def test_a_compressed_fits_object_matches_its_other_bands(tmpdir):
    "BUGS 60: the CSV names a.fits.gz; the object is a, as a.png in the other band."
    src = join(tmpdir, 'src')
    for name in ('a.fits.gz', 'b.fits.fz', 'c.fits.gz'):
        _touch(join(src, 'VIS', name))
    for name in ('a.png', 'b.JPG', 'c.png'):
        _touch(join(src, 'rgb', name))
    rows = [('a.fits.gz', 1), ('b.fits.fz', 0), ('c.fits.gz', 1)]
    result = extraction.extract(_csv(join(tmpdir, 'm.csv'), rows), src, join(tmpdir, 'out'), {1})
    assert result.missing == []
    assert sorted(os.listdir(join(tmpdir, 'out', 'VIS'))) == ['a.fits.gz', 'c.fits.gz']
    assert sorted(os.listdir(join(tmpdir, 'out', 'rgb'))) == ['a.png', 'c.png']


def test_files_are_copied_where_symlinks_are_refused(tmpdir):
    "exFAT and FAT drives: every symlink fails, and nothing used to be extracted."
    src = join(tmpdir, 'src')
    for name in ('a.fits', 'b.fits', 'c.fits'):
        _touch(join(src, 'VIS', name))
    messages = []
    real_symlink = os.symlink

    def refuse(*args, **kwargs):
        raise OSError(1, 'Operation not permitted')
    os.symlink = refuse
    try:
        result = extraction.extract(_csv(join(tmpdir, 'm.csv'), ROWS), src, join(tmpdir, 'out'),
                                    {1}, on_progress=messages.append)
    finally:
        os.symlink = real_symlink
    assert (result.linked, result.copied) == (0, 2)
    assert sorted(os.listdir(join(tmpdir, 'out', 'VIS'))) == ['a.fits', 'c.fits']
    assert not os.path.islink(join(tmpdir, 'out', 'VIS', 'a.fits'))
    assert sum('copying the files instead' in m for m in messages) == 1


def test_a_flat_directory_of_images_is_extracted_flat(tmpdir):
    src = join(tmpdir, 'src')
    for name in ('a.png', 'b.png', 'c.jpg'):
        _touch(join(src, name))
    rows = [('a.png', 1), ('b.png', 0), ('c.jpg', 1)]
    result = extraction.extract(_csv(join(tmpdir, 'm.csv'), rows), src, join(tmpdir, 'out'), {1})
    assert result.bands_processed == ['(flat)']
    assert sorted(os.listdir(join(tmpdir, 'out'))) == ['a.png', 'c.jpg']


def test_a_class_pandas_reads_as_missing_is_still_selected(tmpdir):
    "NA, None, null... are missing values to pandas, and matched no positive class."
    path = join(tmpdir, 'c.csv')
    with open(path, 'w') as f:
        f.write(',file_name,classification,subclassification\n'
                '0,a.fits,NA,NA\n1,b.fits,B,B\n2,c.fits,None,None\n3,d.fits,,\n')
    assert extraction.select_positive_filenames(path, {'NA'}) == ['a.fits']
    assert extraction.select_positive_filenames(path, {'None', 'B'}) == ['b.fits', 'c.fits']


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
