"""Regenerate the README screenshots in docs/img/.

    python docs/make_screenshots.py STAMPS_DIR

STAMPS_DIR holds one directory per band -- VIS, I, Y, J and H, as FITS. The README's
images are from public Euclid Early Release Observations: the grade A and B strong-lens
candidates of Acevedo Barroso et al. 2025 (arXiv:2408.06217, Perseus) and Nagam et al.
2025 (arXiv:2502.09802, the other ERO fields), the stamps named `<field>_<id>.fits`.
Any dataset in that layout works; names not in the picks below fall back to the first
ones found.
Images made from ERO data carry ESA's requested credit, "Data from
ESA/Euclid/Euclid Consortium/NASA/ERO2024" (https://doi.org/10.57780/esa-qmocze3).

Runs offscreen in a throwaway workspace that links to the stamps: no network (the
1-by-1's survey panels are off), and nothing written to your own settings or to
STAMPS_DIR.
"""

import json
import os
import shutil
import sys
import tempfile
from os.path import abspath, dirname, join

REPO = dirname(dirname(abspath(__file__)))
OUT = join(REPO, 'docs', 'img')
sys.path.insert(0, REPO)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

from PySide6 import QtWidgets  # noqa: E402

BANDS = ('VIS', 'I', 'Y', 'J', 'H')
PRESET = 'Euclid ERO (VIS + NISP)'
# The preset's scheme, as the 1-by-1 takes it on the command line.
SCHEME = 'Lens=1;Lens:Arc=q;Lens:Ring=w;Maybe=2;Non-lens=3;Artefact=4'
MOSAIC_PICKS = ['Abell2390_455468', 'Fornax_219844', 'Dorado_100073', 'Perseus_428104',
                'Abell2764_1893', 'Dorado_486217', 'Dorado_145610', 'HolmbergII_37174',
                'Abell2764_450232', 'Perseus_126558', 'NGC2403_28538', 'IC10_749202',
                'Abell2764_489937', 'Perseus_427987', 'Dorado_150733', 'NGC6744_551168']
MOSAIC_LENS = 'Abell2764_450232'     # shown as marked
SINGLE_PICK = 'Abell2390_455468'     # shown graded Lens / Arc


def link_subset(stamps, names, dest):
    "Per-band directories in `dest` linking to `names` (stems) in `stamps`."
    for band in BANDS:
        os.makedirs(join(dest, band))
        for name in names:
            os.symlink(join(stamps, band, name + '.fits'), join(dest, band, name + '.fits'))
    return dest


def settle(app, win, size):
    win.resize(*size)
    win.show()
    for _ in range(5):
        app.processEvents()


def main(stamps):
    found = sorted(f[:-5] for f in os.listdir(join(stamps, 'VIS')) if f.endswith('.fits'))
    picks = [n for n in MOSAIC_PICKS if n in found] or found[:16]
    work = tempfile.mkdtemp(prefix='qtstamp-screenshots-')
    os.chdir(work)
    # Under a neutral name: the lobby shows the path.
    os.symlink(abspath(stamps), join(work, 'ero_lens_candidates'))
    stamps = join(work, 'ero_lens_candidates')
    os.environ['QTSTAMP_CONFIG_DIR'] = join(work, 'no-user-dir')
    os.makedirs(OUT, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtWidgets.QMessageBox.warning = lambda *a, **k: None   # a modal would hang offscreen

    from qtstamp import lobby, mosaic, single_viewer

    def common(data):
        return ['-p', data, '-N', 'ero', '--state-dir', join(work, 'state'),
                '--classifications-dir', join(work, 'cls')]

    # Mosaic: a hand-picked page, in a fixed order (no seed), one stamp marked.
    page = link_subset(stamps, picks, join(work, 'mosaic_page'))
    win = mosaic.MosaicVisualizer(mosaic.parse_args(common(page) + ['-l', '4', '-m', '4',
                                                                     '--no-printname']))
    settle(app, win, (1280, 600))
    if MOSAIC_LENS in picks:
        i = [str(n) for n in win.listimage].index(MOSAIC_LENS + '.fits')
        win.my_label_clicked(None, i, mosaic.C_LENS)
        win.buttons[i].set_candidate_status(mosaic.C_LENS)
        win.buttons[i].paint_background_pixmap(win.lens_background_pixmap)
        app.processEvents()
    win.grab().save(join(OUT, 'mosaic.jpg'), 'JPG', 90)
    win.close()

    # 1-by-1: one object, graded with a subclass, auto-next off so the grade stays lit.
    # The main band and both composites in the first row.
    with open(join(work, 'state', 'preferences_single.json'), 'w') as f:
        json.dump({'row_1': 'VIS;H,Y,I;H,J,Y'}, f)
    win = single_viewer.ApplicationWindow(
        single_viewer.parse_args(common(stamps) + ['-s', '1234', '--no-legacysurvey',
                                                   '--classifications', SCHEME]),
        clipboard=app.clipboard())
    name = (SINGLE_PICK if SINGLE_PICK in found else found[0]) + '.fits'
    win.counter = [str(n) for n in win.listimage].index(name)
    win.go_to_counter_page()
    win.config_dict['autonext'] = False
    win.classify_subclass(['Lens'], 'Arc')
    settle(app, win, (1280, 640))
    win.grab().save(join(OUT, 'single_viewer.png'))
    win.close()

    # Lobby: a returning user's (saved config pointing at the data -- opened with no
    # path, the lobby clears its default composites, BUGS 50), with the preset picked.
    with open(join(work, 'state', 'config_lobby.json'), 'w') as f:
        json.dump({'data_path': stamps, 'session_name': 'ero', 'seed_enabled': True,
                   'seed_value': 1234, 'output_path': join(work, 'ero_positives')}, f)
    win = lobby.LobbyWindow(state_dir_override=join(work, 'state'))
    bar = win.preset_bar
    bar.combo.setCurrentIndex(bar.combo.findText(PRESET))
    bar._on_activated(bar.combo.currentIndex())
    win.log_view.clear()
    settle(app, win, (1280, 860))
    win.grab().save(join(OUT, 'lobby.png'))
    win.close()
    os.chdir(REPO)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Wrote {', '.join(sorted(os.listdir(OUT)))} to {OUT}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
