"""Fingerprint what both viewers produce on two fake datasets, so a refactor
can show it changed nothing (REFACTOR.md). Not a test: pytest does not collect it.

    python tests/snapshot_viewers.py OUT.json [SRC_DIR]
    python tests/snapshot_viewers.py --compare BEFORE.json AFTER.json

SRC_DIR defaults to this checkout. To snapshot another revision, extract it
(`git archive REV | tar -x -C DIR`) and pass DIR; it must be from after REFACTOR.md
step 1, when the viewers started parsing their arguments in main(). Run it from a
directory outside the checkout. Each case runs in a process of its own, offscreen,
in a fresh temporary workspace. The datasets come from sample_data.py, written once
per run; they are seeded, so two runs see the same pixels.

Per case it records the CSV's name, the object order (the shuffle), the CSV minus
its `time` column, the state files written, and an md5 of every mosaic tile or
1-by-1 panel -- on two objects or pages, and again under every scale.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ('mef', 'jpg')
TOOLS = ('mosaic', 'single')


def _case_argv(case, data):
    if case == 'mef':
        return ['-p', os.path.join(data, 'mef'), '--mef', '-b', 'VIS_BGSUB',
                '-B', 'NIR_Y_BGSUB,NIR_J_BGSUB,NIR_H_BGSUB',
                '--rgb-composites', 'NIR_H_BGSUB,NIR_J_BGSUB,NIR_Y_BGSUB', '-N', 'snap', '-s', '7']
    return ['-p', os.path.join(data, 'jpg'), '-b', '', '-B', 'grey,rgb',
            '--rgb-composites', '', '-N', 'snap']


def _md5(data):
    return hashlib.md5(data).hexdigest()


def _snapshot_mosaic(argv):
    import mosaic
    win = mosaic.MosaicVisualizer(mosaic.parse_args(argv + ['-l', '3', '-m', '2']))

    def tiles():
        return {f: _md5(open(os.path.join(win.scratchpath, f), 'rb').read())
                for f in sorted(os.listdir(win.scratchpath))}

    out = {'page0': tiles()}
    win.my_label_clicked(None, 0, mosaic.C_LENS)
    win.my_label_clicked(None, 2, mosaic.C_INTERESTING)
    win.next()
    out['page1'] = tiles()
    for i in range(win.cbscale.count()):
        win.cbscale.setCurrentIndex(i)
        out[f'page1_scale_{win.cbscale.currentText()}'] = tiles()
    return win, out


def _snapshot_single(argv, qapp):
    import numpy as np
    import single_viewer
    win = single_viewer.ApplicationWindow(single_viewer.parse_args(argv + ['--no-legacysurvey']),
                                          clipboard=qapp.clipboard())
    win.show()
    qapp.processEvents()

    def panels():
        shown = {}
        for band, canvas in win.canvas.items():
            if canvas.isVisible() or band in win.images:
                canvas.draw()
                shown[band] = _md5(np.asarray(canvas.buffer_rgba()).tobytes())
        return shown

    out = {'object0': panels()}
    win.classify_major('A')          # auto-next is on by default
    out['object1'] = panels()
    for i in range(win.cbscale.count()):
        win.cbscale.setCurrentIndex(i)
        out[f'object1_scale_{win.cbscale.currentText()}'] = panels()
    return win, out


def _run_case(src, tool, case, data, out_json):
    sys.path.insert(0, src)
    work = tempfile.mkdtemp(prefix=f'snapshot-{tool}-{case}-')
    os.chdir(work)
    argv = _case_argv(case, data) + ['--state-dir', os.path.join(work, 'state'),
                                     '--classifications-dir', os.path.join(work, 'cls')]
    from PySide6 import QtWidgets
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win, result = _snapshot_mosaic(argv) if tool == 'mosaic' else _snapshot_single(argv, qapp)

    import pandas as pd
    df = pd.read_csv(win.df_name)
    result['csv_name'] = os.path.basename(win.df_name)
    result['object_order'] = [str(name) for name in win.listimage]
    result['csv_without_time'] = _md5(df.drop(columns=['time']).to_csv(index=False).encode())
    result['state_files'] = sorted(os.listdir(os.path.join(work, 'state')))
    win.close()
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)


def snapshot(out_json, src=REPO):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg',
               QTSTAMP_CONFIG_DIR=os.path.join(tempfile.gettempdir(), 'qtstamp-snapshot-no-user-dir'))
    env.pop('QTSTAMP_STATE_DIR', None)
    from sample_data import write_band_dirs, write_mef_set
    results = {}
    with tempfile.TemporaryDirectory() as parts:
        data = os.path.join(parts, 'data')
        write_mef_set(os.path.join(data, 'mef'))
        write_band_dirs(os.path.join(data, 'jpg'))
        for tool in TOOLS:
            for case in CASES:
                part = os.path.join(parts, f'{tool}_{case}.json')
                subprocess.run([sys.executable, os.path.abspath(__file__), '--case',
                                os.path.abspath(src), tool, case, data, part],
                               env=env, check=True, stdout=subprocess.DEVNULL)
                with open(part) as f:
                    results[f'{tool}_{case}'] = json.load(f)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f"{out_json}: {sum(len(v) for r in results.values() for v in r.values() if isinstance(v, dict))} "
          f"images fingerprinted across {len(results)} cases")


def compare(before_json, after_json):
    "Prints what differs, per case and key. Returns True when nothing does."
    before, after = json.load(open(before_json)), json.load(open(after_json))
    same = True
    for case in sorted(set(before) | set(after)):
        a, b = before.get(case, {}), after.get(case, {})
        differing = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        same &= not differing
        print(f"{case}: {'identical' if not differing else 'DIFFERS in ' + ', '.join(differing)}")
    return same


if __name__ == '__main__':
    if sys.argv[1:2] == ['--case']:
        _run_case(*sys.argv[2:7])
    elif sys.argv[1:2] == ['--compare']:
        sys.exit(0 if compare(*sys.argv[2:4]) else 1)
    else:
        snapshot(*sys.argv[1:3])
