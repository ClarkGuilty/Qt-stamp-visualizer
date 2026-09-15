# This Python file uses the following encoding: utf-8
import argparse

from astropy.io import fits

import glob

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from PIL import Image

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QPainter, QFont, QKeySequence, QShortcut

import os
from os.path import join

import sys
from time import time

from imaging import (
    identity, log, asinh2, get_value_range_asymmetric, clip_normalize,
    contrast_bias_scale, get_contrast_bias_reasonable_assumptions,
    natural_sort, detect_band_filetype, find_band_file,
    resolve_classifications_dir, ClassificationWriter,
    next_free_csv_path, NEW_DATASET_FORK,
)
import state
from widgets import (
    AlignDelegate, ClickableComboBox, LabelledIntField, NamedLabel,
    PanelOrderPicker,
)


parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="Path to the images to inspect.",
                    default="Color_stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default=None)
parser.add_argument('-b',"--main_band", help='High resolution band. Example: "VIS". This is also the '
                    "tool's default band: the panel that's always individually shown and pre-selected.",
                    default="VIS")
parser.add_argument('-B',"--color_bands", help='Comma-separated bands to make individually selectable as '
                    'their own panel (in addition to the RGB composites). Example: "Y,J,H"',
                    default="Y,J,H")
parser.add_argument('--rgb-composites',
                    help='RGB composites: semicolon-separated R,G,B band-name triples (any directory name '
                    "under --path can be a band -- not just VIS/Y/J/H/I). A composite's own name/label is "
                    'simply its comma-joined member list. Example: "H,Y,I;H,J,Y". '
                    'All three bands in a composite must have the exact same image dimensions '
                    '(and ideally the same zero-point) -- mismatched bands cannot be stacked into one RGB image.',
                    default="H,Y,I;H,J,Y")
parser.add_argument('-l',"--ncols","--gridsize", help="Number of columns per page. Find the optimal value before starting the classification. Once you start the classification do not change this.",type=int,
                    default=5)
parser.add_argument('-m',"--nrows", 
                    help="Number of rows per page. Find the optimal value before starting the classification. Once you start the classification do not change this.",type=int,
                    default=8)
parser.add_argument('-s',"--seed", help="Seed used to shuffle the images.",type=int,
                    default=None)
parser.add_argument("--minimum_size", help="Minimum size of the stamps in the mosaic. The default (66) should be good enough, but you can try smaller values if the mosaic is too big for your screen. You can change this even after you started a classification.",type=int,
                    default=None)
parser.add_argument("--printname",
                    help="Print the file name of every stamp you click (shown in the "
                    "lobby's log pane when mosaic is launched from it). Enabled by "
                    "default; use --no-printname to silence it.",
                    action=argparse.BooleanOptionalAction,
                    default=True)
parser.add_argument("--page", help="Initial page.",type=int,
                    default=None)
parser.add_argument('--resize',
                    help="Set to allow the resizing of the stamps with the window.",
                    action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument("--reset-config", help="Forgets the saved preferences (colormap, scale, "
                    "panels) during startup. Resume positions are kept; use --reset-position "
                    "for those.",
                    action="store_true", default=False)
parser.add_argument("--reset-position", help="Forgets the saved resume page for this "
                    "--path/--name/--seed, so the session restarts at the first page. "
                    "Preferences are kept.",
                    action="store_true", default=False)
parser.add_argument("--classifications-dir", metavar="PATH",
                    help="Directory for the autosaved classification CSVs. Absolute, or "
                    "relative to the directory you launch from; created if it does not "
                    "exist. Default: ./Classifications",
                    default=None)


args = parser.parse_args()
# Empty entries dropped: -B '' means "no color bands", not one band named ''
# (which passes the missing-directory check below -- it resolves to --path itself
# -- and then fails at image load).
args.color_bands = [b.strip() for b in args.color_bands.split(',') if b.strip()]

args.composite_bands = []  # ordered list of composite keys ("H,Y,I")
args.composite_band_members = {}  # key -> (r, g, b) tuple
for entry in args.rgb_composites.split(';'):
    entry = entry.strip()
    if not entry:
        continue
    members = tuple(b.strip() for b in entry.split(','))
    if len(members) != 3:
        parser.error(f'--rgb-composites entry "{entry}" must have exactly 3 comma-separated bands (R,G,B).')
    key = ','.join(members)
    args.composite_bands.append(key)
    args.composite_band_members[key] = members

_referenced_bands = ({args.main_band} | set(args.color_bands) |
                      {b for members in args.composite_band_members.values() for b in members})
_missing_bands = [b for b in _referenced_bands if not os.path.isdir(join(args.path, b))]
if _missing_bands:
    _available = sorted(d for d in os.listdir(args.path) if os.path.isdir(join(args.path, d))) \
        if os.path.isdir(args.path) else []
    print(f"Band director{'y' if len(_missing_bands) == 1 else 'ies'} not found under {args.path}: "
          f"{', '.join(sorted(_missing_bands))}")
    print(f"Available subdirectories: {', '.join(_available) if _available else '(none found)'}")
    sys.exit(1)

C_INTERESTING = 2
C_LENS = 1
C_UNINTERESTING = 0

SESSION_TOOL = 'mosaic'
# Pre-split config file. Read once, at the first startup after the split, then
# kept as a .bak -- see state.migrate_legacy_config.
LEGACY_CONFIG_FILE = '.config_mosaic.json'

if args.reset_config:
    if state.reset_preferences(SESSION_TOOL):
        print("Preferences reset.")
    if os.path.exists(LEGACY_CONFIG_FILE):
        # Otherwise the migration below would immediately restore what was just reset.
        os.replace(LEGACY_CONFIG_FILE, LEGACY_CONFIG_FILE + '.bak')

if args.reset_position:
    if state.forget_session(state.SessionId(SESSION_TOOL, args.path, args.name or '', args.seed)):
        print("Resume position reset.")

def _solid_pixmap(color, size=128):
    pixmap = QPixmap(size, size)
    pixmap.fill(color)
    return pixmap


def _letter_pixmap(letter, size=128):
    "Black tile with a bold white letter, used to mark a classified stamp."
    pixmap = _solid_pixmap(Qt.black, size)
    painter = QPainter(pixmap)
    painter.setPen(Qt.white)
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(int(size * 0.7))
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, letter)
    painter.end()
    return pixmap


def iloc_to_page_and_grid_pos(iloc, gridarea):
    return iloc // gridarea, iloc % gridarea



class MiniMosaicLabels(QtWidgets.QLabel):
    def __init__(self,
                aspectRatioPolicy,
                minimum_size,
                sizePolicy,
                name = None,
                parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.name = name
        self.aspectRatioPolicy = aspectRatioPolicy
        self.setMinimumSize(minimum_size,minimum_size)
        self.setSizePolicy(sizePolicy,sizePolicy)    
        self.setScaledContents(False)
        self.updateGeometry()

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
                            self.width(), self.height(),
                            self.aspectRatioPolicy
                            ))

class MiniMosaics(QtWidgets.QLabel):
    clicked = Signal(str)
    "Widget to hold the image Qlabels"
    def __init__(self,
                    filepaths,
                    bands, lens_background_pixmap,
                    interesting_background_pixmap, deactivated_pixmap, i,
                    status, activation, update_df_func,
                    image_width=None,
                    image_height=None,
                    parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.filepaths = filepaths
        self.bands = bands
        self.n_bands = len(self.bands)
        self.is_activate = activation
        self.lens_background_pixmap = lens_background_pixmap
        self.interesting_background_pixmap = interesting_background_pixmap

        self.mini_layout = QtWidgets.QHBoxLayout(self)
        self.mini_layout.setSpacing(0) #TODO: FIND A GOOD VALUE/RECIPE
        self.mini_layout.setContentsMargins(0,0,0,0)

        self.deactivated_pixmap = deactivated_pixmap
        self.is_a_candidate = status
        self.update_df_func = update_df_func
        self.i = i

        self.target_width = 66 #At the very least, should be the initial size
        self.target_height = 66 #
        self.user_minimum_size = 66 if args.minimum_size is None else args.minimum_size
        if image_width is not None:
            self.target_width = image_width
        if image_height is not None:
            self.target_height = image_height

        self.aspectRatioPolicy = Qt.KeepAspectRatio

        sizePolicy = QtWidgets.QSizePolicy.Ignored
        self.setMinimumSize(self.user_minimum_size*3,self.user_minimum_size)
        self.setSizePolicy(sizePolicy,sizePolicy)

        self.setScaledContents(args.resize)
        
        qlabelSizePolicy = QtWidgets.QSizePolicy.Ignored 

        self.qlabels = [MiniMosaicLabels(self.aspectRatioPolicy,
                                        self.user_minimum_size,
                                        qlabelSizePolicy,
                                        name = name,
                                        ) for name in self.bands]
        
        if self.is_activate:
            if self.is_a_candidate == C_UNINTERESTING:
                self.change_pixmaps(self.filepaths)
            elif self.is_a_candidate == C_LENS:
                self.change_pixmaps([self.lens_background_pixmap]*self.n_bands)
            elif self.is_a_candidate == C_INTERESTING:
                self.change_pixmaps([self.interesting_background_pixmap]*self.n_bands)
        else:
            self.change_pixmaps([self.deactivated_pixmap]*self.n_bands)

        for qlabel in self.qlabels:   
            qlabel.setPixmap(qlabel._pixmap.scaled(
                self.target_width, self.target_height,
                self.aspectRatioPolicy))
            self.mini_layout.addWidget(qlabel, 1)

        self.mini_layout.addStretch()

    def change_pixmaps(self, paths_to_pixmap):
        for qlabel, path_to_pixmap in zip(self.qlabels,paths_to_pixmap):
            qlabel._pixmap = QPixmap(path_to_pixmap)

    def activate(self):
        self.is_activate = True

    def change_filepath(self, filepaths):
        self.filepaths = filepaths
        self.change_pixmaps(self.filepaths)

    def change_and_paint_pixmap(self, filepaths):
        if self.is_activate:
            self.filepaths = filepaths
            self.change_pixmaps(self.filepaths)
            self.repaint_pixmaps()
            
    def repaint_pixmaps(self):
        for qlabel in self.qlabels:        
            qlabel.setPixmap(qlabel._pixmap.scaled( 
                qlabel.width(), qlabel.height(),
                self.aspectRatioPolicy))

    def deactivate(self):
        self.change_and_paint_pixmap([self.deactivated_pixmap] * self.n_bands)
        self.is_activate = False

    def set_candidate_status(self, status):
        if self.is_activate:
            self.is_a_candidate = status
            return True
        return False

    def toggle_candidate_status(self):
        if self.is_activate:
            self.is_a_candidate = not self.is_a_candidate


    def paint_pixmap(self):
        if self.is_activate:
            self.repaint_pixmaps()

    def paint_background_pixmap(self, background_pixmap):
        if self.is_activate:
            self.change_pixmaps([background_pixmap]*self.n_bands)
            self.repaint_pixmaps()


    def mousePressEvent(self, event):
        if self.is_activate:
            modifiers = event.modifiers()
            if self.is_a_candidate != C_UNINTERESTING:
                self.change_and_paint_pixmap(self.filepaths)
                new_class = C_UNINTERESTING
            else:
                if modifiers in [Qt.ControlModifier, Qt.ShiftModifier]:
                    self.paint_background_pixmap(self.interesting_background_pixmap)
                    new_class = C_INTERESTING
                elif modifiers == Qt.NoModifier:
                    self.paint_background_pixmap(self.lens_background_pixmap)
                    new_class = C_LENS

            self.update_df_func(event, self.i, new_class)
            self.is_a_candidate = new_class
        else:
            print('Inactive button')

    def reorder_panels(self, band_order):
        "Show exactly the panels in band_order (in self.bands' fixed relative order), hide the rest."
        for band, qlabel in zip(self.bands, self.qlabels):
            qlabel.setVisible(band in band_order)
        self.setMinimumSize(self.user_minimum_size*len(band_order),self.user_minimum_size)
        self.updateGeometry()



class MosaicVisualizer(QtWidgets.QMainWindow):
    def __init__(self, path_to_the_stamps= args.path):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')

        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = args.seed            
        self.stampspath = path_to_the_stamps

        self.main_band = args.main_band
        self.color_bands = args.color_bands

        self.scratchpath = './.temp'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.classifications_dir = resolve_classifications_dir(args.classifications_dir)
        os.makedirs(self.classifications_dir, exist_ok=True)
        self.clean_dir(self.scratchpath)

        base_band_path = join(self.stampspath, self.main_band)
        self.listimage = sorted(os.path.basename(x) for x in glob.glob(join(base_band_path, '*.fits')))
        self.filetype='FITS'
        if len(self.listimage) < 1:
            self.listimage = sorted(os.path.basename(x)
                            for x in (glob.glob(join(base_band_path, '*.png')) +
                                      glob.glob(join(base_band_path, '*.jpg')) +
                                      glob.glob(join(base_band_path, '*.jpeg'))))
            self.filetype='COMPRESSED'
        print(f"Classifying {len(self.listimage)} sources.")

        if len(self.listimage) < 1:
            print(f"No FITS, PNG, or JPG files found in {base_band_path}.")
            sys.exit(1)

        # Each band's format is a property of its own directory (detected independently),
        # so bands can mix FITS and PNG/JPG within the same session -- only the main band
        # (self.filetype) governs the object list.
        self.band_filetype = {self.main_band: self.filetype}
        for band in (set(self.color_bands) |
                     {b for members in args.composite_band_members.values() for b in members}):
            self.band_filetype.setdefault(band, detect_band_filetype(join(self.stampspath, band)))

        # RGB composites require all three member bands to be FITS -- drop any composite
        # that isn't, rather than crashing or silently mixing formats into one image.
        self.composite_bands = []
        self.composite_band_members = {}
        for key in args.composite_bands:
            members = args.composite_band_members[key]
            non_fits = [b for b in members if self.band_filetype.get(b) != 'FITS']
            if non_fits:
                print(f"RGB composite '{key}' skipped -- requires FITS bands, but "
                      f"{', '.join(non_fits)} {'is' if len(non_fits) == 1 else 'are'} not FITS.")
                continue
            self.composite_bands.append(key)
            self.composite_band_members[key] = members

        self.all_single_bands = ({self.main_band} | set(self.color_bands) |
                                {b for members in self.composite_band_members.values() for b in members})
        self.bands_to_plot = [self.main_band, *self.composite_bands, *self.color_bands]

        if self.random_seed is not None:
            # 99 is always changed to this number when sorting to mantain compatibility with old classifications.
            # 128 bits, proton decay might be more likely than someone *randomly* using this number.
            # Please, do not use this number as your seed.
            seed_to_use = 120552782132343758881253061212639178445 if self.random_seed == 99 else self.random_seed
            rng = np.random.default_rng(seed_to_use)
            rng.shuffle(self.listimage) #inplace shuffling
        
        if len(self.listimage) == 0:
            print("WARNING: no images found in {}".format(self.stampspath))
        self.ncols = args.ncols
        if args.nrows is None:
            self.nrows = self.ncols
        else:
            self.nrows = args.nrows
        self.gridarea = self.nrows*self.ncols
        self.PAGE_MAX = int(np.ceil(len(self.listimage) / self.gridarea))
        self.scale2funct = {'linear': identity,
                            'sqrt': np.sqrt,
                            'cbrt': np.cbrt,
                            'log': log,
                            'asinh': asinh2}

        self.cmname2cm = {
                            'gray':'gist_gray',
                            'gist_gray':'gist_gray',
                            'viridis':'viridis',
                            'yarg':'gist_yarg',
                            'gist_yarg':'gist_yarg',
                            'hot':'hot'
        }

        self.cm2cmname = {
                            'gist_gray':'gray',
                            'gray':'gray',
                            'viridis':'viridis',
                            'yarg':'yarg',
                            'gist_yarg':'yarg',
                            'hot':'hot',


        }

        title_strings = ["Mosaic stamp visualizer"]
        if args.name is not None:
            self.name = args.name
            title_strings.append(self.name)
        else:
            self.name = ''
        self.setWindowTitle(' - '.join(title_strings))

        # Preferences only: they follow the *user*, so they survive a change of
        # --path/--name/--seed. The resume page follows the *dataset* and lives
        # in the session store instead (see state.py). Neither nrows/ncols nor
        # the file count appear anywhere here: the page is re-derived from the
        # stored *filename*, so a reshape lands on whichever page now holds the
        # same object instead of being reset to 0.
        self.defaults = {
            'colormap': 'gist_gray',
            'scale': 'log',
            'panel_order':'',
        }
        self.session_id = state.SessionId(SESSION_TOOL, self.stampspath, self.name, self.random_seed)
        self.df = self.obtain_df()  # sets self.df_name
        # Every autosave goes through this: atomic, and never clobbers a file
        # another process has written since we last saved it.
        self.csv_writer = ClassificationWriter(self.df_name, index=False)

        state.migrate_legacy_config(LEGACY_CONFIG_FILE, self.session_id, self.defaults,
                                    self.listimage, self.legacy_page_index, csv=self.df_name)
        self.config_dict = state.load_preferences(SESSION_TOOL, self.defaults)
        if self.config_dict['scale'] == 'log10':  # renamed upstream long ago
            self.config_dict['scale'] = 'log'
        if self.config_dict['colormap'] == 'gray':
            self.config_dict['colormap'] = 'gist_gray'

        # resolve_position cross-checks the CSV: if obtain_df just resolved to a
        # different classification file than the one the position was recorded
        # against, the two disagree about which session is open and the position
        # is stale by definition.
        position = state.resolve_position(state.load_session(self.session_id),
                                          self.listimage, csv=self.df_name)
        self.page = min(position // self.gridarea, max(self.PAGE_MAX - 1, 0))
        if args.page is not None:
            self.page = max(min(args.page - 1, self.PAGE_MAX - 1), 0)

        if not self.config_dict['panel_order']:
            # Default-checked panels match today's exact view (main band + composites) -- the
            # individual color bands are available in the picker (self.bands_to_plot) but start
            # unchecked, so gaining the ability to show them standalone doesn't change what's
            # shown out of the box.
            default_panel_order = [self.main_band, *self.composite_bands]
            self.config_dict['panel_order'] = ';'.join(default_panel_order)

        self.interesting_background_pixmap = _letter_pixmap('I')
        self.lens_background_pixmap = _letter_pixmap('L')
        self.deactivated_pixmap = _solid_pixmap(Qt.black)
        self.status2background_dict = {C_LENS: self.lens_background_pixmap,
                                       C_INTERESTING: self.interesting_background_pixmap}

        self.bcounter = LabelledIntField('Page', self.page, self.PAGE_MAX)
        self.bcounter.setStyleSheet('background-color: black; color: gray')
        self.bcounter.lineEdit.returnPressed.connect(self.goto)
        self.bcounter.setInputText(self.page)

        self.buttons = []
        self.clean_dir(self.scratchpath)

        self.prepare_pngs(self.gridarea)

        main_layout = QtWidgets.QVBoxLayout(self._main)
        stamp_grid_layout = QtWidgets.QGridLayout()
        bottom_bar_layout = QtWidgets.QHBoxLayout()
        button_bar_layout = QtWidgets.QHBoxLayout()
        page_counter_layout = QtWidgets.QHBoxLayout()


        bottom_bar_layout.addLayout(button_bar_layout,10)
        bottom_bar_layout.addLayout(page_counter_layout,1)
        main_layout.addLayout(stamp_grid_layout, 8)
        main_layout.addLayout(bottom_bar_layout, )

        self.fontsize = 18
        
        #### Buttons
        self.cbscale = ClickableComboBox()
        delegate = AlignDelegate(self.cbscale)
        self.cbscale.setItemDelegate(delegate)
        self.cbscale.setFont(QFont("Arial",self.fontsize))
        self.cbscale.addItems(self.scale2funct.keys())
        self.cbscale.setCurrentIndex(list(self.scale2funct.keys()).index(self.config_dict['scale']))
        self.cbscale.setStyleSheet('background-color: gray')
        self.cbscale.currentIndexChanged.connect(self.change_scale)


        self.cbcolormap = ClickableComboBox()
        delegate = AlignDelegate(self.cbcolormap)
        self.cbcolormap.setItemDelegate(delegate)
        self.cbcolormap.setFont(QFont("Arial",self.fontsize))
        self.listscales = ['gray','viridis','yarg','hot']
        self.cbcolormap.addItems(self.listscales)
        self.cbcolormap.setCurrentIndex(self.listscales.index(self.cm2cmname[self.config_dict['colormap']]))
        self.cbcolormap.setStyleSheet('background-color: gray')
        self.cbcolormap.currentIndexChanged.connect(self.change_colormap)

        panel_labels = {band: band for band in self.bands_to_plot}
        self.panel_picker = PanelOrderPicker(self.bands_to_plot, panel_labels)
        self.panel_picker.button.setFont(QFont("Arial",self.fontsize))
        self.panel_picker.button.setStyleSheet('background-color: gray')
        for panel_key, checkbox in self.panel_picker.menu._checkboxes.items():
            checkbox.setChecked(panel_key in self.config_dict['panel_order'].split(';'))
        self.panel_picker.selectionChanged.connect(self.change_panel_order)


        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)
        self.bprev.setStyleSheet('background-color: gray')
        self.bprev.setFont(QFont("Arial",self.fontsize))

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)
        self.bnext.setStyleSheet('background-color: gray')
        self.bnext.setFont(QFont("Arial",self.fontsize))

        self.bclickcounter = NamedLabel('Clicks', (self.df['classification'] == C_LENS).sum().astype(int))
        self.bclickcounter.setStyleSheet('background-color: black; color: gray')

        ##### Keyboard shortcuts
        self.knext = QShortcut(QKeySequence('j'), self)
        self.knext.activated.connect(self.next)

        self.kprev = QShortcut(QKeySequence('k'), self)
        self.kprev.activated.connect(self.prev)

        self.knext = QShortcut(QKeySequence('f'), self)
        self.knext.activated.connect(self.next)

        self.kprev = QShortcut(QKeySequence('d'), self)
        self.kprev.activated.connect(self.prev)


        button_bar_layout.addWidget(self.cbscale)
        button_bar_layout.addWidget(self.cbcolormap)
        button_bar_layout.addWidget(self.panel_picker)
        button_bar_layout.addWidget(self.bprev)
        button_bar_layout.addWidget(self.bnext)
        page_counter_layout.addWidget(self.bclickcounter)
        page_counter_layout.addWidget(self.bcounter)

        self.total_n_frame = int(len(self.listimage)/(self.gridarea))
        start = self.page*self.gridarea

        for i in range(start,start+self.gridarea):
            try:
                classification = self.df.iloc[i,self.df.columns.get_loc('classification')]
                activation = True
            except IndexError:
                classification = False
                activation = False

            button = MiniMosaics(
                                    self.filepaths(i, self.page),
                                    self.bands_to_plot,
                                    self.lens_background_pixmap,
                                    self.interesting_background_pixmap,
                                    self.deactivated_pixmap,
                                    i-start, classification, activation,
                                    self.my_label_clicked,
                                    )
            button.reorder_panels(self.config_dict['panel_order'].split(';'))
            stamp_grid_layout.addWidget(
                button, i % self.nrows, i // self.nrows)
            self.buttons.append(button)
            button.setAlignment(Qt.AlignCenter) #TODO CHECK HOW TO REACTIVATE THIS. (OR IF IT'S NEEDED)
        if not any(self.band_filetype.get(b) == 'FITS' for b in self.all_single_bands):
            self.cbscale.setEnabled(False)
            self.cbcolormap.setEnabled(False)
        
        self.time_0 = time()

    def go_to_page(self, target_page):
        range_low = self.page*self.gridarea
        range_high = min(len(self.df),(self.page+1)*(self.gridarea))
        if hasattr(self, 'time_0'):
            self.df.iloc[range(range_low,range_high),
                        self.df.columns.get_loc('time')] += (time() - self.time_0)
            self.time_0 = time()
        else:
            True
        self.page = target_page
        self.clean_dir(self.scratchpath)
        self.update_grid()
        self.bcounter.setInputText(self.page)
        # Save the CSV first: if the writer has to fork to a new name, the
        # session entry must record the name we actually ended up writing.
        self.df_name = self.csv_writer.save(self.df)
        self.save_position()

    @Slot()
    def goto(self):
        # Pages are 0..PAGE_MAX-1: PAGE_MAX itself used to be accepted here and
        # then fell off the end of listimage, which update_grid's try/except
        # quietly turned into a gridful of blank buttons.
        if self.bcounter.getValue()>=self.PAGE_MAX:
            print("page: ",self.PAGE_MAX)
            self.status.showMessage('WARNING: There are only {} pages.'.format(
                self.PAGE_MAX),10000)
        elif self.bcounter.getValue()<0:
            self.status.showMessage('WARNING: Pages go from 0 to {}.'.format(
                self.PAGE_MAX-1),10000)
        else:
            self.go_to_page(self.bcounter.getValue())
            self.bcounter.lineEdit.clearFocus()
    @Slot()
    def next(self):
        if self.page+1 >= self.PAGE_MAX:
            self.status.showMessage('You are already at the last page',10000)
        else:
            self.go_to_page(self.page + 1)
    @Slot()
    def prev(self):
        if self.page -1 < 0:
            self.status.showMessage('You are already at the first page',10000)
        else:
            self.go_to_page(self.page - 1)

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()
        self.save_preferences()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid(single_band_only=True)
        self.save_preferences()

    def change_panel_order(self):
        self.config_dict['panel_order'] = ';'.join(self.panel_picker.selected_panels())
        self.update_grid(single_band_only=True,change_panel_order=True)
        self.save_preferences()

    def my_label_clicked(self, event, i, new_class):
        if self.page*self.gridarea+i > len(self.listimage):
            print('Something is wrong. This condition should not be trigger.')
        else:
            object_index = self.gridarea*self.page+i
            if args.printname:
                print(self.df.iloc[object_index,
                            self.df.columns.get_loc('file_name')])
            self.df.iloc[object_index,
                        self.df.columns.get_loc('classification')] = new_class
            
            range_low = self.page*self.gridarea
            range_high = min(len(self.df),(self.page+1)*(self.gridarea))
            if hasattr(self, 'time_0'):
                self.df.iloc[range(range_low,range_high),
                        self.df.columns.get_loc('time')] += (time() - self.time_0)
                self.time_0 = time()

            self.bclickcounter.setText((self.df['classification'] == C_LENS).sum().astype(int))
            self.df_name = self.csv_writer.save(self.df)

    def filepath(self, i, page, band = ''):
        colormap = self.config_dict['colormap'] if band == '' else ''
        return join(self.scratchpath, (str(i+1)+self.config_dict['scale']+
                                       colormap+
                                       str(page)+
                                       str(band)+
                                       '.png')
                                        )

    def filepaths(self, i, page,nvisiblebands = None):
        if nvisiblebands is None:
            nvisiblebands = len(self.bands_to_plot)
        return [self.filepath(i,page,band) 
                    for band in self.bands_to_plot[:nvisiblebands]]

    def save_preferences(self):
        state.save_preferences(SESSION_TOOL, self.config_dict)

    def save_position(self):
        """Persist the page as the filename of the object in its top-left cell.

        Storing the filename rather than the page number is what lets the grid
        shape stay out of the session identity: a reshaped grid re-derives its
        page from wherever that object now falls.
        """
        index = min(self.page * self.gridarea, len(self.listimage) - 1)
        state.save_session(self.session_id,
                           position_filename=self.listimage[index],
                           csv=self.df_name)

    def legacy_page_index(self, legacy):
        """The index the pre-split `.config_mosaic.json` was pointing at, or None.

        The old file stored a page number, valid only for the grid shape it was
        saved under -- so both that shape and `--name` have to match before its
        page can be turned into an index.
        """
        if (legacy.get('name') or '') != self.name:
            return None
        if legacy.get('nrows') != self.nrows or legacy.get('ncols') != self.ncols:
            return None
        page = legacy.get('page')
        if not isinstance(page, int) or isinstance(page, bool):
            return None
        return page * self.gridarea

    def obtain_df(self):
        """The classification CSV for this session, read or created.

        The filename carries the dataset identity -- name, image count and seed
        -- and deliberately *not* the grid shape. The shape is a display choice:
        how many stamps you can comfortably fit on screen, changeable mid-session
        (and `--minimum_size` exists for exactly that). It says nothing about
        which objects are being classified, so it has no business forking the
        file that records the grades. The `page`/`grid_pos` columns, the only
        shape-dependent thing in here, are recomputed on load below.

        The naming now matches `single_viewer.py`'s, which never had a grid to
        encode: `-` introduces an iteration suffix and `_` introduces the seed,
        so the two never collide in a glob.
        """
        base_prefix = 'classification_mosaic_autosave_{}_{}'.format(
                                self.name, len(self.listimage))
        if self.random_seed is None:
            base_filename = base_prefix
            string_to_glob = join(self.classifications_dir, '{}-*.csv'.format(base_filename))
            string_to_glob_for_files_with_seed = join(self.classifications_dir,
                                                      '{}_*.csv'.format(base_filename))
            glob_results = (set(glob.glob(string_to_glob))
                            - set(glob.glob(string_to_glob_for_files_with_seed))
                            | set(glob.glob(join(self.classifications_dir,
                                                 '{}.csv'.format(base_filename)))))
        else:
            base_filename = '{}_{}'.format(base_prefix, self.random_seed)
            string_to_glob = join(self.classifications_dir, '{}*.csv'.format(base_filename))
            glob_results = set(glob.glob(string_to_glob))

        class_file = np.array(natural_sort(glob_results)) #better to use natural sort.
        fresh_name = join(self.classifications_dir, '{}.csv'.format(base_filename))
        new_dataset_name = ''
        if len(class_file) >= 1:
            file_index = 0
            if len(class_file) > 1:
                file_index = -2
            self.df_name = class_file[file_index]
            print('Reading '+ self.df_name)
            df = pd.read_csv(self.df_name)
            if np.all(self.listimage == df['file_name'].values):
                if 'time' not in df.keys():
                    df['time'] = 0
                # `page`/`grid_pos` record where a stamp sat in the grid, so they are
                # only meaningful for the shape they were written under. The unseeded
                # filename does not carry `nrows`, so this same CSV is legitimately
                # reopened after an `--nrows` change -- recompute both from the
                # current grid instead of trusting the old shape's values, which
                # otherwise survive on every page the user doesn't happen to visit
                # and leave the file describing two grids at once.
                page, grid_pos = iloc_to_page_and_grid_pos(np.array(df.index),
                                                           gridarea=self.gridarea)
                df['page'] = page
                df['grid_pos'] = grid_pos
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                if os.path.exists(fresh_name):
                    new_dataset_name = next_free_csv_path(fresh_name, NEW_DATASET_FORK)

        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        self.df_name = new_dataset_name or fresh_name
        print('A new csv will be created', self.df_name)

        if new_dataset_name:
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        df = pd.DataFrame(columns=self.dfc)
        df['file_name'] = self.listimage
        df['classification'] = np.zeros(np.shape(self.listimage))
        page,grid_pos = iloc_to_page_and_grid_pos(np.array(df.index) ,gridarea = self.gridarea)
        df['page'] = page
        df['grid_pos'] = grid_pos
        df['time'] = np.zeros(np.shape(self.listimage))
        return df

    def update_grid(self, single_band_only = False, change_panel_order=False):
        start = self.page*self.gridarea
        n_images = self.gridarea
        self.prepare_pngs(n_images, single_band_only)
        i = start
        j = 0
        for button in self.buttons:
            try:
                object_index = self.gridarea*self.page+j
                status = self.df.iloc[object_index,self.df.columns.get_loc('classification')]
                if status == 0:
                    button.activate()
                    button.change_and_paint_pixmap(self.filepaths(i,self.page))
                    button.set_candidate_status(status)
                else:
                    button.activate()
                    button.change_filepath(self.filepaths(i,self.page))
                    button.paint_background_pixmap(self.status2background_dict[status])
                    button.set_candidate_status(status)

                if change_panel_order:
                    button.reorder_panels(self.config_dict['panel_order'].split(';'))
                self.df.iloc[object_index,
                             self.df.columns.get_loc('grid_pos')] = j

                self.df.iloc[object_index,
                             self.df.columns.get_loc('page')] = self.page

            except (KeyError,IndexError):
                button.deactivate()
            j = j+1
            i = i+1

    def prepare_pngs(self, number, single_band_only = False):
            "Generates the png files from the fits."
            start = self.page*self.gridarea
            for i in np.arange(start, start + number + 0): 
                if i < len(self.listimage):
                    self.prepare_png(i, single_band_only)
                else:
                    image = np.zeros((66, 66))
                    plt.imsave(self.filepath(i, self.page),
                        image, cmap=self.cmname2cm[self.config_dict['colormap']], origin="lower")

    def _band_filepath(self, i, band):
        "Resolves band's file for object i -- matched by stem so bands with different formats/extensions for the same object still line up."
        if band == self.main_band:
            return join(self.stampspath, band, self.listimage[i])
        stem = os.path.splitext(self.listimage[i])[0]
        filepath = find_band_file(self.stampspath, band, stem)
        if filepath is None:
            raise FileNotFoundError(f"No FITS/PNG/JPG file found for '{stem}' in band '{band}'.")
        return filepath

    def prepare_png(self, i, single_band_only):
        # self.composite_bands only ever contains composites whose 3 members are all FITS
        # bands (filtered at startup), so no per-composite format branching is needed below.
        fits_bands = [band for band in self.all_single_bands if self.band_filetype.get(band) == 'FITS']
        band_images = {band: self.read_fits(self._band_filepath(i, band)) for band in fits_bands}

        for band in [self.main_band, *self.color_bands]:
            if self.band_filetype.get(band) == 'FITS':
                image = self.prepare_single_band(band_images[band])
                plt.imsave(self.filepath(i, self.page, band=band),
                        image, cmap=self.cmname2cm[self.config_dict['colormap']], origin="lower")
            else:
                src = self._band_filepath(i, band)
                Image.open(src).save(self.filepath(i, self.page, band=band))

        if not single_band_only:
            for composite_band in self.composite_bands:
                bands = self.composite_band_members[composite_band]
                try:
                    stacked = np.stack([band_images[band] for band in bands],axis=-1)
                except ValueError:
                    shapes = {band: band_images[band].shape for band in bands}
                    raise ValueError(
                        f"RGB composite '{composite_band}' requires all member bands to have the exact "
                        f"same image dimensions -- got {shapes}")
                composite_image = self.prepare_composite_band(stacked)
                plt.imsave(self.filepath(i, self.page, band=composite_band),
                    composite_image, origin="lower")

    def prepare_single_band(self, image):
        scale_min, scale_max = self.scale_val(image)
        image = self.rescale_image(image, scale_min, scale_max)
        image[np.isnan(image)] = np.nanmin(image)
        return image
            

    def prepare_composite_band(self,images,
                                p_low=1, p_high=0.1,
                                value_at_min=0,
                                color_bkg_level=0.015,):
        
        composite_image = np.zeros_like(images,
                                    dtype=float)
        scale_min, scale_max = get_value_range_asymmetric(images,p_low,p_high)
        
        for i in range(images.shape[-1]):
            composite_image[:,:,i] = self.rescale_single_band(
                            images[:,:,i],
                            scale_min,
                            scale_max,
                            value_at_min,
                            color_bkg_level)
        return composite_image

    def rescale_single_band(self, image,
                            scale_min,
                            scale_max,
                            value_at_min=0,
                            color_bkg_level=-0.05):

        image = clip_normalize(image,scale_min,scale_max)
        image = self.scale2funct[self.config_dict['scale']](image)
        contrast, bias = get_contrast_bias_reasonable_assumptions(
                                                                    max(value_at_min,scale_min),
                                                                    color_bkg_level,
                                                                    scale_min,
                                                                    scale_max,
                                                                    self.scale2funct[self.config_dict['scale']])
        return contrast_bias_scale(image, contrast, bias)



    def clean_dir(self, path_dir):
        "Removes everything in the scratch folder."
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))

    def read_fits(self, filepath):
        # Note : memmap=False is much faster when opening/closing many small files
        with fits.open(filepath, memmap=False) as hdu_list:
            image = hdu_list[0].data
        return image

    def rescale_image(self, image, scale_min, scale_max):
        factor = self.scale2funct[self.config_dict['scale']](scale_max - scale_min)
        image = image.clip(min=scale_min, max=scale_max)

        # Clipping to [scale_min, scale_max] before scaling is what keeps bright noise from
        # washing the stamp out.
        indices0 = np.where(image < scale_min)
        indices1 = np.where((image >= scale_min) & (image < scale_max))
        indices2 = np.where(image >= scale_max)
        image[indices0] = 0.0
        image[indices2] = 1.0
        image[indices1] = self.scale2funct[self.config_dict['scale']](image[indices1]) / ((factor) * 1.0)
        return image

    def scale_val(self, image_array):
        if image_array.shape[0] > 173:
            box_size_vmin = np.round(np.sqrt(np.prod(image_array.shape) * 0.001)).astype(int)
            box_size_vmax = np.round(np.sqrt(np.prod(image_array.shape) * 0.01)).astype(int)
        else:
            box_size_vmin = 5
            box_size_vmax = 14
        vmin = np.nanmin(self.background_rms_image(box_size_vmin, image_array))
        if vmin == 0:
            vmin += 1e-3              
        
        xl, yl = np.shape(image_array)
        xmin = int((xl) / 2. - (box_size_vmax / 2.))
        xmax = int((xl) / 2. + (box_size_vmax / 2.))
        ymin = int((yl) / 2. - (box_size_vmax / 2.))
        ymax = int((yl) / 2. + (box_size_vmax / 2.))
        vmax = np.nanmax(image_array[xmin:xmax, ymin:ymax])
        return vmin*1.0, vmax*1.3 #vmin is 1 sigma.

    def background_rms_image(self, cb, image):
        xg, yg = np.shape(image)
        cb=10
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        while len(l) > 1:
            m = np.nanmean(np.nanmean(l, axis=1), axis=1)
            if max(m) > 5 * min(m):
                s = np.sort(l, axis=0)
                l = s[:-1]
            else:
                std = np.nanstd(l)
                return std
        std = np.nanstd(l)
        return std


def main():
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = MosaicVisualizer()
    app.show()
    app.activateWindow()
    qapp.exec()


if __name__ == "__main__":
    main()