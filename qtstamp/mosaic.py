# This Python file uses the following encoding: utf-8
import argparse

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QFontMetrics, QKeySequence, QShortcut

import os
from os.path import join

import sys
from time import time

from . import cli
from .dataset import DatasetError, load_dataset
from . import fits_io
from .imaging import (
    SCALES, MOSAIC_DISPLAY, scale_limits, rescale_image, composite_image,
    open_rendered_image,
    ClassificationWriter,
    next_free_csv_path, NEW_DATASET_FORK,
    MOSAIC_CSV_PREFIX, classification_csv_base, classification_csvs, resume_csv,
)
from . import paths
from . import state
from .widgets import (
    AlignDelegate, ClickableComboBox, LabelledIntField, MissingBandsNote, NamedLabel,
    PanelOrderPicker,
)


C_INTERESTING = 2
C_LENS = 1
C_UNINTERESTING = 0

SESSION_TOOL = 'mosaic'
# Pre-split config file. Read once, at the first startup after the split, then
# kept as a .bak -- see state.migrate_legacy_config.
LEGACY_CONFIG_FILE = '.config_mosaic.json'


def parse_args(argv=None):
    "The mosaic's arguments, normalised (see cli.finish_dataset_args). `argv` defaults to sys.argv."
    parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
    cli.add_dataset_args(parser)
    parser.add_argument('-l',"--ncols","--gridsize", help="Number of columns per page. Find the optimal value before starting the classification. Once you start the classification do not change this.",type=int,
                        default=5)
    parser.add_argument('-m',"--nrows",
                        help="Number of rows per page. Find the optimal value before starting the classification. Once you start the classification do not change this.",type=int,
                        default=8)
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
    args = parser.parse_args(argv)
    cli.finish_dataset_args(parser, args)
    return args


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


def _missing_band_pixmap(bands, size=128):
    "Dark grey tile naming the band(s) an object has no data for (BUGS 36)."
    pixmap = _solid_pixmap(QColor(48, 48, 48), size)
    painter = QPainter(pixmap)
    painter.setPen(Qt.lightGray)
    font = painter.font()
    pixel_size = int(size * 0.2)
    font.setPixelSize(pixel_size)
    widest = max(bands, key=len)
    while pixel_size > 6 and QFontMetrics(font).horizontalAdvance(widest) > size * 0.9:
        pixel_size -= 1
        font.setPixelSize(pixel_size)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, '\n'.join(['no', *bands]))
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
                    minimum_size=None,
                    resize=False,
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
        self.user_minimum_size = 66 if minimum_size is None else minimum_size
        if image_width is not None:
            self.target_width = image_width
        if image_height is not None:
            self.target_height = image_height

        self.aspectRatioPolicy = Qt.KeepAspectRatio

        sizePolicy = QtWidgets.QSizePolicy.Ignored
        self.setMinimumSize(self.user_minimum_size*3,self.user_minimum_size)
        self.setSizePolicy(sizePolicy,sizePolicy)

        self.setScaledContents(resize)
        
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
                # A click on an already-graded stamp clears it, whatever the
                # modifier: there is only one way back to unclassified.
                self.change_and_paint_pixmap(self.filepaths)
                new_class = C_UNINTERESTING
            else:
                if modifiers in [Qt.ControlModifier, Qt.ShiftModifier]:
                    self.paint_background_pixmap(self.interesting_background_pixmap)
                    new_class = C_INTERESTING
                elif modifiers == Qt.NoModifier:
                    self.paint_background_pixmap(self.lens_background_pixmap)
                    new_class = C_LENS
                else:
                    # Any other modifier (Ctrl+Shift, Alt, Meta, and the
                    # combinations) is not a grade. `modifiers` is compared for
                    # equality, not tested bitwise, so these reach here rather
                    # than any branch above -- leave the stamp untouched instead
                    # of falling through with `new_class` unbound.
                    return

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
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')

        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = args.seed            
        self.stampspath = args.path

        self.main_band = args.main_band
        self.color_bands = args.color_bands

        # Scratch renders live with the workspace (2b-v), under the state dir's
        # cache/. `paths.cache_dir` has already made it and confirmed it is
        # writable -- falling back to the per-user cache dir if not, since a
        # mosaic with nowhere to render cannot run at all. `scratch_dir` then
        # carves out a per-process subdirectory, so two mosaics sharing one
        # workspace no longer wipe each other's tiles.
        cache_root = paths.cache_dir(override=args.state_dir_override)
        # Before `scratch_dir`, which sweeps the root: the legacy dir lands there
        # as loose tiles and is swept on this same run, which is what used to
        # happen to it anyway (`clean_dir` wiped it immediately). The point of the
        # migration is only to get `.temp` out of the CWD.
        paths.migrate_once(os.path.join(os.curdir, '.temp'), os.path.join(cache_root, 'temp'))
        self.scratchpath = paths.scratch_dir(cache_root)
        self.classifications_dir = state.open_classifications_dir(args.classifications_dir)
        self.clean_dir(self.scratchpath)

        try:
            dataset = load_dataset(self.stampspath, self.main_band, self.color_bands,
                                   args.composite_bands, args.composite_band_members,
                                   args.mef, self.random_seed)
        except DatasetError as e:
            print(e)
            sys.exit(1)
        self.listimage = dataset.listimage
        self.filetype = dataset.filetype
        self.band_filetype = dataset.band_filetype
        self.composite_bands = dataset.composite_bands
        self.composite_band_members = dataset.composite_band_members
        self.all_single_bands = dataset.all_single_bands
        self.band_sources = dataset.band_sources
        self.bands_to_plot = [self.main_band, *self.composite_bands, *self.color_bands]

        self.ncols = args.ncols
        if args.nrows is None:
            self.nrows = self.ncols
        else:
            self.nrows = args.nrows
        self.gridarea = self.nrows*self.ncols
        self.PAGE_MAX = int(np.ceil(len(self.listimage) / self.gridarea))
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

        self.config_dict, position = state.load_viewer_state(
            self.session_id, self.defaults, self.listimage, LEGACY_CONFIG_FILE,
            self.legacy_page_index, csv=self.df_name,
            classifications_dir=self.classifications_dir, config_dir=args.state_dir_override)

        self.page = min(position // self.gridarea, max(self.PAGE_MAX - 1, 0))
        if args.page is not None:
            self.page = max(min(args.page - 1, self.PAGE_MAX - 1), 0)

        # The panel order is saved per tool, not per dataset: one naming only bands this
        # dataset lacks (VIS;H,Y,I in a PNG session) would hide every tile, so it counts as unset.
        saved_panel_order = [b for b in self.config_dict['panel_order'].split(';') if b in self.bands_to_plot]
        self.config_dict['panel_order'] = ';'.join(saved_panel_order)
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

        # Names this page's objects that lack a band, where the page messages go.
        self.missing_bands_note = MissingBandsNote()
        self.status.addWidget(self.missing_bands_note)

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
        self.cbscale.addItems(SCALES.keys())
        self.cbscale.setCurrentIndex(list(SCALES).index(self.config_dict['scale']))
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
                                    minimum_size=args.minimum_size,
                                    resize=args.resize,
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
        # Two numbering schemes meet here. `self.page` is 0-based, 0..PAGE_MAX-1:
        # PAGE_MAX itself used to be accepted and then fell off the end of
        # listimage, which update_grid's try/except quietly turned into a gridful
        # of blank buttons. The widget is 1-based -- it reads "Page 3 / 12" --
        # and getValue() subtracts the 1, so every bound *shown* to the user is
        # 1..PAGE_MAX while every bound tested here is 0..PAGE_MAX-1.
        #
        # QIntValidator(1, PAGE_MAX) does not make this check redundant: a
        # QLineEdit only refuses input the validator calls Invalid, and "0", an
        # out-of-range "4" and an empty field are all Intermediate, so they stay
        # in the field and arrive here. An empty field is why getValue() is
        # guarded -- int('') raises.
        out_of_range = 'WARNING: Pages go from 1 to {}.'.format(self.PAGE_MAX)
        try:
            page = self.bcounter.getValue()
        except ValueError:
            self.status.showMessage(out_of_range,10000)
            self.bcounter.setInputText(self.page)  # field is blank; put the number back
            return
        if page>=self.PAGE_MAX:
            self.status.showMessage('WARNING: There are only {} pages.'.format(
                self.PAGE_MAX),10000)
        elif page<0:
            self.status.showMessage(out_of_range,10000)
        else:
            self.go_to_page(page)
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
            if self.args.printname:
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
        state.save_preferences(SESSION_TOOL, self.config_dict,
                               config_dir=self.args.state_dir_override)

    def save_position(self):
        """Persist the page as the filename of the object in its top-left cell.

        Storing the filename rather than the page number is what lets the grid
        shape stay out of the session identity: a reshaped grid re-derives its
        page from wherever that object now falls.
        """
        index = min(self.page * self.gridarea, len(self.listimage) - 1)
        state.save_session(self.session_id, self.classifications_dir,
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
        base_filename = classification_csv_base(MOSAIC_CSV_PREFIX, self.name,
                                                len(self.listimage), self.random_seed)
        fresh_name = join(self.classifications_dir, '{}.csv'.format(base_filename))
        new_dataset_name = ''
        existing = resume_csv(classification_csvs(self.classifications_dir, base_filename),
                              self.listimage)
        if existing is not None:
            self.df_name = existing
            print('Reading '+ self.df_name)
            df = pd.read_csv(self.df_name)
            if (len(self.listimage) == len(df) and
                np.all(self.listimage == df['file_name'].values)):
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
            missing_by_object = {}
            for i in np.arange(start, start + number + 0):
                if i < len(self.listimage):
                    missing = self.prepare_png(i, single_band_only)
                    if missing:
                        missing_by_object[os.path.splitext(self.listimage[i])[0]] = missing
                else:
                    image = np.zeros((66, 66))
                    plt.imsave(self.filepath(i, self.page),
                        image, cmap=self.cmname2cm[self.config_dict['colormap']], origin="lower")
            self.show_missing_bands(missing_by_object)

    def show_missing_bands(self, missing_by_object):
        "{stem: {band: MissingBandError}} for this page, into the status-bar note (empty if none)."
        text = '; '.join(f"{stem}: {fits_io.missing_bands_note(missing)}"
                         for stem, missing in missing_by_object.items())
        self.missing_bands_note.set_note(text)

    def _save_missing_band_png(self, bands, path):
        "Placeholder tile for a panel with no data (BUGS 36), where its PNG would have gone."
        _missing_band_pixmap(bands).save(path, 'PNG')

    def prepare_png(self, i, single_band_only):
        """Renders object i's panels. Returns {band: MissingBandError} for every band it
        has no data for -- those panels, and any composite needing one, get a placeholder."""
        # self.composite_bands only ever contains composites whose 3 members are all FITS
        # bands (filtered at startup), so no per-composite format branching is needed below.
        fits_bands = [band for band in self.all_single_bands if self.band_filetype.get(band) == 'FITS']
        stem = os.path.splitext(self.listimage[i])[0]
        band_images, headers, missing = {}, {}, {}
        for band in fits_bands:
            try:
                locator = fits_io.locate(self.band_sources, band, stem)
                band_images[band], headers[band] = fits_io.read_pixels_and_header(locator)
            except fits_io.MissingBandError as e:
                missing[band] = e

        for band in [self.main_band, *self.color_bands]:
            out = self.filepath(i, self.page, band=band)
            if band in missing:
                self._save_missing_band_png([band], out)
            elif self.band_filetype.get(band) == 'FITS':
                pixel_scale = fits_io.pixel_scale_arcsec(headers[band])
                image = self.prepare_single_band(band_images[band], pixel_scale)
                plt.imsave(out, image, cmap=self.cmname2cm[self.config_dict['colormap']], origin="lower")
            else:
                try:
                    src = fits_io.locate(self.band_sources, band, stem).filepath
                except fits_io.MissingBandError as e:
                    missing[band] = e
                    self._save_missing_band_png([band], out)
                    continue
                open_rendered_image(src).save(out)

        if not single_band_only:
            for composite_band in self.composite_bands:
                bands = self.composite_band_members[composite_band]
                absent = [band for band in bands if band in missing]
                if absent:
                    self._save_missing_band_png(absent, self.filepath(i, self.page, band=composite_band))
                    continue
                try:
                    stacked = np.stack([band_images[band] for band in bands],axis=-1)
                except ValueError:
                    shapes = {band: band_images[band].shape for band in bands}
                    raise ValueError(
                        f"RGB composite '{composite_band}' requires all member bands to have the exact "
                        f"same image dimensions -- got {shapes}")
                plt.imsave(self.filepath(i, self.page, band=composite_band),
                    composite_image(stacked, SCALES[self.config_dict['scale']], MOSAIC_DISPLAY),
                    origin="lower")
        return missing

    def prepare_single_band(self, image, pixel_scale_arcsec=None):
        scale_min, scale_max = scale_limits(image, MOSAIC_DISPLAY, pixel_scale_arcsec)
        image = rescale_image(image, scale_min, scale_max, SCALES[self.config_dict['scale']],
                              MOSAIC_DISPLAY)
        image[np.isnan(image)] = np.nanmin(image)
        return image

    def clean_dir(self, path_dir):
        "Removes everything in the scratch folder."
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))


def main(argv=None):
    args = parse_args(argv)
    cli.apply_reset_flags(args, SESSION_TOOL, LEGACY_CONFIG_FILE)

    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = MosaicVisualizer(args)
    app.show()
    app.activateWindow()
    qapp.exec()


if __name__ == "__main__":
    main()