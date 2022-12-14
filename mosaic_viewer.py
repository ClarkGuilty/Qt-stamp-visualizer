# This Python file uses the following encoding: utf-8
import sys
#from PySide6.QtWidgets import QApplication, QWidget, QPushButton
import PySide6

#import time
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import glob
import os
import pandas as pd
import subprocess
from PIL import Image

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QObject, QThread, Signal, QEvent
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut, QIntValidator

# from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg
import matplotlib.pyplot as plt


import urllib

from functools import partial

import webbrowser
import json
from os.path import join


def identity(x):
    return x


def asinh2(x):
    return np.arcsinh(x/2)

class LabelledIntField(QtWidgets.QWidget):
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value=None):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        self.label = QtWidgets.QLabel()
        self.label.setText(title)
        self.label.setFixedWidth(100)
        self.label.setFont(QFont("Arial",weight=QFont.Bold))
        layout.addWidget(self.label)
        
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value != None:
            self.lineEdit.setText(str(initial_value))
        layout.addWidget(self.lineEdit)
        layout.addStretch()
        
    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)
        
    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)
        
    def getValue(self):
        return int(self.lineEdit.text())


def clickable(widget):
        "https://wiki.python.org/moin/PyQt/Making%20non-clickable%20widgets%20clickable"
        class Filter(QObject):
            clicked = Signal()
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QEvent.MouseButtonRelease:
                        if obj.rect().contains(event.pos()):
                            self.clicked.emit()
                            # The developer can opt for .emit(obj) to get the object within the slot.
                            return True
                return False
        filter = Filter(widget)
        widget.installEventFilter(filter)
        return filter.clicked

class AlignDelegate(QtWidgets.QStyledItemDelegate):
    "https://stackoverflow.com/a/54262963/10555034"
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

class ClickableLabel(QtWidgets.QLabel):
    def __init__(self, filepath, backgroundpath, deactivatedpath, i, status, activation, update_df_func, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.filepath = filepath
        self.is_activate = activation
        # self._whenClicked = whenClicked
        self.backgroundpath = backgroundpath
        self.deactivatedpath = deactivatedpath
        self.is_a_candidate = status
        self.update_df_func = update_df_func
        self.i = i
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.MinimumExpanding)
        # self.setScaledContents(True)
        if self.is_activate:
            if self.is_a_candidate:
                self._pixmap = QPixmap(self.backgroundpath)
            else:
                self._pixmap = QPixmap(self.filepath)
        else:
            self._pixmap = QPixmap(self.deactivatedpath)
        self.setPixmap(self._pixmap)

    def activate(self):
        self.is_activate = True

    def deactivate(self):
        self.change_and_paint_pixmap(self.deactivatedpath)
        self.is_activate = False

    def set_candidate_status(self, status):
        if self.is_activate:
            self.is_a_candidate = status
            return True
        return False

    def toggle_candidate_status(self):
        if self.is_activate:
            self.is_a_candidate = not self.is_a_candidate

    def change_and_paint_pixmap(self, filepath):
        if self.is_activate:
            self.filepath = filepath
            self._pixmap = QPixmap(self.filepath)
            # print(self.filepath)
            # if self.is_activate:
            self.setPixmap(self._pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio))

    def paint_pixmap(self):
        if self.is_activate:
            self.setPixmap(self._pixmap)

    def paint_background_pixmap(self):
        if self.is_activate:
            self._pixmap = QPixmap(self.backgroundpath)
            self.setPixmap(self._pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio))

    def change_pixmap(self, filepath):
        self.filepath = filepath
        self._pixmap = QPixmap(self.filepath)

    def mousePressEvent(self, event):
        print(self.is_activate)
        if self.is_activate:
            self.is_a_candidate = not self.is_a_candidate
            if self.is_a_candidate:
                self.paint_background_pixmap()
                # self.setPixmap(QPixmap(self.backgroundpath).scaled(
                #     self.width(), self.height(),
                #     Qt.KeepAspectRatio))
            else:
                # self.setPixmap(QPixmap(self.filepath).scaled(
                #     self.width(), self.height(),
                #     Qt.KeepAspectRatio))
                self.change_and_paint_pixmap(self.filepath)

            self.update_df_func(event, self.i)
        else:
            print('Inactive button')


    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            # 66, 66,
            self.width(), self.height(),
            Qt.KeepAspectRatio))



class MosaicVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')

        self.setWindowTitle("Mosaic Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = 99

        self.stampspath = './Stamps_to_inspect/'
        self.scratchpath = './.temp'
        self.deactivatedpath = './dark.png'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.clean_scratch(self.scratchpath)
        self.listimage = sorted([os.path.basename(x)
                                for x in glob.glob(self.stampspath + '*.fits')])

        self.gridsize = 10
        self.gridarea = self.gridsize**2
        self.PAGE_MAX = int(np.floor(len(self.listimage) / self.gridarea))

        self.scale2funct = {'linear': identity,
                            'sqrt': np.sqrt, 'log10': np.log10, 'asinh': asinh2}

        self.defaults = {
            # 'counter': 0,
            'page': 0,
            'colormap': 'gray',
            'scale': 'log10',
        }
        self.config_dict = self.load_dict()
        self.scale = self.scale2funct[self.config_dict['scale']]
        self.path_background = '.background.png'
        self.deactivatedpath = '.backgrounddark.png'

        main_layout = QtWidgets.QVBoxLayout(self._main)
        stamp_grid_layout = QtWidgets.QGridLayout()
        button_bar_layout = QtWidgets.QHBoxLayout()

        main_layout.addLayout(stamp_grid_layout, 8)
        main_layout.addLayout(button_bar_layout, )

        #### Buttons
        self.cbscale = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbscale)
        self.cbscale.setItemDelegate(delegate)
        # self.cbscale.setEditable(True)
        self.cbscale.setFont(QFont("Arial",20))
        clickable(self.cbscale).connect(self.cbscale.showPopup)
        line_edit = self.cbscale.lineEdit()
        self.cbscale.addItems(['linear','sqrt', 'log10', 'asinh'])
        self.cbscale.setStyleSheet('background-color: gray')
        self.cbscale.currentIndexChanged.connect(self.change_scale)


        self.cbcolormap = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbcolormap)
        self.cbcolormap.setItemDelegate(delegate)
        # self.cbcolormap.setEditable(True)
        self.cbcolormap.setFont(QFont("Arial",20))
        line_edit = self.cbcolormap.lineEdit()
        # line_edit.setAlignment(Qt.AlignCenter)
        # line_edit.setReadOnly(True)
        self.listscales = ['gray','viridis','gist_yarg','hot']
        self.cbcolormap.addItems(self.listscales)
        self.cbcolormap.setCurrentIndex(self.listscales.index(self.config_dict['colormap']))
        self.cbcolormap.setStyleSheet('background-color: gray')
        self.cbcolormap.currentIndexChanged.connect(self.change_colormap)


        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)
        self.bprev.setStyleSheet('background-color: gray')
        self.bprev.setFont(QFont("Arial",20))

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)
        self.bnext.setStyleSheet('background-color: gray')
        self.bnext.setFont(QFont("Arial",20))

        self.bcounter = LabelledIntField('asd', 4)
        self.bcounter.setStyleSheet('background-color: white')



        ##### Keyboard shortcuts
        self.knext = QShortcut(QKeySequence('f'), self)
        self.knext.activated.connect(self.next)

        self.kprev = QShortcut(QKeySequence('d'), self)
        self.kprev.activated.connect(self.prev)


        # self.blinear.clicked.connect(self.set_scale_linear)

        button_bar_layout.addWidget(self.cbscale)
        button_bar_layout.addWidget(self.cbcolormap)
        button_bar_layout.addWidget(self.bprev)
        button_bar_layout.addWidget(self.bnext)
        # button_bar_layout.addWidget(self.bcounter)



        self.buttons = []


        self.clean_scratch(self.scratchpath)
        self.prepare_png(self.gridsize**2)

        self.df = self.obtain_df()
        self.total_n_frame = int(len(self.listimage)/(self.gridsize**2))
        start = self.config_dict['page']*self.gridarea
        print("Page: ",self.config_dict['page'])
        # n_images = len(self.df) % self.gridarea if self.config_dict['page'] == self.PAGE_MAX else self.gridarea

        for i in range(start,start+self.gridarea):
            filepath = self.filepath(i, self.config_dict['page'])
            try:
                classification = self.df.iloc[i,
                                                 self.df.columns.get_loc('classification')]
                activation = True
            except IndexError:
                classification = False
                activation = False

            button = ClickableLabel(filepath, self.path_background, self.deactivatedpath,
                                    i-start, classification, activation,
                                    self.my_label_clicked)
            stamp_grid_layout.addWidget(
                # button, i // self.gridsize, i % self.gridsize)
                button, i % self.gridsize, i // self.gridsize)
            self.buttons.append(button)
            button.setAlignment(Qt.AlignCenter)
            # button.adjustSize()

#

    @Slot()
    def next(self):
        self.clean_scratch(self.scratchpath)
        self.config_dict['page'] = self.config_dict['page'] + 1
        if self.config_dict['page']>self.PAGE_MAX:
            # self.config_dict['counter']=self.PAGE_MAX
            self.config_dict['page']=self.PAGE_MAX
            self.status.showMessage('Last page',500)
        else:
            self.update_grid()
            self.save_dict()

    @Slot()
    def prev(self):
        self.clean_scratch(self.scratchpath)
        self.config_dict['page'] = self.config_dict['page'] - 1
        if self.config_dict['page'] < 0:
            self.config_dict['page'] = 0
            self.status.showMessage('First page')
        else:
            self.update_grid()
            self.save_dict()

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid()

    def my_label_clicked(self, event, i):
        button = event.button()
        modifiers = event.modifiers()
        if modifiers == Qt.NoModifier and button == Qt.LeftButton:
            if self.config_dict['page']*self.gridarea+i > len(self.listimage):
                print('Not an image')
            else:
                self.df.iloc[self.gridarea*self.config_dict['page']+i,
                            self.df.columns.get_loc('grid_pos')] = i+1

                self.df.iloc[self.gridarea*self.config_dict['page']+i,
                            self.df.columns.get_loc('classification')] = int(self.buttons[i].is_a_candidate)

                self.df.to_csv(
                    './Classifications/classification_mosaic_autosave_' +
                    '{}_{}'.format(self.random_seed, len(self.df))+'.csv', index=False)

    def filepath(self, i, page):
        return join(self.scratchpath, str(i+1)+self.config_dict['scale'] + self.config_dict['colormap'] + str(page)+'.png')

    def save_dict(self):
        with open('.config_mosaic.json', 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        try:
            with open('.config_mosaic.json', ) as f:
                return json.load(f)
        except FileNotFoundError:
            print("Loaded default configuration.")
            return self.defaults

    def obtain_df(self):
        class_file = np.sort(glob.glob(
            './Classifications/classification_mosaic_autosave_'+str(self.random_seed)+'*.csv'))
        # print(class_file, len(class_file))
        if len(class_file) >= 1:
            print('reading ' + str(class_file[len(class_file) - 1]))
            df = pd.read_csv(class_file[len(class_file) - 1])
        else:
            print('creating dataframe')
            self.dfc = ['file_name', 'classification', 'grid_pos']
            df = pd.DataFrame(columns=self.dfc)
            df['file_name'] = self.listimage
            df['classification'] = np.zeros(np.shape(self.listimage))
            df['grid_pos'] = np.zeros(np.shape(self.listimage))
        return df

    def update_grid(self):
        start = self.config_dict['page']*self.gridarea
        # self.textnumber.text = str(self.config_dict['page'])

        # self.clean_scratch(self.scratchpath)
        # n_images = len(self.df) % self.gridarea if self.config_dict['page'] == self.PAGE_MAX else self.gridarea
        n_images = self.gridarea
        # print("page,pageMAX,nimages:",self.config_dict['page'],self.PAGE_MAX,n_images)
        self.prepare_png(n_images)

        i = start
        j = 0
        for button in self.buttons:
            try:
                if self.df.iloc[self.gridarea*self.config_dict['page']+j,self.df.columns.get_loc('classification')] == 0:
                    button.activate()
                    button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                    button.set_candidate_status(False)
                else:
                    button.activate()
                    button.change_pixmap(self.filepath(i,self.config_dict['page']))
                    button.paint_background_pixmap()
                    button.set_candidate_status(True)

                self.df.iloc[self.gridarea*self.config_dict['page']+j,self.df.columns.get_loc('grid_pos')]=j+1

            except (KeyError,IndexError) as e:
                print("Out of bounds in the dataframe.")
                button.deactivate()
                # button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                # raise

            j = j+1
            i = i+1

    def prepare_png(self, number):

        start = self.config_dict['page']*self.gridarea
        for i in np.arange(start, start + number + 1):
            # img = self.draw_image(i, self.scale_state)
            try:
                image = self.read_fits(i)
            except:
                image = np.ones((44, 44)) * 0.0000001

            scale_min, scale_max = self.scale_val(image)
            image = self.rescale_image(image, scale_min, scale_max)
            plt.imsave(self.filepath(i, self.config_dict['page']),
                       image, cmap=self.config_dict['colormap'], origin="lower")

            # self.config_dict['counter'] = self.config_dict['counter'] + 1

    def clean_scratch(self, path_dir):
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))

    def read_fits(self, i):
        file = join(self.stampspath, self.listimage[i])
        # Note : memmap=False is much faster when opening/closing many small files
        with fits.open(file, memmap=False) as hdu_list:
            image = hdu_list[0].data
        return image

    def rescale_image(self, image, scale_min, scale_max):
        factor = self.scale2funct[self.config_dict['scale']](scale_max - scale_min)
        image = image.clip(min=scale_min, max=scale_max)
        #image = (image - self.scale_min) / factor
        indices0 = np.where(image < scale_min)
        indices1 = np.where((image >= scale_min) & (image <= scale_max))
        indices2 = np.where(image > scale_max)
        image[indices0] = 0.0
        image[indices2] = 1.0
        image[indices1] = self.scale2funct[self.config_dict['scale']](image[indices1]) / (factor * 1.0)
        return image

    def scale_val(self, image_array):
        if len(np.shape(image_array)) == 2:
            image_array = [image_array]
        vmin = np.min([self.background_rms_image(5, image_array[i])
                      for i in range(len(image_array))])
        xl, yl = np.shape(image_array[0])
        box_size = 14  # in pixel
        xmin = int((xl) / 2. - (box_size / 2.))
        xmax = int((xl) / 2. + (box_size / 2.))
        vmax = np.max([image_array[i][xmin:xmax, xmin:xmax]
                      for i in range(len(image_array))])
        return vmin, vmax*1.5

    def background_rms_image(self, cb, image):
        xg, yg = np.shape(image)
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        m = np.mean(np.mean(l, axis=1), axis=1)
        ml = min(m)
        mm = max(m)
        if mm > 5 * ml:
            s = np.sort(l, axis=0)
            nl = s[:-1]
            std = np.std(nl)
        else:
            std = np.std([cut0, cut1, cut2, cut3])
        return std

#


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = MosaicVisualizer()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
