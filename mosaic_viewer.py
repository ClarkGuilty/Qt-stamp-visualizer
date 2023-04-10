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

import argparse

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

parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="Path to the images to inspect",
                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default="")
parser.add_argument('-l',"--gridsize", help="Number of stamps per side.",type=int,
                    default=10)
parser.add_argument("--printname", help="Whether to print the name when you click",type=bool,default=True)
parser.add_argument("--page", help="Initial page",type=int,
                    default=None)

args = parser.parse_args()


def identity(x):
    return x

def log(x):
    return np.emath.logn(1000,x) #base 1000 like ds9

def asinh2(x):
    return np.arcsinh(x/2)

class LabelledIntField(QtWidgets.QWidget):
    "Widget for the page number."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value,  total_pages):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        self.label = QtWidgets.QLabel()
        self.label.setText(title)
        # self.label.setFixedWidth(100)
        self.label.setFont(QFont("Arial",20,weight=QFont.Bold))
        layout.addWidget(self.label)
        
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setFixedWidth(50)
        self.lineEdit.setValidator(QIntValidator(1,total_pages+1))
        self.lineEdit.setText(str(initial_value+1))
        self.lineEdit.setFont(QFont("Arial",20))
        self.lineEdit.setStyleSheet('background-color: black; color: gray')
        self.lineEdit.setAlignment(Qt.AlignRight)
        layout.addWidget(self.lineEdit)

        self.total_pages = QtWidgets.QLabel()
        self.total_pages.setText("/ "+str(total_pages+1))
        self.total_pages.setFont(QFont("Arial",20))
        layout.addWidget(self.total_pages)

        # layout.addStretch()

    def setInputText(self, input):
        self.lineEdit.setText(str(input+1))
        
    def getValue(self):
        return int(self.lineEdit.text())-1

class NamedLabel(QtWidgets.QWidget):
    "Widget to show unclickable label."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        self.name = QtWidgets.QLabel()
        self.name.setText(title)
        # self.title.setFixedWidth(100)
        self.name.setFont(QFont("Arial",20,weight=QFont.Bold))
        layout.addWidget(self.name)
        
        self.label = QtWidgets.QLineEdit(self)
        self.label.setFixedWidth(50)
        self.label.setEnabled(False)
        self.label.setText(str(initial_value))
        self.label.setFont(QFont("Arial",20))
        self.label.setStyleSheet('background-color: black; color: gray')
        self.label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label)

        # self.total_pages = QtWidgets.QLabel()
        # self.total_pages.setText("/ "+str(total_pages+1))
        # self.total_pages.setFont(QFont("Arial",20))
        # layout.addWidget(self.total_pages)

        # layout.addStretch()

    def setText(self, input):
        self.label.setText(str(input))

        
    def getValue(self):
        return int(self.lineEdit.text())-1



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
    clicked = Signal()
    def __init__(self, filepath, backgroundpath, deactivatedpath, i,
     status, activation, update_df_func, parent=None):
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
        # print(self.is_activate)
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
    def __init__(self, path_to_the_stamps= args.path):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')
        #print(path_to_the_stamps)
        self.setWindowTitle("Mosaic Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = 99

        self.stampspath = path_to_the_stamps
        self.scratchpath = './.temp'
        self.deactivatedpath = './dark.png'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.clean_dir(self.scratchpath)
        self.listimage = sorted([os.path.basename(x)
                                for x in glob.glob(join(self.stampspath, '*.fits'))])
        # print(join(self.stampspath, '*.fits'))
        # print(glob.glob(self.stampspath + '*.fits'))
        self.gridsize = args.gridsize
        self.gridarea = self.gridsize**2
        self.PAGE_MAX = int(np.floor(len(self.listimage) / self.gridarea))

        self.scale2funct = {'linear': identity,
                            'sqrt': np.sqrt,
                            'cbrt': np.cbrt,
                            'log': log,
                            # 'log10': log, #just for retrocompatibility
                            'asinh': asinh2}

        self.defaults = {
            # 'counter': 0,
            'page': 0, #Defaults to 0. Gets overwritten by --page argument.
            # 'total': -1,
            'colormap': 'gray',
            'scale': 'asinh',
            'name': args.name,
            'gridsize': args.gridsize
        }
        self.config_dict = self.load_dict()
        self.scale = self.scale2funct[self.config_dict['scale']]
        self.path_background = '.background.png'
        self.deactivatedpath = '.backgrounddark.png'


        self.buttons = []
        self.clean_dir(self.scratchpath)

        self.df = self.obtain_df()
        self.prepare_png(self.gridsize**2)

        main_layout = QtWidgets.QVBoxLayout(self._main)
        stamp_grid_layout = QtWidgets.QGridLayout()
        bottom_bar_layout = QtWidgets.QHBoxLayout()
        button_bar_layout = QtWidgets.QHBoxLayout()
        page_counter_layout = QtWidgets.QHBoxLayout()


        bottom_bar_layout.addLayout(button_bar_layout,10)
        bottom_bar_layout.addLayout(page_counter_layout,1)
        main_layout.addLayout(stamp_grid_layout, 8)
        main_layout.addLayout(bottom_bar_layout, )

        #### Buttons
        self.cbscale = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbscale)
        self.cbscale.setItemDelegate(delegate)
        # self.cbscale.setEditable(True)
        self.cbscale.setFont(QFont("Arial",20))
        clickable(self.cbscale).connect(self.cbscale.showPopup)
        line_edit = self.cbscale.lineEdit()
        # self.cbscale.addItems(['linear','sqrt','' 'log', 'asinh'])
        self.cbscale.addItems(self.scale2funct.keys())
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

        # self.bcounter = LabelledIntField('Page', self.config_dict['page'], self.PAGE_MAX)
        self.bcounter = LabelledIntField('Page', self.config_dict['page'], self.PAGE_MAX)
        self.bcounter.setStyleSheet('background-color: black; color: gray')
        self.bcounter.lineEdit.returnPressed.connect(self.goto)

        print(self.df['classification'].sum().astype(int))
        self.bclickcounter = NamedLabel('Clicks', self.df['classification'].sum().astype(int))
        self.bclickcounter.setStyleSheet('background-color: black; color: gray')

        # button.clicked.connect(line_edit.clear)


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
        page_counter_layout.addWidget(self.bclickcounter)
        page_counter_layout.addWidget(self.bcounter)


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
    def goto(self):
        if self.bcounter.getValue()>self.PAGE_MAX:
            print(self.PAGE_MAX)
            self.status.showMessage('WARNING: There are only {} pages.'.format(
                self.PAGE_MAX+1),10000)
        elif self.bcounter.getValue()<0:
            self.status.showMessage('WARNING: Pages go from 1 to {}.'.format(
                self.PAGE_MAX+1),10000)
        else:
            self.config_dict['page'] = self.bcounter.getValue()
            self.update_grid()
            # self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    @Slot()
    def next(self):
        self.config_dict['page'] = self.config_dict['page'] + 1
        if self.config_dict['page']>self.PAGE_MAX:
            # self.config_dict['counter']=self.PAGE_MAX
            self.config_dict['page']=self.PAGE_MAX
            self.status.showMessage('You are already at the last page',10000)
        else:
            self.clean_dir(self.scratchpath)
            self.update_grid()
            self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    @Slot()
    def prev(self):
        self.config_dict['page'] = self.config_dict['page'] - 1
        if self.config_dict['page'] < 0:
            self.config_dict['page'] = 0
            self.status.showMessage('You are already at the first page',10000)
        else:
            self.clean_dir(self.scratchpath)
            self.update_grid()
            self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()
        self.save_dict()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid()
        self.save_dict()

    def my_label_clicked(self, event, i):
        button = event.button()
        modifiers = event.modifiers()
        if modifiers == Qt.NoModifier and button == Qt.LeftButton:
            if self.config_dict['page']*self.gridarea+i > len(self.listimage):
                print('Not an image')
            else:
                self.df.iloc[self.gridarea*self.config_dict['page']+i,
                            self.df.columns.get_loc('grid_pos')] = i+1
                print(self.df.iloc[self.gridarea*self.config_dict['page']+i,
                            self.df.columns.get_loc('file_name')]) if args.printname else True
                self.df.iloc[self.gridarea*self.config_dict['page']+i,
                            self.df.columns.get_loc('classification')] = int(self.buttons[i].is_a_candidate)

                self.bclickcounter.setText(self.df['classification'].sum().astype(int))
                self.df.to_csv(
                    self.df_name, index=False)

    def filepath(self, i, page):
        return join(self.scratchpath, str(i+1)+self.config_dict['scale'] + self.config_dict['colormap'] + str(page)+'.png')

    def save_dict(self):
        with open('.config_mosaic.json', 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    # def load_dict(self):
    #     try:
    #         with open('.config_mosaic.json', ) as f:
    #             return json.load(f)
    #     except FileNotFoundError:
    #         print("Loaded default configuration.")
    #         return self.defaults

    def load_dict(self):
        try:
            with open('.config_mosaic.json', ) as f:
                temp_dict = json.load(f)
                if ((temp_dict['name'] != args.name) or
                    (temp_dict['gridsize'] != args.gridsize)):
                    temp_dict['name'] = args.name
                    temp_dict['name'] = args.gridsize
                if args.page is not None:
                    temp_dict['page'] = args.page
                if temp_dict['scale'] == 'log10':
                    temp_dict['scale'] = 'log'
                return temp_dict
        except FileNotFoundError:
            print("Loaded default configuration.")
            return self.defaults

    def obtain_df(self):
        string_to_glob = './Classifications/classification_mosaic_autosave_{}_{}_{}_{}*.csv'.format(
                                    args.name,len(self.listimage),self.gridsize, str(self.random_seed))
        class_file = np.sort(glob.glob(
            string_to_glob))
        print("Globing for", string_to_glob)

        # print(class_file, len(class_file))
        if len(class_file) >= 1:
            print('Reading '+str(class_file[len(class_file)-1]))
            self.df_name = class_file[len(class_file) - 1]
            df = pd.read_csv(self.df_name)
            if len(self.listimage) == len(df):
                # print("saved classification has the same number of rows as there are images.")
                self.listimage = df['file_name'].values
                return df
            else:
                print("The number of rows in the csv and the number of images must be equal.")
        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        self.df_name = './Classifications/classification_mosaic_autosave_{}_{}_{}_{}.csv'.format(
                                    args.name,len(self.listimage),self.gridsize,str(self.random_seed))
        print('Creating new csv', self.df_name)
        df = pd.DataFrame(columns=self.dfc)
        df['file_name'] = self.listimage
        df['classification'] = np.zeros(np.shape(self.listimage))
        df['page'] = np.zeros(np.shape(self.listimage))
        df['grid_pos'] = np.zeros(np.shape(self.listimage))
        self.config_dict['page'] = 0
        self.bcounter.setInputText(self.config_dict['page'])
        return df

    def update_grid(self):
        start = self.config_dict['page']*self.gridarea
        n_images = self.gridarea
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

                self.df.iloc[self.gridarea*self.config_dict['page']+j,
                             self.df.columns.get_loc('grid_pos')] = j+1

                self.df.iloc[self.gridarea*self.config_dict['page']+j,
                             self.df.columns.get_loc('page')] = self.config_dict['page']

            except (KeyError,IndexError) as e:
                # print("Out of bounds in the dataframe.")
                button.deactivate()
                # button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                # raise

            j = j+1
            i = i+1

    def prepare_png(self, number):
        "Generates the png files from the fits."
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

    def clean_dir(self, path_dir):
        "Removes everything in the scratch folder."
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
    # app.raise_()
    qapp.exec()
