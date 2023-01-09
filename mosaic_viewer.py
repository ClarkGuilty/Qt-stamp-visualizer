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
from PySide6.QtCore import Qt, Slot, QObject, QThread
from PySide6.QtGui import QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvas
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


class ClickableLabel(QtWidgets.QLabel):
    def __init__(self, filepath, backgroundpath, i, update_df_func, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.filepath = filepath
        # self._whenClicked = whenClicked
        self.backgroundpath = backgroundpath
        self.is_a_candidate = False
        self.update_df_func = update_df_func
        self.i = i

    def set_candidate_status(self,status):
        self.is_a_candidate = status

    def toggle_candidate_status(self):
        self.is_a_candidate = not self.is_a_candidate

    def change_pixmap(self,filepath):
        self.filepath = filepath
        self.setPixmap(QPixmap(self.filepath))

    def mousePressEvent(self, event):
        # self._whenClicked(event)
        if self.is_a_candidate:
            self.setPixmap(QPixmap(self.filepath))
        else:
            self.setPixmap(QPixmap(self.backgroundpath))
        
        self.is_a_candidate = not self.is_a_candidate
        self.update_df_func(event,self.i)
        # self.update_df_func()



        
class MosaicVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setWindowTitle("Mosaic Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = 99

        self.stampspath = './Stamps_to_inspect/'
        self.scratchpath = './.temp'
        os.makedirs(self.scratchpath,exist_ok=True)

        self.listimage = sorted([os.path.basename(x) for x in glob.glob(self.stampspath+ '*.fits')])


        self.scale2funct = {'identity':identity,'sqrt':np.sqrt,'log10':np.log10, 'asinh2':asinh2}

        self.defaults = {
                    'counter':0,
                    'page':0,
                    'colormap':'gray',
                    'scale':'log10',
                        }
        self.config_dict = self.load_dict()
        self.scale = self.scale2funct[self.config_dict['scale']]
        self.path_background = '.background.png'

        main_layout = QtWidgets.QGridLayout(self._main)


        self.buttons = []
        self.gridsize = 10
        self.gridarea = self.gridsize**2

        self.clean_scratch(self.scratchpath)
        self.prepare_png(self.gridsize**2)

        self.df = self.obtain_df()

        self.total_n_frame =int(len(self.listimage)/(self.gridsize**2))
        for i in range(self.gridarea):
            filepath = self.filepath(i,0)
            button = ClickableLabel(filepath,self.path_background,i,self.my_label_clicked)
            button.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                       QtWidgets.QSizePolicy.Ignored)
            button.setScaledContents(True)
            button.setPixmap(QPixmap(filepath))
            main_layout.addWidget(button,i // self.gridsize, i % self.gridsize)
            self.buttons.append(button)
            button.adjustSize()

#

    def my_label_clicked(self, event, i):
        button = event.button()
        modifiers = event.modifiers()

        if modifiers == Qt.NoModifier and button == Qt.LeftButton:
            # print(self.df)
            self.df.iloc[self.gridarea*self.config_dict['page']+i,
             self.df.columns.get_loc('grid_pos')] =i+1

            self.df.iloc[self.gridarea*self.config_dict['page']+i,
                 self.df.columns.get_loc('classification')] = int(self.buttons[i].is_a_candidate)

            self.df.to_csv(
                './Classifications/classification_mosaic_autosave_'+
                '{}_{}'.format(self.random_seed,len(self.df))+'.csv', index=False)


    def filepath(self,i,page):
        return join(self.scratchpath,str(i+1)+self.config_dict['scale'] +self.config_dict['colormap']+ str(page)+'.png')

    def load_dict(self):
        try:
            with open('.config_mosaic.json', ) as f:
                return json.load(f)
        except FileNotFoundError:
            return self.defaults

    @Slot()
    def open_ds9(self):
        subprocess.Popen(["ds9",  ])

    def set_start_number(self):
        self.config_dict['counter'] = self.config_dict['page'] * self.gridarea

    def obtain_df(self):
        class_file = np.sort(glob.glob(
            './Classifications/classification_mosaic_autosave_'+str(self.random_seed)+'*.csv'))
        print (class_file, len(class_file))
        if len(class_file) >= 1:
            print('reading ' + str(class_file[len(class_file) - 1]))
            df = pd.read_csv(class_file[len(class_file) - 1])
        else:
            print('creating dataframe')
            self.dfc = ['file_name', 'classification','grid_pos']
            df = pd.DataFrame(columns=self.dfc)
            df['file_name'] = self.listimage
            df['classification'] = np.zeros(np.shape(self.listimage))
            df['grid_pos'] = np.zeros(np.shape(self.listimage))
        return df

    def update_grid():
        self.set_start_number()
        start = self.config_dict['counter']
        self.textnumber.text = str(self.config_dict['page'])

        self.clean_scratch(self.scratchpath)
        self.prepare_png(self.gridarea)

        i = start
        j=0
        for button in self.buttons:
            if self.dataframe['classification'][self.gridarea*self.config_dict['page']+j]==0:
                # button.set_source(self.scratchpath+str(i+1)+self.config_dict['scale']+self.colormap+ str(start)+'.png')
                button.set_candidate_status(True)

            else:
                # button.set_source(self.path_background)
                # button.set_lensing_value(1)
                pass
            self.dataframe['grid_pos'].iloc[self.gridarea * self.config_dict['page'] + j] = j + 1
            # button.set_source('cutecat.png')
            j=j+1
            i=i+1

    def prepare_png(self, number):

        start = self.config_dict['counter']
        for i in np.arange(start, start + number + 1):
            # img = self.draw_image(i, self.scale_state)
            image = self.read_fits(i)
            scale_min, scale_max = self.scale_val(image)
            image = self.rescale_image(image,scale_min,scale_max)
            plt.imsave(join(self.scratchpath,str(i+1)+self.config_dict['scale']+
              self.config_dict['colormap']+str(start)+'.png'),image,cmap=self.config_dict['colormap'])

            self.config_dict['counter'] = self.config_dict['counter'] + 1

    def clean_scratch(self, path_dir):
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))

    def read_fits(self, i):
        file = join(self.stampspath, self.listimage[i])
        # Note : memmap=False is much faster when opening/closing many small files
        with fits.open(file, memmap=False) as hdu_list:
            image = hdu_list[0].data
        return image

    def rescale_image(self, image,scale_min, scale_max):
            factor = self.scale(scale_max - scale_min)
            image = image.clip(min=scale_min, max=scale_max)
            #image = (image - self.scale_min) / factor
            indices0 = np.where(image < scale_min)
            indices1 = np.where((image >= scale_min) & (image <= scale_max))
            indices2 = np.where(image > scale_max)
            image[indices0] = 0.0
            image[indices2] = 1.0
            image[indices1] = self.scale(image[indices1]) / (factor * 1.0)
            return image

    def scale_val(self,image_array):
        if len(np.shape(image_array)) == 2:
            image_array = [image_array]
        vmin = np.min([self.background_rms_image(5, image_array[i]) for i in range(len(image_array))])
        xl, yl = np.shape(image_array[0])
        box_size = 14  # in pixel
        xmin = int((xl) / 2. - (box_size / 2.))
        xmax = int((xl) / 2. + (box_size / 2.))
        vmax = np.max([image_array[i][xmin:xmax, xmin:xmax] for i in range(len(image_array))])
        return vmin, vmax*1.3

    def background_rms_image(self,cb, image):
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
