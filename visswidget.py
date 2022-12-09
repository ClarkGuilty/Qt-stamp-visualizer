# This Python file uses the following encoding: utf-8
import sys
#from PySide6.QtWidgets import QApplication, QWidget, QPushButton
import PySide6

import time
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import glob
import os
import pandas as pd
import subprocess

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg

import urllib
import concurrent.futures

def identity(x):
    return x

def asinh2(x):
    return np.arcsinh(x/2)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setWindowTitle("Stamp Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()

        self.status_legacy_survey_panel = False

        self.scale = identity
        self.colormap = "gray"

        self.stampspath = './Stamps_to_inspect/'
        self.legacy_survey_path = './Legacy_survey/'
        self.listimage = sorted([os.path.basename(x) for x in glob.glob(self.stampspath+ '*.fits')])
        self.classification = ['None'] * len(self.listimage)
        self.subclassification = ['None'] * len(self.listimage)
        self.ra = ['None'] * len(self.listimage)
        self.dec = ['None'] * len(self.listimage)
        self.comment = [' '] * len(self.listimage)
        self.df = self.obtain_df()

        self.counter = 0
        self.number_graded = 0
        self.COUNTER_MIN =0
        self.COUNTER_MAX = len(self.listimage)
        self.filename = self.stampspath + self.listimage[self.counter]
        self.status.showMessage(self.listimage[self.counter],)
        self.figure = [Figure(figsize=(5,3)),Figure(figsize=(5,3))]

        self.legacy_survey_qlabel = QtWidgets.QLabel(alignment=Qt.AlignCenter)
        pixmap = QPixmap()
        #self.figure.gca().set_facecolor('black')

        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        self.plot_layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()
        button_row1_layout = QtWidgets.QHBoxLayout()
        button_row2_layout = QtWidgets.QHBoxLayout()
        button_row3_layout = QtWidgets.QHBoxLayout()

        self.label_plot = [QtWidgets.QLabel(self.listimage[self.counter], alignment=Qt.AlignCenter),
                            QtWidgets.QLabel("Legacy Survey", alignment=Qt.AlignCenter)]
        font = [x.font() for x in self.label_plot]
        for i in range(len(font)):
            font[i].setPointSize(16)
            self.label_plot[i].setFont(font[i])
        self.label_layout.addWidget(self.label_plot[0])


        self.canvas = [FigureCanvas(self.figure[0]),FigureCanvas(self.figure[1])]
        self.plot_layout.addWidget(self.canvas[0])

        self.ax = [self.figure[0].subplots(),self.figure[1].subplots()]

        self.plot()

        self.bds9 = QtWidgets.QPushButton('ds9')
        self.bds9.clicked.connect(self.open_ds9)

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)

        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)

        self.blegsur = QtWidgets.QPushButton('Legacy Survey')
        self.blegsur.clicked.connect(self.set_legacy_survey)

        self.blinear = QtWidgets.QPushButton('Linear')
        self.blinear.clicked.connect(self.set_scale_linear)

        self.bsqrt = QtWidgets.QPushButton('Sqrt')
        self.bsqrt.clicked.connect(self.set_scale_sqrt)

        self.blog = QtWidgets.QPushButton('Log10')
        self.blog.clicked.connect(self.set_scale_log)

        self.basinh = QtWidgets.QPushButton('Asinh')
        self.basinh.clicked.connect(self.set_scale_asinh)

        self.bInverted = QtWidgets.QPushButton('Inverted')
        self.bInverted.clicked.connect(self.set_colormap_Inverted)

        self.bBb8 = QtWidgets.QPushButton('Bb8')
        self.bBb8.clicked.connect(self.set_colormap_Bb8)

        self.bGray = QtWidgets.QPushButton('Gray')
        self.bGray.clicked.connect(self.set_colormap_Gray)

        self.bViridis = QtWidgets.QPushButton('Viridis')
        self.bViridis.clicked.connect(self.set_colormap_Viridis)

        button_row2_layout.addWidget(self.blinear)
        button_row2_layout.addWidget(self.bsqrt)
        button_row2_layout.addWidget(self.blog)
        button_row2_layout.addWidget(self.basinh)
        button_row3_layout.addWidget(self.bInverted)
        button_row3_layout.addWidget(self.bBb8)
        button_row3_layout.addWidget(self.bGray)
        button_row3_layout.addWidget(self.bViridis)


        button_row1_layout.addWidget(self.bprev)
        button_row1_layout.addWidget(self.bnext)
        button_row1_layout.addWidget(self.bds9)
        button_row1_layout.addWidget(self.blegsur)

        button_layout.addLayout(button_row1_layout, 34)
        button_layout.addLayout(button_row2_layout, 33)
        button_layout.addLayout(button_row3_layout, 33)


        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout, 88)
        main_layout.addLayout(button_layout, 10)


    def get_legacy_survey(self,pixscale = '0.06'):
        url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(self.ra) + '&dec=' + str(
            self.dec) + '&layer=dr8&pixscale='+str(pixscale)
        savename = 'N' + str(self.counter)+ '_' + str(self.ra) + '_' + str(self.dec) + 'dr8.jpg'
        urllib.request.urlretrieve(url, os.path.join(self.legacy_survey_path, savename))
#        url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(self.ra) + '&dec=' + str(
#            self.dec) + '&layer=dr8-resid&pixscale=0.06'
#        savename = 'N' + str(self.counter)+ '_' + str(self.ra) + '_' + str(self.dec) + 'dr8-resid.jpg'
#        urllib.request.urlretrieve(url, os.path.join(self.legacy_survey_path, savename))
        print(url)
        #return savename
        return os.path.join(self.legacy_survey_path, savename)

    def plot_legacy_survey(self, filepath, title='12x12', canvas_id = 1):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        self.ax[canvas_id].imshow(mpimg.imread(filepath))
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    @Slot()
    def set_legacy_survey(self):
        if self.status_legacy_survey_panel:
            legacy_filename = self.get_legacy_survey()
            self.plot_legacy_survey(legacy_filename)
        else:
            self.status_legacy_survey_panel = True
            legacy_filename = self.get_legacy_survey()
            self.label_layout.addWidget(self.label_plot[1])
            self.plot_legacy_survey(legacy_filename)
            self.plot_layout.addWidget(self.canvas[1])



    @Slot()
    def open_ds9(self):
        subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])


    @Slot()
    def set_scale_linear(self):
        self.scale = identity
        self.replot()

    @Slot()
    def set_scale_sqrt(self):
        self.scale = np.sqrt
        self.replot()

    @Slot()
    def set_scale_log(self):
        self.scale = np.log10
        self.replot()

    @Slot()
    def set_scale_asinh(self):
        self.scale = asinh2
        self.replot()

    @Slot()
    def set_colormap_Inverted(self):
        self.colormap = "gist_yarg"
        self.replot()

    @Slot()
    def set_colormap_Bb8(self):
        self.colormap = "hot"
        self.replot()

    @Slot()
    def set_colormap_Gray(self):
        self.colormap = "gray"
        self.replot()

    @Slot()
    def set_colormap_Viridis(self):
        self.colormap = "viridis"
        self.replot()

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

    def scale_val(self,image_array):
        if len(np.shape(image_array)) == 2:
            image_array = [image_array]
        vmin = np.min([self.background_rms_image(5, image_array[i]) for i in range(len(image_array))])
        xl, yl = np.shape(image_array[0])
        box_size = 14  # in pixel
        xmin = int((xl) / 2. - (box_size / 2.))
        xmax = int((xl) / 2. + (box_size / 2.))
        vmax = np.max([image_array[i][xmin:xmax, xmin:xmax] for i in range(len(image_array))])
        return vmin, vmax*2.0

    def rescale_image(self, image):
            factor = self.scale(self.scale_max - self.scale_min)
            image = image.clip(min=self.scale_min, max=self.scale_max)
            #image = (image - self.scale_min) / factor
            indices0 = np.where(image < self.scale_min)
            indices1 = np.where((image >= self.scale_min) & (image <= self.scale_max))
            indices2 = np.where(image > self.scale_max)
            image[indices0] = 0.0
            image[indices2] = 1.0
            image[indices1] = self.scale(image[indices1]) / (factor * 1.0)
            return image

    def load_fits(self,filepath):
        opened_fits = fits.open(filepath)
        self.ra,self.dec = self.get_ra_dec(opened_fits[0].header)
        return opened_fits[0].data

    def get_ra_dec(self,header):
        w = WCS(header,fix=False)
        sky = w.pixel_to_world_values([w.array_shape[0]//2], [w.array_shape[1]//2])
        return sky[0][0], sky[1][0]

    def plot(self, scale_min = None, scale_max = None, canvas_id = 0):
        self.label_plot[canvas_id].setText(self.listimage[self.counter])

        self.ax[canvas_id].cla()
        image = self.load_fits(self.filename)
        self.image = np.copy(image)
        if scale_min is not None and scale_max is not None:
            self.scale_min = scale_min
            self.scale_max = scale_max
        else:
            self.scale_min, self.scale_max = self.scale_val(image)

        image = self.rescale_image(image)

        self.ax[canvas_id].imshow(image,cmap=self.colormap, origin='lower')
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def replot(self, scale_min = None, scale_max = None,canvas_id = 0):
        self.label_plot[canvas_id].setText(self.listimage[self.counter])

        self.ax[canvas_id].cla()

        image = np.copy(self.image)
        image = self.rescale_image(image)

        self.ax[canvas_id].imshow(image,cmap=self.colormap, origin='lower')
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def obtain_df(self):
        class_file = np.sort(glob.glob('./Classifications/classification_autosave*.csv'))
        print (class_file, len(class_file))
        if len(class_file) >=1:
            print ('loop')
            print('reading '+str(class_file[len(class_file)-1]))
            df = pd.read_csv(class_file[len(class_file)-1])
            self.nf = len(class_file)
#            self.classification = df['classification'].tolist()
#            self.subclassification = df['subclassification'].tolist()
#            self.comment = df['comment'].tolist()

            firstnone=self.classification.index('None')
            self.counter =firstnone #Remembering last position

        else:
            df=[]
        if len(df) != len(self.listimage):
            print('creating classification_autosave'+str(len(class_file)+1)+'.csv')
            dfc = ['file_name', 'classification', 'subclassification','comment']
            self.nf = len(class_file) + 1
            df = pd.DataFrame(columns=dfc)
            df['file_name'] = self.listimage
            df['classification'] = self.classification
            df['subclassification'] = self.subclassification
            df['comment'] = self.comment

        return df

    @Slot()
    def next(self):
        self.counter = self.counter + 1

        if self.counter>self.COUNTER_MAX-1:
            self.counter=self.COUNTER_MAX-1
            self.status.showMessage('Last image')

        else:
            #self.status.showMessage(self.listimage[self.counter])
            #self.label_plot0 = QtWidgets.QLabel(self.listimage[self.counter], alignment=Qt.AlignCenter)
            self.filename = self.stampspath + self.listimage[self.counter]
            self.plot()

    @Slot()
    def prev(self):
        self.counter = self.counter - 1

        if self.counter<self.COUNTER_MIN:
            self.counter=self.COUNTER_MIN
            self.status.showMessage('First image')

        else:
            #self.status.showMessage(self.listimage[self.counter])
            self.filename = self.stampspath + self.listimage[self.counter]
            self.plot()
            
            
            
if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
