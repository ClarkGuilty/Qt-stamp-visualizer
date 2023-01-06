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

import urllib

from functools import partial

import webbrowser
import json
from os.path import join

def identity(x):
    return x

def asinh2(x):
    return np.arcsinh(x/2)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setWindowTitle("Mosaic Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()

        main_layout = QtWidgets.QVBoxLayout(self._main)

        self.bds9 = QtWidgets.QPushButton('ds9')
        self.bds9.clicked.connect(self.open_ds9)

        self.figure = [Figure(figsize=(5,3),layout="constrained",facecolor='black'),
            Figure(figsize=(5,3),layout="constrained",facecolor='black')]
        self.canvas = [FigureCanvas(self.figure[0]),FigureCanvas(self.figure[1])]
        self.canvas[0].setStyleSheet('background-color: blue')
        self.canvas[0].clicked.connect(self.open_ds9)
# 
        main_layout.addWidget(self.canvas[1])
        main_layout.addWidget(self.bds9)

    @Slot()
    def open_ds9(self):
        subprocess.Popen(["ds9",  ])

            
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
