#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 01:42:48 2023

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
#%%
classification_path = "Classifications"
#classification_csv = "classification_autosave1_early_test.csv"
classification_csv = "classification_mosaic_autosave_99_33107.csv"

df = pd.read_csv(join(classification_path,classification_csv))


selected_df = df[df.classification > 0]
for index, row in selected_df.iterrows():
    # print(row)
# /home/alejandro/imported/inferences
    print(row.file_name)
