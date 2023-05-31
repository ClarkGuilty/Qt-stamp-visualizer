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

import argparse
#%%
parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-c',"--path_to_classification", help="Path to classification csv")
args = parser.parse_args()
#%%
# classification_path = "Classifications"
# classification_csv = "classification_autosave1_early_test.csv"
# classification_csv = "classification_mosaic_autosave__33107_99.csv"
# classification_csv = "classification_mosaic_autosave_false_positive_inspection_943_10_99.csv"

# df = pd.read_csv(join(classification_path,classification_csv))
df = pd.read_csv(args.path_to_classification)


selected_df = df[df.classification == 1.0]
for index, row in selected_df.iterrows():
    # print(row)
# /home/alejandro/imported/inferences
    print(row.file_name)
