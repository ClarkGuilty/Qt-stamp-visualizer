#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:15:48 2023

@author: Javier Alejandro Acevedo Barroso
"""

import argparse
from os.path import join, getmtime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
#%%
parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
# parser.add_argument('-p',"--path", help="Path to the images to inspect",
#                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default="")


args = parser.parse_args()

classification_path = "Classifications"
#classification_csv = "classification_autosave1_early_test.csv"
no_elements = '*'
mosaic_pattern = f"classification_mosaic_autosave_{args.name}_{no_elements}_10_99.csv"
single_pattern = f"classification_single_{args.name}_{no_elements}.csv"

def search_files(path,name_pattern):
    files = sorted(glob(join(path,name_pattern)),key=getmtime)
    if len(files) == 0:
        raise FileNotFoundError(f'No files found with the pattern: {name_pattern}')
    print(files) if len(files) > 1 else False
    return files[-1]

mosaic_csv = search_files(classification_path,mosaic_pattern)
single_csv = search_files(classification_path,single_pattern)
mosaic_df = pd.read_csv(mosaic_csv,)
single_df = pd.read_csv(single_csv,)

def print_mosaic_summary(df,name=args.name):
    print(f"Mosaic classification, name: {name}")
    print(f"Number of selected objects: {df.classification.sum().astype(int):d}")
    print(f"Density of selected objects: {round(100*df.classification.mean(),2)} %")

class_dict = {'SL': 3,
              'ML': 2,
              'FL': 1,
              'NL': 0}

def print_single_summary(df):
    print(f"Single stamp classification, name: {args.name}")
    numerical_class = df.classification.map(str.strip).map(class_dict)
    print(f"Number of interesting objects: {np.sum(numerical_class > 0)}")
    print(f"Density of interesting objects: {round(100*np.mean(numerical_class > 0),2)} %")
    print(df.keys())
    for row in df[numerical_class > 0]:
        print(row)
#    print(f"Density of selected objects: {round(100*df.classification.mean(),4)} %")

#print(mosaic_df)
print_mosaic_summary(mosaic_df)
print_single_summary(single_df)
#selected_df = df[df.classification > 0]
#for index, row in selected_df.iterrows():
    # print(row)
# /home/alejandro/imported/inferences
#    print(row.file_name)
