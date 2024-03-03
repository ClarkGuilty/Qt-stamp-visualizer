from os.path import join
import os
import pandas as pd
from shutil import copy2

import argparse


parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument("path_to_classification_file", help="Path to the classification file.")
parser.add_argument("path_to_the_images", help="Path to directory containing the images in the different bands.")
parser.add_argument("output_path", help="Path to save the selected stamps.")
parser.add_argument("--interesting", help='Set if you want to extract the objects labeled as "interesting"',
                    action=argparse.BooleanOptionalAction,
                    default=False)
args = parser.parse_args()

df = pd.read_csv(args.path_to_classification_file)
selected_filenames = df.loc[df.classification == 1.0].file_name

try:
    os.makedirs(args.output_path)
except FileExistsError:
    print(f"WARNING: The output filepath already existed")

bands = os.listdir(args.path_to_the_images)

for band in bands:
    base_path = join(args.path_to_the_images,band)
    output_path = join(args.output_path,band)
    os.makedirs(output_path,exist_ok=True)
    
    for filename in selected_filenames:
        try:
            copy2(join(base_path,filename), join(output_path,filename) )
        except FileNotFoundError:
            print(f"File not found: {join(base_path,filename)}")
            print(join(output_path,filename))
