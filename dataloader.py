import os
import json
import nibabel as nib
import pandas as pd
from tqdm import tqdm

def oasis_convert(filepath = None):

    with (open("localdata.json", 'r')) as json_file:
        oasis_path = json.load(json_file).get("oasis")

    if filepath is None:
        data_path = os.path.join(oasis_path, "data")
        for folder in tqdm(os.listdir(data_path)):
            dir = os.path.join(data_path, folder)
            if os.path.isdir(dir):
                img = nib.load(os.path.join(dir, "mri/aseg.mgz"))
                if img.get_fdata().shape != (256, 256, 256):
                    print("ERR")

# def oasis_loader(filepath: None):
#
#     with (open("localdata.json", 'r')) as json_file:
#         oasis_path = json.load(json_file).get("oasis")
#
#     if filepath is None:
#         for dir in os.listdir(oasis_path):

if __name__ == "__main__":
    oasis_convert()