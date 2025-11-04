"""
10/14/25
Provides tools to load data from an external file source
"""

import os
import json
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import pickle
import time


def load_oasis(filename: str = "brain") -> pd.DataFrame:
    """
    Loads OASIS data into a DataFrame from the location specified in localdata.json
    :param filename: *optional, default "brain"* The file name of the OASIS data file, in .mgz format
    :type filename: str
    :return: DataFrame containing the OASIS data
    :rtype: pd.DataFrame, columns=["name", "image"]
    """
    images = pd.DataFrame(columns=["name", "image"])

    with (open("localdata.json", 'r')) as json_file:
        oasis_path = json.load(json_file).get("oasis")

    data_path = os.path.join(oasis_path, "data")

    for folder in tqdm(os.listdir(data_path), desc="OASIS"):
        patient_dir = os.path.join(data_path, folder)
        if os.path.isdir(patient_dir):
            img = nib.load(os.path.join(patient_dir, f"mri/{filename}.mgz"))
            images.loc[len(images)] = [folder, pickle.dumps(img)]

    return images


def load_candi(filename: str = "procimg") -> pd.DataFrame:
    """
    Loads CANDI data into a DataFrame from the location specified in localdata.json
    :param filename: *optional, default "procimg"* The file name of the CANDI data file, in .nii.gz format
    :type filename: str
    :return: DataFrame containing the CANDI data
    :rtype: pd.DataFrame, columns=["name", "image"]
    """

    images = pd.DataFrame(columns=["name", "image"])

    with open("localdata.json", 'r') as json_file:
        candi_path = json.load(json_file).get("candi")

    data_path = os.path.join(candi_path, "data")

    bar_total = 0
    for p in os.listdir(data_path):
        bar_total += len(os.listdir(os.path.join(data_path, p)))
    bar = tqdm(total=bar_total, desc="CANDI")

    for folder in os.listdir(data_path):
        group_dir = os.path.join(data_path, folder)
        if os.path.isdir(group_dir):
            for patient_folder in os.listdir(group_dir):
                bar.update()
                patient_dir = os.path.join(group_dir, patient_folder)
                if os.path.isdir(patient_dir):
                    img = nib.load(os.path.join(patient_dir, f"{patient_folder}_{filename}.nii.gz"))
                    images.loc[len(images)] = [patient_folder, pickle.dumps(img)]

    return images

def load_data(sql_engine):
    images = pd.concat([load_oasis(), load_candi()])

    start_time = time.time()
    print("Writing to file...", end="", flush=True)
    images.to_sql(name="raw", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", time.time() - start_time, " sec)", sep="")

    return images