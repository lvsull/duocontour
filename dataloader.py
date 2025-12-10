"""
10/14/25
Provides tools to load data from an external file source
"""

import json
import os
import pickle
from time import time

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(sql_engine, filename: str = "brainmask") -> None:
    """
    Loads OASIS data into a DataFrame from the location specified in localdata.json
    :param filename: *optional, default "brain"* The file name of the OASIS data file, in .mgz format
    :type filename: str
    :return: DataFrame containing the OASIS data
    :rtype: pd.DataFrame, columns=["name", "image"]
    """
    images = pd.DataFrame(columns=["name", "image"])

    for dataset_name in ("OASIS", "CANDI"):
        with (open("localdata.json", 'r')) as json_file:
            data_path = os.path.join(json.load(json_file).get("raw"), dataset_name)

        data_path = os.path.join(data_path, "data")

        for folder in tqdm(os.listdir(data_path), desc=dataset_name):
            patient_dir = os.path.join(data_path, folder)
            if os.path.isdir(patient_dir):
                img = nib.load(os.path.join(patient_dir, f"mri/{filename}.mgz"))
                images.loc[len(images)] = [folder, pickle.dumps(img)]

    images.to_sql(name="raw", con=sql_engine, if_exists="replace", index=False)