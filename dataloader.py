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
import sqlalchemy

bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

def load_images(sql_engine: sqlalchemy.engine.Engine, filename: str = "brainmask", save_table: str = "raw") -> None:
    """
    Loads all image data into a DataFrame from the location specified in localdata.json
    :param sql_engine: SQLAlchemy engine object for the database file
    :type sql_engine: sqlalchemy.engine.Engine
    :param filename: *optional, default "brain"* The file name of the OASIS data file, in .mgz format
    :type filename: str
    :param save_table: *optional, default "raw"* The table name where the data will be saved
    :type save_table: str
    :return: DataFrame containing the image data
    :rtype: pd.DataFrame, columns=["name", "image"]
    """
    images = pd.DataFrame(columns=["name", "image"])

    for dataset_name in ("OASIS", "CANDI"):
        with (open("localdata.json", 'r')) as json_file:
            data_path = os.path.join(json.load(json_file).get("raw"), dataset_name)

        data_path = os.path.join(data_path, "data")

        for folder in tqdm(os.listdir(data_path),
                           desc=f"Loading {dataset_name} {"Images" if filename == "brainmask" else "Labels"}",
                           bar_format=bf):
            patient_dir = os.path.join(data_path, folder)
            if os.path.isdir(patient_dir):
                img = nib.load(os.path.join(patient_dir, f"mri/{filename}.mgz"))
                images.loc[len(images)] = [folder, pickle.dumps(img)]

    images.to_sql(name=save_table, con=sql_engine, if_exists="replace", index=False)