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
import numpy as np


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

    start_time = time.time()
    print("Writing to file...", end="", flush=True)
    images.to_sql(name="raw", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round(time.time() - start_time, 3), " sec)", sep="")