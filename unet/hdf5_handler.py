import pandas as pd
import pickle
import h5py
from tqdm import tqdm
from shutil import rmtree
from os import mkdir
import sqlalchemy
import numpy as np


def save_images(sql_engine: sqlalchemy.engine.base.Engine, table: str, save_path: str) -> None:
    """
    Saves images from a SQL table to HDF5 files as required by pytorch3dunet
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: The name of the table to read from
    :type table: str
    :param save_path: The path to save HDF5 files to
    :type save_path: str
    :return: None
    :rtype: NoneType
    """
    data = pd.read_sql_query(f"SELECT * FROM {table}", sql_engine)

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for row in tqdm(range(len(data)), desc="Saving images to HDF5"):
        raw = pickle.loads(data.loc[row, "image"]).get_fdata().astype(np.int_)
        label = pickle.loads(data.loc[row, "label"]).get_fdata().astype(np.int_)

        if not raw.shape == label.shape:
            raise Exception(f"Image shape {raw.shape} does not match label shape {label.shape}.")

        with h5py.File(f"{save_path}/{data.loc[row, "name"]}.hdf5", 'w') as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("label", data=label)
