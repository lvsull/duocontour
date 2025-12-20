import pandas as pd
import pickle
import h5py
from tqdm import tqdm
from shutil import rmtree
from os import mkdir
import sqlalchemy
import numpy as np


bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

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

    for row in tqdm(range(len(data)), desc=f"Saving {table} to HDF5", bar_format=bf):
        raw = pickle.loads(data.loc[row, "image"]).get_fdata()
        label = pickle.loads(data.loc[row, "label"]).get_fdata()

        if not raw.shape == label.shape:
            raise Exception(f"Image shape {raw.shape} does not match label shape {label.shape}.")

        with h5py.File(f"{save_path}/{data.loc[row, "name"]}.hdf5", 'w') as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("label", data=label)


c_to_f = pd.read_csv("seg_values.csv")["SegID"].to_dict()
f_to_c = {y: x for x, y in c_to_f.items()}


def fs_to_cont(value: int) -> int:
    """
    Convert a label value in FreeSurfer format to continuous format
    :param value: The FreeSurfer label value to convert
    :type value: int
    :return: ``value`` converted to continuous format
    :rtype: int
    """
    try:
        return f_to_c[int(value)]
    except KeyError:
        return 0


def cont_to_fs(value):
    """
    Convert a label value in continuous format to FreeSurfer format
    :param value: The continuous label value to convert
    :type value: int
    :return: ``value`` converted to FreeSurfer format
    :rtype: int
    """
    try:
        return c_to_f[int(value)]
    except KeyError:
        return 0