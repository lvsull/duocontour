import pandas as pd
import pickle
import h5py
from tqdm import tqdm
from shutil import rmtree
from os import mkdir
import sqlalchemy


def save_images(sql_engine: sqlalchemy.engine.base.Engine, save_path: str) -> None:
    """
    Saves images and label data to HDF5 files as required by pytorch3dunet
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param save_path: The directory to save HDF5 tables to
    :type save_path: str
    :return: None
    :rtype: NoneType
    """
    raw_data = pd.read_sql_query("SELECT * FROM preprocessed", sql_engine)
    label_data = pd.read_sql_query("SELECT * FROM label", sql_engine)

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for row in tqdm(range(len(raw_data)), desc="Saving images to HDF5"):
        raw = pickle.loads(raw_data.loc[row, "image"]).get_fdata()
        label = pickle.loads(label_data.loc[row, "image"]).get_fdata()

        if not raw.shape == label.shape:
            raise Exception("Some images do not have the same shape")

        with h5py.File(f"{save_path}/{raw_data.loc[row, "name"]}.hdf5", 'w') as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("label", data=label)
