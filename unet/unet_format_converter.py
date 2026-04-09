import pickle
from os import mkdir
from shutil import rmtree

import h5py
import pandas as pd
import sqlalchemy
import yaml
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
# from preprocessor import AFFINE
from pathlib import Path


bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

AFFINE = np.array([[-1, 0, 0, 128],
                   [0, 0, 1, -128],
                   [0, -1, 0, 128],
                   [0, 0, 0, 1]])


def images_to_hdf5(sql_engine: sqlalchemy.engine.base.Engine, table: str, save_path: str) -> None:
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


def hdf5_to_images(sql_engine: sqlalchemy.engine.base.Engine, table: str, save_path: str) -> None:
    """
    Converts HDF5 tables to Nifti images using Nibabel
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: The name of the table to save to
    :type table: str
    :param save_path: The path to save Nifti images to
    :type save_path: str
    :return: None
    :rtype: NoneType
    """
    with open("config.yaml", 'r') as f:
        test_conf = yaml.safe_load(f)["unet"]["test_config"]
        with open(test_conf, "r") as tc:
            read_path = Path(yaml.safe_load(tc)["predictor"]["pred_path"])

    if save_path is not Path:
        save_path = Path(save_path)

    data = pd.DataFrame(columns=["name", "image"])

    for filepath in read_path.iterdir():
        hf = h5py.File(filepath, 'r')
        fdata = hf.get("predictions")[()]
        img = nib.Nifti1Image(fdata, AFFINE)
        nib.save(img, (save_path / filepath.stem).with_suffix(".nii.gz"))
        data.loc[len(data)] = [filepath.stem, pickle.dumps(img)]

    data.to_sql(name=table, con=sql_engine, index=False, if_exists="replace")

seg_values = pd.read_csv("seg_values.csv")
c_to_f = seg_values["SegID"].to_dict()
f_to_c = {y: x for x, y in c_to_f.items()}
c_to_s = seg_values["SingleID"].to_dict()
f_to_s = pd.read_csv("seg_values.csv", index_col="SegID")["SingleID"].to_dict()


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


def cont_to_fs(value: int) -> int:
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


def cont_to_single(value):
    try:
        return c_to_s[int(value)]
    except KeyError:
        return 0


def fs_to_single(value):
    try:
        return f_to_s[int(value)]
    except KeyError:
        return 0


if __name__ == "__main__":
    # files = list(Path('unet/pred').rglob('*.hdf5'))
    # for file in tqdm(files, total=len(files)):
    #     file = str(file)
    #     with h5py.File(file, 'r') as f:
    #         label = f.get("predictions")[()]
    #     label = np.vectorize(cont_to_single)(label)
    #     label = label.astype(np.uint32)
    #     with h5py.File(file, "w") as f:
    #         f.create_dataset("predictions", data=label)

    hdf5_to_images(sqlalchemy.create_engine(f'sqlite:///{r"D:\Liam Sullivan LTS\images.db"}', echo=False), "predictions", "output/pred")