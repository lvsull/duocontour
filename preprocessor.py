"""
10/4/25
"""
import copy

import pandas as pd
from pickle import dumps, loads
from time import time
from os import path, mkdir
from shutil import rmtree
from json import load
import sqlalchemy
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from nilearn import datasets, image

bf = "{desc:<25}{percentage:3.0f}%|{bar:20}{r_bar}"

def save_and_get_view(arr, orig_image, save_path, name):
    corrected_image = nib.Nifti1Image(arr, orig_image.affine, orig_image.header)

    filepath = path.join(save_path, f"{name}.nii.gz")
    nib.save(corrected_image, filepath)

    return nib.load(filepath)


def pad_images(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw", dimensions=(256, 256, 256)) -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("padded")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Padding Images", bar_format=bf):
        orig_image = loads(image_tuple.image)
        fdata = orig_image.get_fdata(caching="unchanged")
        shape = fdata.shape
        x_diff = dimensions[0] - shape[0]
        y_diff = dimensions[1] - shape[1]
        z_diff = dimensions[2] - shape[2]
        padded_arr = np.pad(fdata, ((int(np.floor(x_diff / 2)), int(np.ceil(x_diff / 2))),
                                    (int((np.floor(y_diff / 2))), int(np.ceil(y_diff / 2))),
                                    (int((np.floor(z_diff / 2))), int(np.ceil(z_diff / 2)))))

        padded_image_view = save_and_get_view(padded_arr, orig_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(padded_image_view)

    data.to_sql(name="padded", con=sql_engine, if_exists="replace", index=False)


def correct_bias_fields(sql_engine: sqlalchemy.engine.base.Engine, table: str = "padded") -> None:
    """
    Corrects bias fields of images stored on a SQL table
    :param sql_engine: The SLQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "raw"* The name of the table to use
    :type table: str
    :return: None
    :rtype: NoneType
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("bias_corrected")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Correcting Bias Fields", bar_format=bf):
        orig_image = loads(image_tuple.image)
        fdata = orig_image.get_fdata(caching="unchanged")
        image = sitk.GetImageFromArray(fdata)
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

        corrected_image_itk = corrector.Execute(image, mask_image)
        corrected_array = sitk.GetArrayFromImage(corrected_image_itk)

        corrected_image_view = save_and_get_view(corrected_array, orig_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(corrected_image_view)

    data.to_sql(name="bias_corrected", con=sql_engine, if_exists="replace", index=False)


def normalize(sql_engine: sqlalchemy.engine.base.Engine, table: str = "bias_corrected") -> None:
    """
    Normalizes all data from a SQL table using the selected engine
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "bias_corrected"* The name of the table to normalize
    :type table: str
    :return: None
    :rtype: NoneType
    """

    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("normalized")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Normalizing Images", bar_format=bf):
        orig_image = loads(image_tuple.image)
        fdata = orig_image.get_fdata(caching="unchanged")
        orig_shape = fdata.shape
        masked_data = np.ma.masked_equal(fdata.flatten(), 0)
        mean = np.mean(masked_data)
        stdev = np.std(masked_data)
        masked_data -= mean
        masked_data /= stdev

        norm_image_arr = masked_data.reshape(orig_shape).filled(fill_value=0)

        norm_image_view = save_and_get_view(norm_image_arr, orig_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(norm_image_view)

    data.to_sql(name="normalized", con=sql_engine, if_exists="replace", index=False)


def map_to_mni(sql_engine: sqlalchemy.engine.base.Engine, table: str = "normalized",
               dimensions: tuple = (256, 256, 256), save_name: str = "preprocessed") -> None:
    """
    Map images from a SQL table to the MNI152 space
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "normalized"* The SQL table to read images from
    :type table: str
    :param dimensions: *optional, default (256, 256, 256)* The dimensions of the final image
    :type dimensions: (int, int, int)
    :param save_name: *optional, default "preprocessed"* The name of the location to save images to on disk and in the SQL database
    :type save_name: str
    :return: None
    :rtype: NoneType
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get(save_name)

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    mni_template_orig = datasets.load_mni152_template()
    mni_template_data = mni_template_orig.get_fdata().astype(np.int_)
    mni_shape = mni_template_data.shape
    x_diff = dimensions[0] - mni_shape[0]
    y_diff = dimensions[1] - mni_shape[1]
    z_diff = dimensions[2] - mni_shape[2]
    padded_mni_template_data = np.pad(mni_template_data, ((int(np.floor(x_diff / 2)), int(np.ceil(x_diff / 2))),
                                                          (int((np.floor(y_diff / 2))), int(np.ceil(y_diff / 2))),
                                                          (int((np.floor(z_diff / 2))), int(np.ceil(z_diff / 2)))),
                                      mode="constant", constant_values=0)
    orig_affine = mni_template_orig.affine
    mni_template = nib.Nifti1Image(padded_mni_template_data,
                                   affine=np.asarray([[1, 0, 0] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 1, 0] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 0, 1] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 0, 0, 1]]))

    image_df = pd.DataFrame(columns=["name", "image"])

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Mapping to MNI Space", bar_format=bf):
        subject_image = loads(image_tuple.image)
        mapped_image = image.resample_to_img(subject_image, mni_template, force_resample=True, copy_header=True)
        image_df.loc[len(image_df)] = [image_tuple.name, mapped_image.get_fdata().astype(np.int_)]

        mapped_image_view = save_and_get_view(mapped_image.get_fdata().astype(np.int_),
                                              mapped_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(mapped_image_view)

    data.to_sql(name=save_name, con=sql_engine, if_exists="replace", index=False)

def impute_unknown(sql_engine):
    """
    Impute unknown values from label data stored in a SQL table
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM raw_label", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("imputed_label")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for subject in tqdm(data.itertuples(), total=len(data), desc="Imputing Unknown Values", bar_format=bf):
        orig_image = loads(subject.image)
        fdata = orig_image.get_fdata().astype(np.int_)
        for slc, row, col in np.argwhere((fdata == 29) | (fdata == 61)
                                         | (fdata == 78) | (fdata == 79) | (fdata == 80) | (fdata == 81) | (fdata == 82)
                                         | (fdata == 251) | (fdata == 252) | (fdata == 253)
                                         | (fdata == 254) | (fdata == 255)):
            match fdata[slc][row][col]:
                case 29 | 61:  # unknown value
                    values = []
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            val = fdata[slc][row + y][col + x]
                            if not (x == 0 and y == 0) and not (val == 29 or val == 61):
                                values.append(val)
                    fdata[slc][row][col] = max(values, key=values.count)
                    break
                case 78 | 79 | 80 | 81 | 82:  # hypointensities
                    fdata[slc][row][col] = 77
                    break
                case 251 | 252 | 253 | 254 | 255:  # corpus callosum segmentation, should not exist
                    raise ValueError("CC value encountered")

        imputed_label_view = save_and_get_view(fdata, orig_image, save_path, subject.name)
        data.loc[subject.Index, "image"] = dumps(imputed_label_view)

    data.to_sql(name="imputed_label", con=sql_engine, if_exists="replace", index=False)

def correct_class_labels(sql_engine):
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM imputed_label", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("corrected_label")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    labels = pd.read_csv("seg_values.csv")["SegID"]

    for subject in tqdm(data.itertuples(), total=len(data), desc="Imputing Unknown Values", bar_format=bf):
        orig_image = loads(subject.image)
        fdata = orig_image.get_fdata().astype(np.int_)
        for slc, row, col in np.argwhere(fdata != 0):
            fdata[slc][row][col] = labels[labels.isin([fdata[slc][row][col]])].index[0]