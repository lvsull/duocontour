"""
10/4/25
"""

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

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Padding Images"):
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

    start_time = time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="padded", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round(time() - start_time, 2), " sec)", sep="")


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

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Correcting Bias Fields"):
        orig_image = loads(image_tuple.image)
        fdata = orig_image.get_fdata(caching="unchanged")
        image = sitk.GetImageFromArray(fdata)
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

        corrected_image_itk = corrector.Execute(image, mask_image)
        corrected_array = sitk.GetArrayFromImage(corrected_image_itk)

        corrected_image_view = save_and_get_view(corrected_array, orig_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(corrected_image_view)

    start_time = time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="bias_corrected", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round(time() - start_time, 2), " sec)", sep="")


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

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Normalizing Images"):
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

    start_time = time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="normalized", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round(time() - start_time, 2), " sec)", sep="")


def map_to_mni(sql_engine: sqlalchemy.engine.base.Engine, table: str = "normalized",
               dimensions: tuple = (256, 256, 256)) -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("preprocessed")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    mni_template_orig = datasets.load_mni152_template()
    mni_template_data = mni_template_orig.get_fdata()
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

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Mapping to MNI Space"):
        subject_image = loads(image_tuple.image)
        mapped_image = image.resample_to_img(subject_image, mni_template, force_resample=True, copy_header=True)
        image_df.loc[len(image_df)] = [image_tuple.name, mapped_image.get_fdata()]

        mapped_image_view = save_and_get_view(mapped_image.get_fdata(), mapped_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(mapped_image_view)

    start_time = time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="preprocessed", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round(time() - start_time, 2), " sec)", sep="")