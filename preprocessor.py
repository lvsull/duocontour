"""
10/4/25
"""

from json import load
from os import mkdir, path
from pickle import dumps, loads
from shutil import rmtree

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import sqlalchemy
from nilearn import datasets
from tqdm import tqdm

from unet.unet_format_converter import fs_to_cont

bf = "{desc:<25}{percentage:3.0f}%|{bar:20}{r_bar}"


def save_and_get_view(arr, orig_image, save_path, name):
    corrected_image = nib.Nifti1Image(arr, orig_image.affine, orig_image.header)

    filepath = path.join(save_path, f"{name}.nii.gz")
    nib.save(corrected_image, filepath)

    return nib.load(filepath)


def center_pad(arr, dimensions=(256, 256, 256)):
    x_diff = dimensions[0] - arr.shape[0]
    y_diff = dimensions[1] - arr.shape[1]
    z_diff = dimensions[2] - arr.shape[2]
    return np.pad(arr, ((int(np.floor(x_diff / 2)), int(np.ceil(x_diff / 2))),
                        (int((np.floor(y_diff / 2))), int(np.ceil(y_diff / 2))),
                        (int((np.floor(z_diff / 2))), int(np.ceil(z_diff / 2)))),
                  mode="constant", constant_values=0)


def pad_images(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw", save_table="padded",
               dimensions=(256, 256, 256)) -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get(save_table)

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
        padded_arr = center_pad(fdata, dimensions)

        padded_image_view = save_and_get_view(padded_arr, orig_image, save_path, image_tuple.name)

        data.loc[image_tuple.Index, "image"] = dumps(padded_image_view)

    data.to_sql(name=save_table, con=sql_engine, if_exists="replace", index=False)


def correct_bias_fields(sql_engine: sqlalchemy.engine.base.Engine, table: str = "mni_registered") -> None:
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
        fdata = orig_image.get_fdata()
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
        save_path = load(json_file).get("preprocessed")

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

    data.to_sql(name="preprocessed", con=sql_engine, if_exists="replace", index=False)


def load_mni_template(save_path):
    dimensions = (256, 256, 256)
    mni_template_orig = datasets.load_mni152_template()
    mni_template_data = mni_template_orig.get_fdata()
    mni_shape = mni_template_data.shape
    x_diff = dimensions[0] - mni_shape[0]
    y_diff = dimensions[1] - mni_shape[1]
    z_diff = dimensions[2] - mni_shape[2]
    padded_mni_template_data = center_pad(mni_template_data, dimensions)
    orig_affine = mni_template_orig.affine
    mni_template = nib.Nifti1Image(padded_mni_template_data,
                                   affine=np.asarray([[1, 0, 0] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 1, 0] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 0, 1] + [orig_affine[0][3] - int(np.floor(x_diff / 2))],
                                                      [0, 0, 0, 1]]))

    nib.save(mni_template, save_path)

    return nib.load(save_path)

def register_to_mni(sql_engine, read_name, save_name):
    """
    Registers images to MNI space\n
    Adapted from https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html#overview
    :return:
    :rtype:
    """

    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {read_name}", conn)

    with open("localdata.json") as json_file:
        f = load(json_file)
        read_path = f.get(read_name)
        save_path = f.get(save_name)
        mni_template_path = f.get("mni_template")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    load_mni_template(mni_template_path)

    fixed = sitk.ReadImage(mni_template_path, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    def command_iteration(method):
        method.GetOptimizerIteration()
        method.GetMetricValue()
        method.GetOptimizerPosition()

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Registering to MNI", bar_format=bf):
        image_path = path.join(read_path, f"{image_tuple.name}.nii.gz")

        moving = sitk.ReadImage(image_path, sitk.sitkFloat32)

        outTx = R.Execute(fixed, moving)
        outTx.SetOffset([np.round(x) for x in outTx.GetOffset()])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        out = resampler.Execute(moving)

        moved_img_arr = sitk.GetArrayFromImage(out)

        img_save_path = path.join(save_path, f"{image_tuple.name}.nii.gz")

        sitk.WriteImage(sitk.GetImageFromArray(moved_img_arr), img_save_path)

        moved_image_view = nib.load(img_save_path)

        data.loc[image_tuple.Index, "image"] = dumps(moved_image_view)

    data.to_sql(name=save_name, con=sql_engine, if_exists="replace", index=False)


def impute_unknown(sql_engine, table="mni_registered_label"):
    """
    Impute unknown values from label data stored in a SQL table
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("imputed_label")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for subject in tqdm(data.itertuples(), total=len(data), desc="Imputing Unknown Labels", bar_format=bf):
        orig_image = loads(subject.image)
        fdata = np.round(orig_image.get_fdata())
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

                case 78 | 79 | 80 | 81 | 82:  # hypointensities
                    fdata[slc][row][col] = 77

                case 251 | 252 | 253 | 254 | 255:  # corpus callosum segmentation, should not exist
                    fdata[slc][row][col] = 0

        imputed_label_view = save_and_get_view(fdata, orig_image, save_path, subject.name)
        data.loc[subject.Index, "image"] = dumps(imputed_label_view)

    data.to_sql(name="imputed_label", con=sql_engine, if_exists="replace", index=False)

def correct_class_labels(sql_engine: sqlalchemy.engine.base.Engine, table="imputed_label") -> None:
    """
    Corrects class labes to be continuous [0, 36]
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    with open("localdata.json") as json_file:
        save_path = load(json_file).get("label")

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for subject in tqdm(data.itertuples(), total=len(data), desc="Correcting Label Values", bar_format=bf):
        orig_image = loads(subject.image)
        fdata = np.round(orig_image.get_fdata())
        for slc, row, col in np.argwhere(fdata != 0):
            fdata[slc][row][col] = fs_to_cont(fdata[slc][row][col])

        corrected_label_view = save_and_get_view(fdata, orig_image, save_path, subject.name)
        data.loc[subject.Index, "image"] = dumps(corrected_label_view)

    data.to_sql(name="label", con=sql_engine, if_exists="replace", index=False)