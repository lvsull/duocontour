"""
10/4/25
"""

import pandas as pd
import pickle
import time

import sqlalchemy
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from nilearn import datasets, image, plotting

def correct_bias_fields(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw") -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Correcting Bias Fields"):
        orig_image = pickle.loads(image_tuple.image)
        fdata = orig_image.get_fdata(caching="unchanged", dtype=np.float32)
        image = sitk.GetImageFromArray(fdata)
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

        corrected_image_itk = corrector.Execute(image, mask_image)
        corrected_array = sitk.GetArrayFromImage(corrected_image_itk)

        corrected_image = nib.freesurfer.mghformat.MGHImage(corrected_array, orig_image.affine, orig_image.header)

        data.loc[image_tuple.Index, "image"] = pickle.dumps(corrected_image)

    start_time = time.time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="bias_corrected", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round((time.time() - start_time) / 60, 2), " min)", sep="")

def normalize(sql_engine: sqlalchemy.engine.base.Engine, table: str = "bias_corrected") -> None:
    """
    Normalizes all data from a SQL table using the selected engine
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "raw"* The name of the table to normalize
    :type table: str
    :return: None
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    print(len(data))

    for image_tuple in tqdm(data.itertuples(), total=len(data), desc="Normalizing Images"):
        fdata = pickle.loads(image_tuple.image).get_fdata(caching="unchanged")
        orig_shape = fdata.shape
        masked_data = np.ma.masked_equal(fdata.flatten(), 0)
        mean = np.mean(masked_data)
        stdev = np.std(masked_data)
        masked_data -= mean
        masked_data /= stdev

        norm_image = masked_data.reshape(orig_shape).filled(fill_value=0)
        data.loc[image_tuple.Index, "image"] = norm_image

    start_time = time.time()
    print(f"Writing {len(data)} rows to file...", end="", flush=True)
    data.to_sql(name="normalized", con=sql_engine, if_exists="replace", index=False)
    print(" DONE! (", np.round((time.time() - start_time) / 60, 2), " min)", sep="")

def map_to_mni(sql_engine: sqlalchemy.engine.base.Engine, table: str = "normalized",
               dimensions: tuple = (256, 256, 256)) -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 3", conn)

    mni_template_orig = datasets.load_mni152_template()
    mni_template_data = mni_template_orig.get_fdata()
    mni_shape = mni_template_data.shape
    x_diff = dimensions[0] - mni_shape[0]
    y_diff = dimensions[1] - mni_shape[1]
    z_diff = dimensions[2] - mni_shape[2]
    padded_mni_template_data = np.pad(mni_shape, ((int(np.floor(x_diff / 2)), int(np.ceil(x_diff / 2))),
                                                  int((np.floor(y_diff / 2))), int(np.ceil(y_diff / 2)),
                                                  int((np.floor(z_diff / 2))), int(np.ceil(z_diff / 2))),
                                      mode="constant", constant_values=0)
    mni_template = nib.Nifti1Image(padded_mni_template_data, mni_template_orig.affine)

    plotting.plot_img(mni_template, cut_coords=(0, 0, 0))

    image_df = pd.DataFrame(columns=["name", "image"])

    for image_tuple in data.itertuples():
        subject_image = pickle.loads(image_tuple.image)
        mapped_image = image.resample_to_img(subject_image, mni_template, force_resample=True, copy_header=True)
        image_df.loc[len(image_df)] = [image_tuple.name, mapped_image.get_fdata()]
        plotting.plot_img(subject_image, cut_coords=(0, 0, 0))
        plotting.plot_img(mapped_image, cut_coords=(0, 0, 0))

    plotting.show()
    print()
