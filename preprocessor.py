"""
10/4/25
"""
from cProfile import label

import pandas as pd
import pickle
import time

import sqlalchemy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from nilearn import datasets, image, plotting

def normalize(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw") -> None:
    """
    Normalizes all data from a SQL table using the selected engine
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "raw"* The name of the table to normalize
    :type table: str
    :return: None
    """
    with (sql_engine.connect() as conn):
        data = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 5", conn)

        # def update_mean(mean, num_samples, new_num):
        #     return ((mean * (num_samples - 1)) + new_num) / num_samples
        #
        # def update_stdev(sum_of_squares, square_of_sum, num_samples):
        #     if num_samples == 1:
        #         return 0
        #     return np.sqrt((sum_of_squares / num_samples) - ((square_of_sum / num_samples) ** 2))
        #
        # num_batches = 16
        # batch_size = np.ceil(len(data) / num_batches)
        #
        # means = []
        # stdevs = []
        # batch_sizes = []
        #
        # for batch_num in range(num_batches):
        #     first = batch_num * batch_size
        #     last = (batch_num + 1) * batch_size - 1 if batch_num < num_batches - 1 else len(data) - 1
        #
        #     mean = 0
        #     stdev = 0
        #     num_samples = 0
        #     sum_of_squares = 0
        #     sum_of_samples = 0
        #
        #     for image in tqdm(data.loc[first:last, "image"], desc=f"Batch {batch_num}"):
        #         fdata = pickle.loads(image).get_fdata()
        #         for row in fdata:
        #             for col in row:
        #                 for val in col:
        #                     if val != 0:
        #                         num_samples += 1
        #                         sum_of_squares += val ** 2
        #                         sum_of_samples += val
        #                         mean = update_mean(mean, num_samples, val)
        #                         stdev = update_stdev(sum_of_squares, sum_of_samples, num_samples)
        #
        #     means.append(mean)
        #     stdevs.append(stdev)
        #     batch_sizes.append(last - first)
        #
        # mean = sum([means[i] * batch_sizes[i] for i in range(len(means))]) / len(data)
        # stdev =

        fig, ax = plt.subplots(2, 1)
        # ax[0].set_xlim(-0.5, 2)
        ax[1].set_xlim(-0.5, 2)

        for i, row in tqdm(data.iterrows(), total=len(data)):
            # print(row["name"])
            fdata = pickle.loads(row["image"]).get_fdata(caching="unchanged")
            orig_shape = fdata.shape
            masked_data = np.ma.masked_equal(fdata.flatten(), 0)
            # print(masked_data.min(), masked_data.max())
            counts, bins = np.histogram(masked_data.compressed(), bins=np.linspace(0, 125, 400))
            ax[0].stairs(counts, bins)
            mean = np.mean(masked_data)
            stdev = np.std(masked_data)
            # print(mean, stdev)
            # print(data)
            masked_data -= mean
            masked_data /= stdev
            # print(data)
            # print(data.shape)
            counts, bins = np.histogram(masked_data.compressed(), bins=np.linspace(-0.2, 12, 400))
            ax[1].stairs(counts, bins)
            # print(min(data), max(data))
            # print(counts, bins)

        plt.show()

        # norm_data = pd.DataFrame(columns=["name", "image"])
        #
        # for image_tuple in tqdm(data.itertuples(), desc="Normalizing images"):
        #     fdata = pickle.loads(image_tuple.image).get_fdata()
        #     for x in tqdm(range(len(fdata)), leave=False):
        #         for y in range(len(fdata[x])):
        #             for z in range(len(fdata[x][y])):
        #                 value = fdata[x][y][z]
        #                 if value != 0:
        #                     value -= mean
        #                     value /= stdev
        #                     fdata[x][y][z] = value
        #
        #     norm_data[len(norm_data)] = [image_tuple.name, fdata]
        #
        # start_time = time.time()
        # print("Writing to file...", end="", flush=True)
        # # norm_data.to_sql(name="preprocessed", con=sql_engine, if_exists="replace", index=False)
        # print(" DONE! (", np.round((time.time() - start_time) / 60, 2), " min)", sep="")

def correct_bias_fields(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw") -> None:
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    for image_tuple in tqdm(data.itertuples(), total=len(data)):
        fdata = pickle.loads(image_tuple.image).get_fdata()
        image = sitk.GetImageFromArray(fdata)
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

        corrected_image = corrector.Execute(image, mask_image)
        corrected_array = sitk.GetArrayFromImage(corrected_image)

        data.loc[image_tuple.Index, "image"] = corrected_array

    data.to_sql(name="preprocessed", con=sql_engine, if_exists="replace", index=False)

def map_to_mni(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw", dimensions: tuple = (256, 256, 256)) -> None:
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
