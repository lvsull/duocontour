"""
10/4/25
"""

import pandas as pd
import pickle
import time

import sqlalchemy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def normalize(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw") -> None:
    """
    Normalizes all data from a SQL table using the selected engine
    :param sql_engine: The SQLAlchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "raw"* The name of the table to normalize
    :type table: str
    :return: None
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", conn)

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

        for image in data.loc[:, "image"]:
            fdata = pickle.loads(image).get_fdata(caching="unchanged")
            orig_shape = fdata.shape
            data = fdata.flatten()
            mean = np.mean(data)
            stdev = np.std(data)
            data = np.array(list(map(lambda x: ((x - mean) / stdev) * 10, data)))
            hist = np.histogram(data, bins=np.linspace(-3, 115, 100))
            print(min(data), max(data))
            print(hist)
            plt.hist(hist[0], hist[1], alpha=0.5)

        plt.show()

        print(mean)
        print(stdev)

        norm_data = pd.DataFrame(columns=["name", "image"])

        for image_tuple in tqdm(data.itertuples(), desc="Normalizing images"):
            fdata = pickle.loads(image_tuple.image).get_fdata()
            for x in range(len(fdata)):
                for y in range(len(fdata[x])):
                    for z in range(len(fdata[x][y])):
                        if fdata[x][y][z] != 0:
                            fdata[x][y][z] = (fdata[x][y][z] - mean) / stdev

                            norm_data[len(norm_data)] = [image_tuple.name, fdata[x][y][z]]

        start_time = time.time()
        print("Writing to file...", end="", flush=True)
        norm_data.to_sql(name="normalized", con=sql_engine, if_exists="replace", index=False)
        print(" DONE! (", np.round((time.time() - start_time) / 60, 2), " min)", sep="")