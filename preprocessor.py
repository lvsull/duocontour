"""
10/4/25
"""

import pandas as pd
import pickle
import time

import sqlalchemy
from tqdm import tqdm
import numpy as np

def normalize(sql_engine: sqlalchemy.engine.base.Engine, table: str = "raw") -> None:
    """
    Normalizes all data from a SQL table using the selected engine
    :param sql_engine: The SQLALchemy engine to use
    :type sql_engine: sqlalchemy.engine.base.Engine
    :param table: *optional, default "raw"* The name of the table to normalize
    :type table: str
    :return: None
    """
    with sql_engine.connect() as conn:
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        all_values = []

        for image in tqdm(data.loc[:, "image"], desc="Calculating mean & SD"):
            fdata = pickle.loads(image).get_fdata()
            for row in fdata:
                for col in row:
                    for val in col:
                        if val != 0:
                            all_values += val

        mean = float(np.mean(all_values))
        stdev = float(np.std(all_values))

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
        norm_data.to_sql(name="raw", con=sql_engine, if_exists="replace", index=False)
        print(" DONE! (", time.time() - start_time, " sec)", sep="")