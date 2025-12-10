import pandas as pd
import pickle
import h5py
from tqdm import tqdm
from shutil import rmtree
from os import mkdir

def save_train(engine):
    save_path = "unet/train"
    data = pd.read_sql_query("SELECT * FROM preprocessed", engine)

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    for row in tqdm(data.itertuples(), total=len(data)):
        image = pickle.loads(row.image).get_fdata()
        with h5py.File(f"{save_path}/{row.name}.hdf5", 'w') as f:
            f.create_dataset('raw', data=image)