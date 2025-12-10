"""
11/4/25
"""

import json
import time

import torch
from sqlalchemy import create_engine

from dataloader import load_data
from preprocessor import correct_bias_fields, map_to_mni, normalize, pad_images
from unet.hdf5_handler import save_train

if __name__ == "__main__":
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    with open("localdata.json") as json_file:
        open_file = json.load(json_file)
        database_location = open_file.get("database")
        train_config_file = open_file.get("train_config")

    engine = create_engine(f'sqlite:///{database_location}', echo=False)

    load_data(engine)
    pad_images(engine)
    correct_bias_fields(engine)
    normalize(engine)
    map_to_mni(engine)

    save_train(engine)

    train_unet(train_config_file)

    print("Finished in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))