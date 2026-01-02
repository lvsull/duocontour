"""
11/4/25
"""

import json
import time

import pandas as pd
import torch
from pytorch3dunet.train import main as train_unet
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from dataloader import load_images
from preprocessor import correct_bias_fields, correct_class_labels, impute_unknown, normalize, pad_images, \
    register_to_mni, scale_pad_label
from unet.unet_format_converter import save_images

if __name__ == "__main__":
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    device = torch.device("cpu")
    print("Using device: CPU")
    with open("localdata.json", "r") as f:
        open_file = json.load(f)
        database_location = open_file.get("database")
        train_config_file = open_file.get("train_config")
        train_path = open_file.get("train")
        validation_path = open_file.get("validation")

    engine = create_engine(f'sqlite:///{database_location}', echo=False)

    print("\nLoading Images...")

    load_images(engine, "brainmask", "raw")
    load_images(engine, "aseg", "raw_label")

    print("\nPreprocessing Images...")

    pad_images(engine, "raw", "padded")
    scale_pad_label(engine, "raw_label", "padded_label")
    register_to_mni(engine, "padded", "padded_label", "mni_registered", "mni_registered_label")
    correct_bias_fields(engine)
    normalize(engine)
    impute_unknown(engine)
    correct_class_labels(engine)

    print("\nSaving Images to HDF5...")

    with engine.connect() as conn:
        images = pd.read_sql_query("SELECT * FROM preprocessed", conn)
        labels = pd.read_sql_query("SELECT * FROM label", conn)

    images.sort_values(by=["name"], inplace=True, ignore_index=True)
    labels.sort_values(by=["name"], inplace=True, ignore_index=True)

    data = images.copy().reset_index(drop=True)
    data["label"] = labels["image"]

    train, val = train_test_split(data, test_size=0.15)

    train.to_sql(name="train", con=engine, if_exists="replace", index=False)
    val.to_sql(name="validation", con=engine, if_exists="replace", index=False)

    save_images(engine, "train", train_path)
    save_images(engine, "validation", validation_path)

    print("\nFinished preprocessing in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print("\nTraining model...")

    train_unet(train_config_file)

    print("Finished in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))