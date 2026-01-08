"""
11/4/25
"""

import yaml
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


def main():
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    with open("config.yaml", "r") as f:
        database_location = yaml.safe_load(f).get("database")

    sql_engine = create_engine(f'sqlite:///{database_location}', echo=False)

    load_all(sql_engine)
    preprocess(sql_engine)
    save_to_hdf5(sql_engine)
    train_model()

    print("Finished in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def load_all(sql_engine):
    start_time = time.time()

    print("\nLoading Images...")

    load_images(sql_engine, "brainmask", "raw")
    load_images(sql_engine, "aseg", "raw_label")

    print("Finished loading in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def preprocess(sql_engine):
    start_time = time.time()

    print("\nPreprocessing Images...")

    pad_images(sql_engine, "raw", "padded")
    scale_pad_label(sql_engine, "raw_label", "padded_label")
    register_to_mni(sql_engine, "padded", "padded_label", "mni_registered", "mni_registered_label")
    correct_bias_fields(sql_engine)
    normalize(sql_engine)
    impute_unknown(sql_engine)
    correct_class_labels(sql_engine)

    print("Finished preprocessing in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

def save_to_hdf5(sql_engine):
    start_time = time.time()

    print("\nSaving Images to HDF5...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_config_file = open_file.get("train_config")
        train_path = open_file.get("train")
        validation_path = open_file.get("validation")

    with sql_engine.connect() as conn:
        images = pd.read_sql_query("SELECT * FROM preprocessed", conn)
        labels = pd.read_sql_query("SELECT * FROM label", conn)

    images.sort_values(by=["name"], inplace=True, ignore_index=True)
    labels.sort_values(by=["name"], inplace=True, ignore_index=True)

    data = images.copy().reset_index(drop=True)
    data["label"] = labels["image"]

    train, val = train_test_split(data, test_size=0.15)

    train.to_sql(name="train", con=sql_engine, if_exists="replace", index=False)
    val.to_sql(name="validation", con=sql_engine, if_exists="replace", index=False)

    save_images(sql_engine, "train", train_path)
    save_images(sql_engine, "validation", validation_path)

    print("Finished saving to HDF5 in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

def train_model():
    start_time = time.time()

    print("\nTraining Model...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_config_file = open_file.get("train_config")

    train_unet(train_config_file)

    print("Finished training in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

if __name__ == "__main__":
    main()