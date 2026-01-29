"""
11/4/25
"""

import yaml
import time
import logging
import sys

import pandas as pd
import torch
from pytorch3dunet.train import main as train_unet
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from dataloader import load_images
from preprocessor import correct_bias_fields, correct_class_labels, impute_unknown, normalize, pad_images, \
    register_to_mni, scale_pad_label
from unet.unet_format_converter import save_images

logger = logging.getLogger("DuoContour")

def main():
    start_time = time.time()

    with open("config.yaml", "r") as f:
        yaml_file = yaml.safe_load(f)
        database_location = yaml_file["database"]
        log_file = yaml_file["log_file"]

    with open(log_file, 'w'):
        pass

    log_format = "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"

    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device:" + torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")

    sql_engine = create_engine(f'sqlite:///{database_location}', echo=False)

    # load_all(sql_engine)
    # preprocess(sql_engine)
    # save_to_hdf5(sql_engine)
    train_model()

    logger.info(f"Finished in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def load_all(sql_engine):
    start_time = time.time()

    logger.info("Loading Images...")

    load_images(sql_engine, "brainmask", "raw_image")
    load_images(sql_engine, "aseg", "raw_label")

    logger.info(f"Finished loading in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def preprocess(sql_engine):
    start_time = time.time()

    logger.info("Preprocessing Images...")

    pad_images(sql_engine, "raw_image", "padded_image")
    scale_pad_label(sql_engine, "raw_label", "padded_label")
    register_to_mni(sql_engine, "padded_image", "padded_label", "mni_registered_image", "mni_registered_label")
    correct_bias_fields(sql_engine)
    normalize(sql_engine)
    impute_unknown(sql_engine)
    correct_class_labels(sql_engine)

    logger.info(f"Finished preprocessing in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def save_to_hdf5(sql_engine):
    start_time = time.time()

    logger.info("Saving Images to HDF5...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_path = open_file["unet"]["train"]
        testing_path = open_file["unet"]["validation"]

    with sql_engine.connect() as conn:
        images = pd.read_sql_query("SELECT * FROM preprocessed_image", conn)
        labels = pd.read_sql_query("SELECT * FROM preprocessed_label", conn)

    images.sort_values(by=["name"], inplace=True, ignore_index=True)
    labels.sort_values(by=["name"], inplace=True, ignore_index=True)

    data = images.copy().reset_index(drop=True)
    data["label"] = labels["image"]

    train_test, val = train_test_split(data, test_size=0.15)
    train, test = train_test_split(train_test, test_size=0.15)

    train.to_sql(name="train", con=sql_engine, if_exists="replace", index=False)
    test.to_sql(name="testing", con=sql_engine, if_exists="replace", index=False)

    save_images(sql_engine, "train", train_path)
    save_images(sql_engine, "testing", testing_path)

    logger.info(f"Finished saving to HDF5 in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

def train_model():
    start_time = time.time()

    logger.info("Training Model...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_config_file = open_file["unet"]["train_config"]

    train_unet(train_config_file)

    logger.info(f"Finished training in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

if __name__ == "__main__":
    main()