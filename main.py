"""
11/4/25
"""
import sqlalchemy
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
from preprocessor import reorient, correct_bias_fields, correct_class_labels, impute_unknown, normalize, pad_images, \
    register_to_mni, scale_pad_labels
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

    load_all(sql_engine)
    preprocess(sql_engine)
    save_to_hdf5(sql_engine)
    train_model()

    logger.info(f"Finished in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def load_all(sql_engine: sqlalchemy.engine.base.Engine) -> None:
    """
    Load all OASIS and CANDI images into a SQL database
    :param sql_engine: Engine connected to the database to save images to
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """

    start_time = time.time()

    logger.info("Loading Images...")

    load_images(sql_engine, "brainmask", "raw_image")
    load_images(sql_engine, "aseg", "raw_label")

    logger.info(f"Finished loading in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def preprocess(sql_engine: sqlalchemy.engine.base.Engine) -> None:
    """
    Preprocessing pipeline. Steps are as follows:\n
    1. Pad images to a uniform size (256, 256, 256) by default\n
    2. Scale labels to the same resolution and pad to the same size\n
    3. Register all images and labels to the MNI template\n
    4. Correct bias fields in images\n
    5. Normalize image intensities using z-score standardization\n
    6. Impute unknown label values\n
    7. Correct class label values from the FreeSurfer format to a continuous format [0, 36]\n
    8. Reorient images so the correct side is up in their arrays\n
    :param sql_engine: Engine connected to the database to read and save images to
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """
    start_time = time.time()

    logger.info("Preprocessing Images...")

    pad_images(sql_engine, "raw_image", "padded_image")
    scale_pad_labels(sql_engine, "raw_label", "padded_label")
    register_to_mni(sql_engine, "padded_image", "padded_label", "mni_registered_image", "mni_registered_label")
    correct_bias_fields(sql_engine)
    normalize(sql_engine)
    impute_unknown(sql_engine)
    correct_class_labels(sql_engine)
    reorient(sql_engine, "normalized_image", "preprocessed_image")
    reorient(sql_engine, "corrected_label", "preprocessed_label")

    logger.info(f"Finished preprocessing in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")


def save_to_hdf5(sql_engine: sqlalchemy.engine.base.Engine) -> None:
    """
    Save images to the HDF5 format required by pytorch3dunet
    :param sql_engine: Engine connected to the database to save images to
    :type sql_engine: sqlalchemy.engine.base.Engine
    :return: None
    :rtype: NoneType
    """
    start_time = time.time()

    logger.info("Saving Images to HDF5...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_path = open_file["unet"]["train"]
        validation_path = open_file["unet"]["validation"]
        test_path = open_file["unet"]["test"]

    with sql_engine.connect() as conn:
        images = pd.read_sql_query("SELECT * FROM preprocessed_image", conn)
        labels = pd.read_sql_query("SELECT * FROM preprocessed_label", conn)

    images.sort_values(by=["name"], inplace=True, ignore_index=True)
    labels.sort_values(by=["name"], inplace=True, ignore_index=True)

    data = images.copy().reset_index(drop=True)
    data["label"] = labels["image"]

    train, temp = train_test_split(data, train_size=0.70)
    val, test = train_test_split(temp, train_size=0.50)

    train.to_sql(name="train", con=sql_engine, if_exists="replace", index=False)
    val.to_sql(name="validation", con=sql_engine, if_exists="replace", index=False)
    test.to_sql(name="testing", con=sql_engine, if_exists="replace", index=False)

    save_images(sql_engine, "train", train_path)
    save_images(sql_engine, "validation", validation_path)
    save_images(sql_engine, "testing", test_path)

    logger.info(f"Finished saving to HDF5 in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

def train_model() -> None:
    """
    Train the U-Net model
    :return: None
    :rtype: NoneType
    """
    start_time = time.time()

    logger.info("Training Model...")

    with open("config.yaml", "r") as f:
        open_file = yaml.safe_load(f)
        train_config_file = open_file["unet"]["train_config"]

    train_unet(train_config_file)

    logger.info(f"Finished training in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

if __name__ == "__main__":
    main()