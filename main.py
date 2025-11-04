"""
11/4/25
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
from dataloader import load_data
from sqlalchemy import create_engine
from tqdm import tqdm
from preprocessor import normalize

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    with open("localdata.json") as json_file:
        database_location = json.load(json_file).get("database")

    engine = create_engine(f'sqlite:///{database_location}', echo=False)

    load_data(engine)

    normalize(engine)