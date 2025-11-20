"""
11/4/25
"""

import json
import torch
from sqlalchemy import create_engine
from dataloader import load_data
from preprocessor import normalize, correct_bias_fields, map_to_mni

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

    # load_data(engine)

    correct_bias_fields(engine)
    # normalize(engine)
    # map_to_mni(engine)