from dataloader import load_data
import torch
import json
from sqlalchemy import create_engine
import pandas as pd

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    with open("localdata.json") as json_file:
        database_location = json.load(json_file)

    engine = create_engine(f'sqlite:///{database_location}', echo=False)

    load_data(engine)

    with engine.connect() as conn:
        top = pd.read_sql_query("SELECT * FROM images LIMIT 5", conn)

    print(top)