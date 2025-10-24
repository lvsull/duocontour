import os
import json
import nibabel as nib
from tqdm import tqdm

def load_oasis(filename = "brain"):

    images = {}

    with (open("localdata.json", 'r')) as json_file:
        oasis_path = json.load(json_file).get("oasis")

    data_path = os.path.join(oasis_path, "data")

    for folder in tqdm(os.listdir(data_path), desc="OASIS"):
        patient_dir = os.path.join(data_path, folder)
        if os.path.isdir(patient_dir):
            img = nib.load(os.path.join(patient_dir, f"mri/{filename}.mgz"))
            images[folder] = img

    return images

def load_candi(filename = "procimg"):

    images = {}

    with open("localdata.json", 'r') as json_file:
        candi_path = json.load(json_file).get("candi")

    data_path = os.path.join(candi_path, "data")

    bar_total = 0
    for p in os.listdir(data_path):
        bar_total += len(os.listdir(os.path.join(data_path, p)))
    bar = tqdm(total=bar_total, desc="CANDI")

    for folder in os.listdir(data_path):
        group_dir = os.path.join(data_path, folder)
        if os.path.isdir(group_dir):
            for patient_folder in os.listdir(group_dir):
                bar.update()
                patient_dir = os.path.join(group_dir, patient_folder)
                if os.path.isdir(patient_dir):
                    img = nib.load(os.path.join(patient_dir, f"{patient_folder}_{filename}.nii.gz"))
                    images[patient_folder] = img

    return images