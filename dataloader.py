import os
import json
import nibabel as nib
from tqdm import tqdm
import sys

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

    print("OASIS images loaded (", sys.getsizeof(images), " bytes)", sep="")

    return images

def load_candi(filename = "procimg"):

    images = {}

    with open("localdata.json", 'r') as json_file:
        candi_path = json.load(json_file).get("candi")

    data_path = os.path.join(candi_path, "data")

    for folder in tqdm(os.listdir(data_path), desc="CANDI"):
        group_dir = os.path.join(data_path, folder)
        if os.path.isdir(group_dir):
            for patient_folder in os.listdir(group_dir):
                patient_dir = os.path.join(group_dir, patient_folder)
                if os.path.isdir(patient_dir):
                    img = nib.load(os.path.join(patient_dir, f"{patient_folder}_{filename}.nii.gz"))
                    images[patient_folder] = img

    print("CANDI images loaded (", sys.getsizeof(images), " bytes)", sep="")

    return images


if __name__ == "__main__":
    images = load_oasis() | load_candi()

    print("All images loaded (", sys.getsizeof(images), " bytes)", sep="")
