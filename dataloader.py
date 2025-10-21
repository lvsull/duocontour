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
            if img.get_fdata(caching="unchanged").shape != (256, 256, 256):
                raise Exception("Unexpected shape:", img.get_fdata(caching="unchanged").shape)
            else:
                images[folder] = img

    print("OASIS images loaded (", sys.getsizeof(images), " bytes)", sep="")

    return images

if __name__ == "__main__":
    images = load_oasis()
