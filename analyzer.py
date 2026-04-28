import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import convolve
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import nibabel as nib
import os

bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

def mean_dsc(img1: np.ndarray, img2: np.ndarray, structs: list) -> tuple:
    img1 = img1.flatten()
    img2 = img2.flatten()
    struct_dscs = {s: 0.0 for s in structs}
    struct_weights = {s: 0.0 for s in structs}
    for struct in tqdm(structs, desc="Comparing structures", bar_format=bf):
        struct_image = img1 == struct
        struct_atlas = img2 == struct

        tp = len(np.where((struct_image == True) & (struct_atlas == True))[0])
        fp = len(np.where((struct_image == True) & (struct_atlas == False))[0])
        fn = len(np.where((struct_image == False) & (struct_atlas == True))[0])

        if not any([tp, fp, fn]):
            struct_dscs[struct] = 1.0
        else:
            struct_dscs[struct] = (2 * tp) / ((2 * tp) + fp + fn)

        struct_weights[struct] = len(img1[struct_image])

    return np.average(list(struct_dscs.values()), weights=list(struct_weights.values())), struct_dscs

def dsc(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.flatten()
    img2 = img2.flatten()
    struct_image = img1 == 1
    struct_atlas = img2 == 1
    tp = len(np.where((struct_image == True) & (struct_atlas == True))[0])
    fp = len(np.where((struct_image == True) & (struct_atlas == False))[0])
    fn = len(np.where((struct_image == False) & (struct_atlas == True))[0])

    if not any([tp, fp, fn]):
        return 1.0
    else:
        return (2 * tp) / ((2 * tp) + fp + fn)


def channel_mda(img1, img2):
    kernel = [
        [[-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1]],
        [[-1, -1, -1],
         [-1, 27, -1],
         [-1, -1, -1]],
        [[-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1]]
    ]

    boundary1 = np.argwhere(convolve(img1, kernel, mode='constant') > 1)
    boundary2 = np.argwhere(convolve(img2, kernel, mode='constant') > 1)

    if not (len(boundary1) and len(boundary2)):
        return "NA"

    nb = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nb.fit(boundary2)

    distances = nb.kneighbors(boundary1)[0]

    return np.mean(distances)


def analyze_images(img_dir, out_file):
    with open("config.yaml", "r") as f:
        label_dir = yaml.safe_load(f)["preprocessed_label"]

    output = pd.DataFrame(columns=["name", "channel", "DSC", "MDA"])

    for image_dir in tqdm(Path(img_dir).iterdir(), desc="Finding DSCs", bar_format=bf, total=len(os.listdir(r"C:\Users\Liam Sullivan\research-25-26\output\ac_output"))):
        name = image_dir.stem
        lbl = nib.load(f"{label_dir}/{name}.nii.gz").get_fdata()
        for img_path in tqdm(image_dir.iterdir(), desc=name, bar_format=bf, total=len(os.listdir(r"C:\Users\Liam Sullivan\research-25-26\output\ac_output"))-1, leave=False):
            try:
                channel = int(img_path.stem.split(".")[0]) + 1
            except ValueError:
                continue

            gt = (lbl == channel).astype(int)
            img = nib.load(img_path).get_fdata()

            output.loc[len(output)] = [name, channel, dsc(img, gt), channel_mda(img, gt)]

    output.to_csv(out_file, index=False)

if __name__ == "__main__":
    ac = pd.read_csv("acc_ac.csv").sort_values(by=["name", "channel"])
    unet = pd.read_csv("acc_unet.csv").sort_values(by=["name", "channel"])

    out = pd.DataFrame({
        "name": ac["name"],
        "channel": ac["channel"],
        "DSC_unet": unet["DSC"],
        "MDA_unet": unet["MDA"],
        "DSC_ac": ac["DSC"],
        "MDA_ac": ac["MDA"]
    })

    out.to_csv("acc.csv", index=False)