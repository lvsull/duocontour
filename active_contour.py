import multiprocessing as mp

import nibabel as nib
import numpy as np
from skimage.segmentation import morphological_chan_vese
import time
from pathlib import Path
from tqdm import tqdm
import os
from shutil import rmtree
import pandas as pd
from os import mkdir

bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

AFFINE = np.array([[-1, 0, 0, 128],
                   [0, 0, 1, -128],
                   [0, -1, 0, 128],
                   [0, 0, 0, 1]])


def single_channel_chan_vese(image: np.ndarray, label: np.ndarray, name: str, channel: int) -> tuple[np.ndarray, int]:
    """
    Create a Chan-Vese segmentation for a single channel
    :param image: the MRI image to contour
    :type image: numpy.ndarray
    :param label: base segmentation, used as the initial level set for Chan-Vese segmentation
    :type label: numpy.ndarray
    :param channel: which channel from ``label`` to segment
    :type channel: int
    :return: the resulting segmentation of ``channel``
    :rtype: numpy.ndarray
    """
    channel_start_time = time.time()

    channel_label = label == channel

    out = morphological_chan_vese(image, 300, init_level_set=channel_label)

    nib.save(nib.Nifti1Image(out, AFFINE), f"./output/ac_output/{name}/{channel}.nii.gz")

    # print(f"{channel} finished in {time.strftime("%H:%M:%S", time.gmtime(time.time() - channel_start_time))}")
    return out, channel


def multi_channel_chan_vese(image: np.ndarray, label: np.ndarray, name: str,
                            channels: tuple[int, int] = (1, 38)) -> np.ndarray:
    """
    Create a Chan-Vese segmentation across multiple channels
    :param image: the MRI image to contour
    :type image: numpy.ndarray
    :param label: base segmentation, used as the initial level set for Chan-Vese segmentation
    :type label: numpy.ndarray
    :param channels: *optional, default (1, 38)* tuple (lower, upper) of labels to analyze, inclusive
    :type channels: tuple[int, int]
    :return: the resulting segmentation across ``channels``
    :rtype: numpy.ndarray
    """
    start_time = time.time()

    save_path = f"./output/ac_output/{name}"

    try:
        rmtree(save_path)
    except FileNotFoundError:
        pass
    mkdir(save_path)

    out = np.zeros_like(image)

    seg_values = list(range(channels[0], channels[1] + 1))

    with mp.Pool(22) as pool:
        contours = pool.starmap(single_channel_chan_vese, [(image, label, name, channel) for channel in seg_values])

    ordered_contours = (pd.DataFrame({"image": [contour[0] for contour in contours],
                                      "channel": [contour[1] for contour in contours],
                                      "size": [np.count_nonzero(contour[0]) for contour in contours]})
                        .sort_values("size", ignore_index=True))

    for _, row in ordered_contours.iterrows():
        out[np.array(row["image"]) == 1] = row["channel"]

    print(f"Finished active contour in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

    return out


if __name__ == "__main__":
    mp.freeze_support()

    for path in tqdm(Path.iterdir(Path("./output/pred")), desc="Active Contour", total=len(os.listdir("./output/pred")),
                     bar_format=bf):
        base = path.stem.split("_predictions")[0]
        pred = nib.load(path).get_fdata()
        lbl = nib.load(Path(r"D:\Liam Sullivan LTS\preprocessed_label", f"{base}.nii.gz")).get_fdata()

        ac_out = multi_channel_chan_vese(pred, lbl, base, (1, 22))

        nib.save(nib.Nifti1Image(ac_out, AFFINE), f"./output/ac_output/{base}/label.nii.gz")
