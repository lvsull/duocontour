import multiprocessing as mp

import nibabel as nib
import numpy as np
from skimage.segmentation import morphological_chan_vese
import time


def single_channel_chan_vese(image: np.ndarray, label: np.ndarray, channel: int) -> np.ndarray:
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

    out = morphological_chan_vese(image, 250, init_level_set=channel_label)

    print(f"{channel} finished in {time.strftime("%H:%M:%S", time.gmtime(time.time() - channel_start_time))}")
    return out


def multi_channel_chan_vese(image: np.ndarray, label: np.ndarray, channels: tuple[int, int] = (1, 38)) -> np.ndarray:
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

    out = np.zeros_like(image)

    seg_values = list(range(channels[0], channels[1]))

    with mp.Pool(20) as pool:
        contours = pool.starmap(single_channel_chan_vese, [(image, label, channel) for channel in seg_values])

    for i in range(len(seg_values)):
        out[contours[i] == 1] = seg_values[i]

    print(f"Finished active contour in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

    return out


if __name__ == "__main__":
    mp.freeze_support()
    image = nib.load(r"D:\Liam Sullivan LTS\preprocessed_image\BPDwoPsy_046.nii.gz").get_fdata()
    label = nib.load(r"D:\Liam Sullivan LTS\preprocessed_label\BPDwoPsy_046.nii.gz").get_fdata()

    out = multi_channel_chan_vese(image, label)
    np.save("active_contour.npy", out)
