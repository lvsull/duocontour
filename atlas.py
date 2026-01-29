import nibabel as nib
import numpy as np
import yaml
from nibabel import Nifti1Image
from nilearn import datasets, image
import SimpleITK as sitk
from tqdm import tqdm
from os import path
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

from unet.unet_format_converter import fs_to_cont
from preprocessor import load_mni_template, center_pad


def load_atlas(atlas_path):
    with open("config.yaml") as config_file:
        mni_path = yaml.safe_load(config_file)["mni_template"]

    dimensions = (256, 256, 256)
    atlas = nib.load(atlas_path)
    fdata = np.round(atlas.get_fdata())
    # for slc, row, col in np.argwhere(fdata != 0):
    #     fdata[slc][row][col] = fs_to_cont(fdata[slc][row][col])
    fdata = center_pad(fdata, dimensions)
    atlas = nib.Nifti1Image(fdata, np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]))
    atlas_save_path = path.splitext(atlas_path)[0]
    atlas_save_path = path.splitext(atlas_save_path)[0]
    atlas_save_path += ".nii.gz"
    nib.save(atlas, atlas_save_path)
    atlas = nib.load(atlas_save_path)

    mni_template_fdata = load_mni_template(mni_path).get_fdata()

    def find_bounds(arr):
        bounds = [[0, arr.shape[0]], [0, arr.shape[1]], [0, arr.shape[2]]]

        transposed_arrays = [arr, arr.transpose(1, 0, 2), arr.transpose(2, 0, 1)]

        for i in range(len(transposed_arrays)):
            for start in range(len(transposed_arrays[i])):
                if np.any(transposed_arrays[i][start]) and bounds[i][0] == 0:
                    bounds[i][0] = start
            for end in range(len(transposed_arrays[i]) - 1, 0, -1):
                if np.any(transposed_arrays[i][end]) and bounds[i][1] == arr.shape[i]:
                    bounds[i][1] = end

        return bounds

    mni_bounds = find_bounds(mni_template_fdata)
    atlas_bounds = find_bounds(atlas.get_fdata())

    scale = [1, 1, 1]

    for dim in range(3):
        scale[dim] = (mni_bounds[dim][1] - mni_bounds[dim][0]) / (atlas_bounds[dim][1] - atlas_bounds[dim][0])

    scaled_atlas = zoom(atlas.get_fdata(), scale)

    new_mni_bounds = find_bounds()

    return nib.load(atlas_save_path)

def compare_to_atlas(image: np.ndarray, atlas: np.ndarray, structs: list) -> dict:
    image_card = len(np.nonzero(image))
    atlas_card = len(np.nonzero(atlas))
    struct_dscs = {s: 0.0 for s in structs}
    for struct in tqdm(structs):
        struct_image = np.where(image == struct)
        struct_atlas = np.where(atlas == struct)
        int_card = len(np.intersect1d(struct_image, struct_atlas))

        struct_dscs[struct] = (2 * int_card) / (image_card + atlas_card)

    return struct_dscs


if __name__ == "__main__":
    img = load_atlas(r"D:\Liam Sullivan LTS\labels.mgz")