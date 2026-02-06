from os import path

import nibabel as nib
import numpy as np
import yaml
from nibabel.filebasedimages import FileBasedImage
from skimage.transform import resize
from tqdm import tqdm

from preprocessor import center_pad, load_mni_template
from unet.unet_format_converter import fs_to_cont


def load_atlas(atlas_path: str) -> FileBasedImage:
    """
    Load and preprocess atlas from the specified path
    :param atlas_path: Location of the raw atlas
    :type atlas_path: str
    :return: View of the preprocessed atlas
    :rtype: nibabel.filebasedimages.FileBasedImage
    """
    with open("config.yaml") as config_file:
        mni_path = yaml.safe_load(config_file)["mni_template"]

    dimensions = (256, 256, 256)
    atlas = nib.load(atlas_path)
    fdata = np.round(atlas.get_fdata())
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
    atlas_fdata = atlas.get_fdata()

    for slc, row, col in np.argwhere(atlas_fdata == 31):
        atlas_fdata[slc][row][col] = 2
    for slc, row, col in np.argwhere(atlas_fdata == 63):
        atlas_fdata[slc][row][col] = 41

    mni_template_fdata = np.flip(load_mni_template(mni_path).get_fdata().transpose(0, 2, 1), 1)

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
    atlas_bounds = find_bounds(atlas_fdata)

    scale = [1, 1, 1]

    for dim in range(3):
        scale[dim] = (mni_bounds[dim][1] - mni_bounds[dim][0]) / (atlas_bounds[dim][1] - atlas_bounds[dim][0])

    resized_arr = resize(atlas_fdata, (256 * scale[0], 256 * scale[1], 256 * scale[2]), order=0)

    window = [[0, 0], [0, 0], [0, 0]]

    for i in range(len(resized_arr.shape)):
        window[i][0] = int(np.round((resized_arr.shape[i] - 256) / 2))
        window[i][1] = window[i][0] + 256

    resized_arr = resized_arr[window[0][0]:window[0][1], window[1][0]:window[1][1], window[2][0]:window[2][1]]

    for slc, row, col in np.argwhere(resized_arr != 0):
        resized_arr[slc][row][col] = fs_to_cont(resized_arr[slc][row][col])

    nib.save(nib.Nifti1Image(resized_arr, np.eye(4)), atlas_save_path)

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
    atlas = load_atlas(r"labels.mgz")