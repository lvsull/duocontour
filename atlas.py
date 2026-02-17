from os import path
import os

import h5py
import nibabel as nib
import numpy as np
import yaml
from nibabel.filebasedimages import FileBasedImage
from skimage.transform import resize
from tqdm import tqdm

from preprocessor import center_pad, load_mni_template
from unet.unet_format_converter import fs_to_cont

import SimpleITK as sitk

bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

AFFINE = np.array([[-1, 0, 0, 128],
                   [0, 0, 1, -128],
                   [0, -1, 0, 128],
                   [0, 0, 0, 1]])

def find_bounds(arr, threshold = 0.05):
    bounds = [[0, arr.shape[0]], [0, arr.shape[1]], [0, arr.shape[2]]]

    transposed_arrays = [arr, arr.transpose(1, 0, 2), arr.transpose(2, 0, 1)]

    for i in range(len(transposed_arrays)):
        for start in range(len(transposed_arrays[i])):
            if np.count_nonzero(transposed_arrays[i][start]) > 255 * threshold and bounds[i][0] == 0:
                bounds[i][0] = start
        for end in range(len(transposed_arrays[i]) - 1, 0, -1):
            if np.count_nonzero(transposed_arrays[i][end]) > 255 * threshold and bounds[i][1] == arr.shape[i]:
                bounds[i][1] = end

    return bounds


def atlas_to_image(image: np.ndarray, atlas: np.ndarray):
    image_bounds = find_bounds(image)
    atlas_bounds = find_bounds(atlas)

    scale = [1, 1, 1]

    for dim in range(3):
        scale[dim] = (image_bounds[dim][1] - image_bounds[dim][0]) / (atlas_bounds[dim][1] - atlas_bounds[dim][0])

    resized_arr = resize(atlas, (256 * scale[0], 256 * scale[1], 256 * scale[2]), order=0)

    window = [[0, 0], [0, 0], [0, 0]]

    for i in range(len(resized_arr.shape)):
        window[i][0] = int(np.round((resized_arr.shape[i] - 256) / 2))
        window[i][1] = window[i][0] + 256

    resized_arr = resized_arr[window[0][0]:window[0][1], window[1][0]:window[1][1], window[2][0]:window[2][1]]

    resized_bounds = find_bounds(resized_arr)

    for axis in range(3):
        resized_arr = np.roll(resized_arr, image_bounds[axis][0] - resized_bounds[axis][0], axis=axis)

    return resized_arr

def register_to_mni(image, atlas):

    fixed = sitk.ReadImage(image, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    def command_iteration(method):
        method.GetOptimizerIteration()
        method.GetMetricValue()
        method.GetOptimizerPosition()

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    moving = sitk.ReadImage(atlas, sitk.sitkFloat32)

    outTx = R.Execute(fixed, moving)
    outTx.SetOffset([np.round(x) for x in outTx.GetOffset()])
    outTx.SetOffset([256 - x if x > 128 else x for x in outTx.GetOffset()])

    resampler.SetTransform(outTx)

    out_img = resampler.Execute(moving)

    sitk.WriteImage(out_img, atlas)

    moved_image_view = nib.load(atlas)

    return moved_image_view


def load_atlas(atlas_path: str) -> FileBasedImage:
    """
    Load and preprocess atlas from the specified path
    :param atlas_path: Location of the raw atlas
    :type atlas_path: str
    :return: View of the preprocessed atlas
    :rtype: nibabel.filebasedimages.FileBasedImage
    """

    dimensions = (256, 256, 256)
    fdata = np.round(nib.load(atlas_path).get_fdata())
    fdata = center_pad(fdata, dimensions)
    atlas_save_path = path.splitext(atlas_path)[0]
    atlas_save_path = path.splitext(atlas_save_path)[0]
    atlas_save_path += ".nii.gz"

    for slc, row, col in np.argwhere(fdata == 31):
        fdata[slc][row][col] = 2
    for slc, row, col in np.argwhere(fdata == 63):
        fdata[slc][row][col] = 41

    for slc, row, col in np.argwhere(fdata != 0):
        fdata[slc][row][col] = fs_to_cont(fdata[slc][row][col])

    fdata = np.flip(fdata, 0)

    nib.save(nib.Nifti1Image(fdata, AFFINE), atlas_save_path)

    return nib.load(atlas_save_path)


def compare_to_atlas(image: np.ndarray, atlas: np.ndarray, structs: list) -> tuple:
    image = image.flatten()
    atlas = atlas.flatten()
    struct_dscs = {s: 0.0 for s in structs}
    struct_weights = {s: 0.0 for s in structs}
    for struct in tqdm(structs, desc="Comparing structures", bar_format=bf):
        struct_image = image == struct
        struct_atlas = atlas == struct

        tp = len(np.where((struct_image == True) & (struct_atlas == True))[0])
        fp = len(np.where((struct_image == True) & (struct_atlas == False))[0])
        fn = len(np.where((struct_image == False) & (struct_atlas == True))[0])

        if not any([tp, fp, fn]):
            struct_dscs[struct] = 1.0
        else:
            struct_dscs[struct] = (2 * tp) / ((2 * tp) + fp + fn)

        struct_weights[struct] = len(image[struct_image])

        pass

    return np.average(list(struct_dscs.values()), weights=list(struct_weights.values())), struct_dscs


if __name__ == "__main__":
    atlas = load_atlas(r"D:\Liam Sullivan LTS\labels.mgz")
    # atlas = nib.load("labels.nii.gz")
    for path in os.listdir(r"C:\Users\Liam Sullivan\research-25-26\unet\pred"):
        pred = h5py.File(f"C:/Users/Liam Sullivan/research-25-26/unet/pred/{path}")["predictions"][:]
        img = nib.load(f"D:/Liam Sullivan LTS/preprocessed_image/{path.replace("_predictions.h5", ".nii.gz")}").get_fdata()
        lbl = nib.load(f"D:/Liam Sullivan LTS/preprocessed_label/{path.replace("_predictions.h5", ".nii.gz")}").get_fdata().flatten()
        for i in range(len(lbl)):
            if 20 <= lbl[i] <= 29:
                lbl[i] -= 19
            elif 30 <= lbl[i] <= 31:
                lbl[i] -= 16
            elif 32 <= lbl[i] <= 33:
                lbl[i] -= 15
        lbl = lbl.reshape(img.shape)
        resized_atlas = atlas_to_image(img, atlas.get_fdata()).flatten()
        for i in range(len(resized_atlas)):
            if 20 <= resized_atlas[i] <= 29:
                resized_atlas[i] -= 19
            elif 30 <= resized_atlas[i] <= 31:
                resized_atlas[i] -= 16
            elif 32 <= resized_atlas[i] <= 33:
                resized_atlas[i] -= 15
        resized_atlas = resized_atlas.reshape(img.shape)
        # resized_atlas = register_to_mni("images/preprocessed_label/OAS1_0025_MR1.nii.gz", "labels.nii.gz").get_fdata()
        nib.save(nib.Nifti1Image(resized_atlas, AFFINE), "image1.nii.gz")
        nib.save(nib.Nifti1Image(pred, AFFINE), "image2.nii.gz")
        pred_avg, pred_dscs = compare_to_atlas(pred, resized_atlas, list(range(1, 37)))
        gt_avg, gt_dscs = compare_to_atlas(lbl, resized_atlas, list(range(1, 37)))
        dsc_diffs = {}
        for key in pred_dscs.keys():
            try:
                dsc_diffs[key] = (pred_dscs[key]) / (gt_dscs[key])
            except ZeroDivisionError:
                dsc_diffs[key] = 1.0
        print(dsc_diffs)
        print(f"{path.replace("_predictions.h5", "")}: {np.round(pred_avg/gt_avg*100, 3)}%")
