import nibabel as nib
import numpy as np
import json
from nilearn import datasets, image
import SimpleITK as sitk
from tqdm import tqdm
from os import path
import pandas as pd

from unet.unet_format_converter import fs_to_cont
from preprocessor import load_mni_template, center_pad


def load_atlas(atlas_path):
    with open("config.yaml") as config_file:
        mni_path = json.load(config_file).get("mni_template")

    dimensions = (256, 256, 256)
    atlas = nib.load(atlas_path)
    fdata = np.round(atlas.get_fdata())
    for slc, row, col in np.argwhere(fdata != 0):
        fdata[slc][row][col] = fs_to_cont(fdata[slc][row][col])
    center_pad(fdata, dimensions)
    atlas = nib.Nifti1Image(fdata, np.eye(4))
    atlas_save_path = path.splitext(atlas_path)[0]
    atlas_save_path = path.splitext(atlas_save_path)[0]
    atlas_save_path += ".nii.gz"
    nib.save(atlas, atlas_save_path)

    load_mni_template(mni_path)

    fixed = sitk.ReadImage(mni_path, sitk.sitkFloat32)

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

    moving = sitk.ReadImage(atlas_save_path, sitk.sitkFloat32)

    outTx = R.Execute(fixed, moving)
    outTx.SetOffset([np.round(x) for x in outTx.GetOffset()])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)

    moved_img_arr = sitk.GetArrayFromImage(simg2)

    sitk.WriteImage(sitk.GetImageFromArray(moved_img_arr), atlas_save_path)

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
    atlas_data = nib.load(r"D:\Liam Sullivan LTS\labels.mgz").get_fdata()
    # atlas_data = load_atlas(r"D:\Liam Sullivan LTS\labels.mgz").get_fdata()
    image_data = nib.load(r"D:\Liam Sullivan LTS\label\OAS1_0075_MR1.nii.gz").get_fdata()
    structs = list(range(len(pd.read_csv("seg_values.csv").index)))
    dscs = compare_to_atlas(image_data, atlas_data, structs)
    print(dscs)