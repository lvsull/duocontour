import numpy as np
from tqdm import tqdm

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