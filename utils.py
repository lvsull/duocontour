import numpy as np

bf = "{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}"

AFFINE = np.array([[-1, 0, 0, 128],
                   [0, 0, 1, -128],
                   [0, -1, 0, 128],
                   [0, 0, 0, 1]])