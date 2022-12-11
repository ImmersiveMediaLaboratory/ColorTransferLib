"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# HistogramIntersection
# ...
#
# Source: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HistogramIntersection:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, bins=[10,10,10]):
        histo1 = src.get_color_statistic_3D(bins=bins)
        histo2 = ref.get_color_statistic_3D(bins=bins)
        minima = np.minimum(histo1, histo2)
        intersection = np.true_divide(np.sum(minima), np.sum(histo2))
        return round(intersection, 4)