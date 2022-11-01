"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
from module.ImageProcessing.ColorSpaces import ColorSpaces
from module.Utils.BaseOptions import BaseOptions
from scipy.linalg import fractional_matrix_power


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: The Linear Monge-Kantorovitch Linear Colour Mapping forExample-Based Colour Transfer
#   Author: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
#   Published in: 4th European Conference on Visual Media Production
#   Year of Publication: 2007
#
# Abstract:
#   A common task in image editing is to change the colours of a picture to match the desired colour grade of another 
#   picture. Finding the correct colour mapping is tricky because it involves numerous interrelated operations, like 
#   balancing the colours, mixing the colour channels or adjusting the contrast. Recently, a number of automated tools 
#   have been proposed to find an adequate one-to-one colour mapping. The focus in this paper is on finding the best 
#   linear colour transformation. Linear transformations have been proposed in the literature but independently. The aim 
#   of this paper is thus to establish a common mathematical background to all these methods. Also, this paper proposes 
#   a novel transformation, which is derived from the Monge-Kantorovitch theory of mass transportation. The proposed 
#   solution is optimal in the sense that it minimises the amount of changes in the picture colours. It favourably 
#   compares theoretically and experimentally with other techniques for various images and under various colour spaces.
#
# Link: https://doi.org/10.1049/cp:20070055
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MongeKLColorTransfer:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "MongeKLColorTransfer",
            "title": "The Linear Monge-Kantorovitch Linear Colour Mapping forExample-Based Colour Transfer",
            "year": 2007,
            "abstract": "A common task in image editing is to change the colours of a picture to match the desired " 
                        "colour grade of another picture. Finding the correct colour mapping is tricky because it "
                        "involves numerous interrelated operations, like balancing the colours, mixing the colour "
                        "channels or adjusting the contrast. Recently, a number of automated tools have been proposed "
                        "to find an adequate one-to-one colour mapping. The focus in this paper is on finding the best "
                        "linear colour transformation. Linear transformations have been proposed in the literature but "
                        "independently. The aim of this paper is thus to establish a common mathematical background to "
                        "all these methods. Also, this paper proposes a novel transformation, which is derived from "
                        "the Monge-Kantorovitch theory of mass transportation. The proposed solution is optimal in the "
                        "sense that it minimises the amount of changes in the picture colours. It favourably compares "
                        "theoretically and experimentally with other techniques for various images and under various "
                        "colour spaces."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, options=[]):
        opt = BaseOptions(options)


        # Convert colors from RGB to lalphabeta color space
        src_color_rgb = src.reshape(src.shape[0], 3)
        ref_color_rgb = ref.reshape(ref.shape[0], 3)


        #src_mean = np.mean(src_color_rgb, axis=0)
        #ref_mean = np.mean(ref_color_rgb, axis=0)
        src_cov = np.cov(src_color_rgb, rowvar=False)
        ref_cov = np.cov(ref_color_rgb, rowvar=False)


        src_covs = fractional_matrix_power(src_cov, 0.5)
        src_covsr = fractional_matrix_power(src_cov, -0.5)

        M = np.dot(src_covsr, np.dot(fractional_matrix_power(np.dot(src_covs, np.dot(ref_cov, src_covs)), 0.5), src_covsr))

        out = np.dot(src_color_rgb, M)

        #lab_new = f_out[:,:3]
        #lab_new = lab_new.reshape(src.get_num_vertices(), 1, 3)
        #lab_new = ColorSpaces.lab_to_rgb_host(np.ascontiguousarray(lab_new, dtype=np.float32))
        #lab_new = np.clip(lab_new, 0.0, 1.0)

        return out
