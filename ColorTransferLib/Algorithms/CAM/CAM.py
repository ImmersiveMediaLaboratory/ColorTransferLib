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
import time

import ColorTransferLib.Algorithms.CAM.color_aware_st as cwst
import cv2
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility

import torch
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: CAMS: Color-Aware Multi-Style Transfer
#   Author: Mahmoud Afifi, Abdullah Abuolaim, Mostafa Hussien, Marcus A. Brubaker, Michael S. Brown
#   Published in: I...
#   Year of Publication: 2021
#
# Abstract:
#   Image style transfer aims to manipulate the appearance of a source image, or "content" image, to share similar
#   texture and colors of a target "style" image. Ideally, the style transfer manipulation should also preserve the
#   semantic content of the source image. A commonly used approach to assist in transferring styles is based on Gram
#   matrix optimization. One problem of Gram matrix-based optimization is that it does not consider the correlation
#   between colors and their styles. Specifically, certain textures or structures should be associated with specific
#   colors. This is particularly challenging when the target style image exhibits multiple style types. In this work,
#   we propose a color-aware multi-style transfer method that generates aesthetically pleasing results while preserving
#   the style-color correlation between style and generated images. We achieve this desired outcome by introducing a
#   simple but efficient modification to classic Gram matrix-based style transfer optimization. A nice feature of our
#   method is that it enables the users to manually select the color associations between the target style and content
#   image for more transfer flexibility. We validated our method with several qualitative comparisons, including a user
#   study conducted with 30 participants. In comparison with prior work, our method is simple, easy to implement, and
#   achieves visually appealing results when targeting images that have multiple styles. Source code is available at
#   this https URL.
#
# Info:
#   Name: CamsTransfer
#   Identifier: CAM
#   Link: https://doi.org/10.48550/arXiv.2106.13920
#   Source: https://github.com/mahmoudnafifi/color-aware-style-transfer
#
# Implementation Details:
#   in ComputeHistogram add small value to prevent division by zero when using images with small color depth
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CAM:
    compatibility = {
        "src": ["Image"],
        "ref": ["Image"]
    }
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
    def get_info():
        info = {
            "identifier": "CamsTransfer",
            "title": "CAMS: Color-Aware Multi-Style Transfer",
            "year": 2021,
            "abstract": "Image style transfer aims to manipulate the appearance of a source image, or content image, "
                        "to share similar texture and colors of a target style image. Ideally, the style transfer "
                        "manipulation should also preserve the semantic content of the source image. A commonly used "
                        "approach to assist in transferring styles is based on Gram matrix optimization. One problem "
                        "of Gram matrix-based optimization is that it does not consider the correlation between colors "
                        "and their styles. Specifically, certain textures or structures should be associated with "
                        "specific colors. This is particularly challenging when the target style image exhibits "
                        "multiple style types. In this work, we propose a color-aware multi-style transfer method that "
                        "generates aesthetically pleasing results while preserving the style-color correlation between "
                        "style and generated images. We achieve this desired outcome by introducing a simple but "
                        "efficient modification to classic Gram matrix-based style transfer optimization. A nice "
                        "feature of our method is that it enables the users to manually select the color associations "
                        "between the target style and content image for more transfer flexibility. We validated our "
                        "method with several qualitative comparisons, including a user study conducted with 30 "
                        "participants. In comparison with prior work, our method is simple, easy to implement, and "
                        "achieves visually appealing results when targeting images that have multiple styles. Source "
                        "code is available at this https URL.",
            "types": ["Image"]
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, options):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, CAM.compatibility)

        if output["status_code"] == -1:
            return output
        
        if not torch.cuda.is_available():
            options.device = "cpu"


        # resize ref to fit src
        ref.resize(src.get_width(), src.get_height())

        # Preprocessing
        src_img = src.get_raw() * 255.0
        ref_img = ref.get_raw() * 255.0
        out_img = deepcopy(src)

        out = cwst.apply(src_img, ref_img, options)

        out_img.set_raw(out)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output

