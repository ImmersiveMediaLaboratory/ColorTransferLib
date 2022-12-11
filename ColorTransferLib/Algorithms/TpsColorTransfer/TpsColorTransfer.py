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
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces

import os
os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
from oct2py import octave, Oct2Py
from ColorTransferLib.Utils.BaseOptions import BaseOptions
import cv2

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: L2 Divergence for robust colour transfer
#   Author: Mairéad Grogan, Rozenn Dahyot
#   Published in: Computer Vision and Image Understanding
#   Year of Publication: 2019
#
# Abstract:
#   Optimal Transport (OT) is a very popular framework for performing colour transfer in images and videos. We have
#   proposed an alternative framework where the cost function used for inferring a parametric transfer function is
#   defined as the robust L2 divergence between two probability density functions Grogan and Dahyot (2015). In this
#   paper, we show that our approach combines many advantages of state of the art techniques and outperforms many
#   recent algorithms as measured quantitatively with standard quality metrics, and qualitatively using perceptual
#   studies Grogan and Dahyot (2017). Mathematically, our formulation is presented in contrast to the OT cost function
#   that shares similarities with our cost function. Our formulation, however, is more flexible as it allows colour
#   correspondences that may be available to be taken into account and performs well despite potential occurrences of
#   correspondence outlier pairs. Our algorithm is shown to be fast, robust and it easily allows for user interaction
#   providing freedom for artists to fine tune the recoloured images and videos Grogan et al. (2017).
#
# Link: https://doi.org/10.1016/j.cviu.2019.02.002
# Source: https://github.com/groganma/gmm-colour-transfer
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class TpsColorTransfer:
    identifier = "TpsColorTransfer"
    title = "L2 Divergence for robust colour transfer"
    year = 2019

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
    @staticmethod
    def get_info():
        info = {
            "identifier": "TpsColorTransfer",
            "title": "L2 Divergence for robust colour transfer",
            "year": 2019,
            "abstract": "Optimal Transport (OT) is a very popular framework for performing colour transfer in images "
                        "and videos. We have proposed an alternative framework where the cost function used for "
                        "inferring a parametric transfer function is defined as the robust L2 divergence between two "
                        "probability density functions Grogan and Dahyot (2015). In this paper, we show that our "
                        "approach combines many advantages of state of the art techniques and outperforms many recent "
                        "algorithms as measured quantitatively with standard quality metrics, and qualitatively using "
                        "perceptual studies Grogan and Dahyot (2017). Mathematically, our formulation is presented in "
                        "contrast to the OT cost function that shares similarities with our cost function. Our "
                        "formulation, however, is more flexible as it allows colour correspondences that may be "
                        "available to be taken into account and performs well despite potential occurrences of "
                        "correspondence outlier pairs. Our algorithm is shown to be fast, robust and it easily allows "
                        "for user interaction providing freedom for artists to fine tune the recoloured images and "
                        "videos Grogan et al. (2017)."
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        octave.addpath(octave.genpath('.'))
        #octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        octave.eval('pkg load image')
        octave.eval('pkg load statistics')
        octave.eval("dir")

        outp = octave.ctfunction(ref, src, opt.cluster_method, opt.cluster_num, opt.colorspace)

        return outp
