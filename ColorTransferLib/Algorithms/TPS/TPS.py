"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
import os
from copy import deepcopy
from sys import platform

from ColorTransferLib.Utils.Helper import check_compatibility

if platform == "linux" or platform == "linux2":
    # linux
    os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
elif platform == "darwin":
    # OS X
    os.environ["OCTAVE_EXECUTABLE"] = "/opt/homebrew/bin/octave-cli"
elif platform == "win32":
    # Windows...
    pass

from oct2py import octave
from ColorTransferLib.ImageProcessing.Video import Video
from ColorTransferLib.MeshProcessing.VolumetricVideo import VolumetricVideo


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: L2 Divergence for robust colour transfer
#   Author: Mairéad Grogan, Rozenn Dahyot
#   Published in: Computer Vision and Image Understanding
#   Year of Publication: 2019

# Info:
#   Name: TpsColorTransfer
#   Identifier: TPS
#   Link: https://doi.org/10.1016/j.cviu.2019.02.002
#   Source: https://github.com/groganma/gmm-colour-transfer
#
# Implementation Details:
#   Usage of Octave to run the Matlab-Scripts
#   Clustering is done using KMeans because MVQ does not work in Octave
#   Internal image resizing (mg applyK-Means.m) to 300x350px for clustering
#   Remove largescale and TolCon option in gmmregrbfl2.m because unrecognized
# 
# Note:
#   The Octave Forge package repository is no longer actively maintained. 
#   Please find Octave Packages at https://packages.octave.org. 
#   pkg install "https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/image-2.14.0.tar.gz"
#   pkg install "https://github.com/gnu-octave/statistics/archive/refs/tags/release-1.7.0.tar.gz"
#   ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class TPS:
    # ------------------------------------------------------------------------------------------------------------------
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):

        print()

        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = TPS.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = TPS.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = TPS.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = TPS.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = TPS.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = TPS.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = TPS.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_img, ref_img, opt):
        # NOTE: sudo apt-get install liboctave-dev
        # NOTE: pkg install -forge image
        # NOTE: pkg install -forge statistics

        # check if method is compatible with provided source and reference objects
        #output = check_compatibility(src, ref, TPS.compatibility)

        #if output["status_code"] == -1:
        #    output["response"] = "Incompatible type."
        #    return output

        # Preprocessing
        # NOTE RGB space needs multiplication with 255
        src_img = src_img * 255
        ref_img = ref_img * 255

        # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        #octave.addpath(octave.genpath('.'))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("-----------------")
        print(os.path.join(current_dir, 'L2RegistrationForCT'))
        print("-----------------")
        octave.addpath(octave.genpath(os.path.join(current_dir, 'L2RegistrationForCT')))
        # octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        #octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        octave.eval('pkg load image')
        octave.eval('pkg load statistics')

        outp = octave.ctfunction(ref_img, src_img, opt.cluster_method, opt.cluster_num, opt.colorspace)
        outp = outp.astype(np.float32)

        return outp
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = TPS.__color_transfer(src_img, ref_img, opt)
        out_img.set_raw(out_colors)

        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_images()[0])

            out_colors = TPS.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)

        outp = Video(imgs=out_raw_arr)

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = TPS.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_raw_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_raw = src_lightfield_array[row][col].get_raw()
                ref_raw = ref.get_raw()

                out_colors = TPS.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_raw(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = TPS.__color_transfer(src_img, ref_img, opt)

        out_img.set_raw(out_colors)
        outp = out_img
        return outp

