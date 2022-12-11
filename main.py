"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import time
import numpy as np

from ColorTransferLib.MeshProcessing.PLYLoader import PLYLoader
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer


from ColorTransferLib.Utils.Math import get_random_3x3rotation_matrix
from ColorTransferLib.Evaluation.SSIM.SSIM import SSIM
from ColorTransferLib.Evaluation.PSNR.PSNR import PSNR
from ColorTransferLib.Evaluation.HistogramIntersection.HistogramIntersection import HistogramIntersection
from ColorTransferLib.Evaluation.PerceptualMetric.PerceptualMetric import PerceptualMetric
import cv2
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import os




# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Possible values for ct_approach:
# ["GlobalColorTransfer",
#  "PdfColorTransfer",
#  "NeuralStyleTransfer",
#  "HistogramAnalogy",
#  "CamsTransfer",
#  "DeepPhotoStyleTransfer",
#  "TpsColorTransfer",
#  "FuzzyColorTransfer",
#  "GmmEmColorTransfer",
#  "Eb3dColorTransfer",
#  "PSNetStyleTransfer",
#  "MongeKLColorTransfer",
#  "FrequencyColorTransfer"]
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #src_img = "/home/hpadmin/Downloads/psource.png"
    #ref_img = "/home/hpadmin/Downloads/preference.png"

    #pm = PerceptualMetric()
    #pm.apply(src, ref)
    #exit()

    #src = Image(file_path=src_img)
    #ref = Image(file_path=ref_img)
    #cstat = HistogramIntersection.apply(src, ref)
    #print(cstat)
    #exit()

    ct_approach = "GlobalColorTransfer"
    ct_input = "pc-img"

    ref_img = "data/images/starry-night.jpg"
    src_img = "data/images/woman-with-hat-matisse.jpg"

    src_pc = "data/pointclouds/athen_postprocessed_simp.ply"
    ref_pc = "data/pointclouds/Wappentier_blue.ply"


    if ct_input == "img-img":
        src = Image(file_path=src_img)
        ref = Image(file_path=ref_img)
    elif ct_input == "pc-pc":
        loader = PLYLoader(src_pc)
        src = loader.get_mesh()
        loader2 = PLYLoader(ref_pc)
        ref = loader2.get_mesh()
    elif ct_input == "img-pc":
        src = Image(file_path=src_img)
        loader2 = PLYLoader(ref_pc)
        ref = loader2.get_mesh()
    elif ct_input == "pc-img":
        loader = PLYLoader(src_pc)
        src = loader.get_mesh()
        ref = Image(file_path=ref_img)
    start_time = time.time()


    ct = ColorTransfer(src, ref, ct_approach)
    ct.set_option("colorspace", "rgb")
    output = ct.apply()

    print("TOTAL: " + str(time.time() - start_time))

    if ct_input == "img-img" or ct_input == "img-pc":
        #output.resize(1024, 1024)
        output.write("data/test.png")
        
        #output.show()
    elif ct_input == "pc-pc" or ct_input == "pc-img":
        out_loader = PLYLoader(mesh=output)
        #out_loader.write('data/pointclouds/out.ply')
        out_loader.write("data/out.ply")

