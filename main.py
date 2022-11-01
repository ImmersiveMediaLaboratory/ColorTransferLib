"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import time

from module.MeshProcessing.PLYLoader import PLYLoader
from module.ImageProcessing.Image import Image
from module.ColorTransfer import ColorTransfer
import json

from module.Utils.Math import get_random_3x3rotation_matrix
import numpy as np
from module.Algorithms.NeuralStyleTransfer.NeuralStyleTransfer import NeuralStyleTransfer
import cv2
import tensorflow_probability as tfp
import tensorflow as tf
import torch
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
    ct_approach = "GlobalColorTransfer"
    ct_input = "img-img"
    #src_img = "data/images/2020_Lee_Example-18_Source.png"
    #ref_img = "data/images/2020_Lee_Example-18_Reference.png"
    #src_img = "data/images/starry-night.jpg"
    #ref_img = "data/images/woman-with-hat-matisse.jpg"
    src_img = "data/images/WhiteRose.jpg"
    ref_img = "data/images/rose.jpg"
    #src_img = "data/images/the_scream.jpg"
    #ref_img = "data/images/northern_lights.jpg"
    src_pc = "data/pointclouds/table.ply"
    ref_pc = "data/pointclouds/lamp.ply"

    #src = Image(file_path=src_img)
    # print(src.get_color_statistic()[2])
    # exit()


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

    with open("module/Options/GlobalColorTransfer.json", 'r') as f:
        options = json.load(f)
    #options[0]["max_iterations"] = 5000

    ct = ColorTransfer(src, ref, options)
    output = ct.apply(ct_approach)

    print("TOTAL: " + str(time.time() - start_time))

    if ct_input == "img-img" or ct_input == "img-pc":
        output.write("/home/hpadmin/Downloads/test.png")
        #output.show()
    elif ct_input == "pc-pc" or ct_input == "pc-img":
        out_loader = PLYLoader(mesh=output)
        #out_loader.write('data/pointclouds/out.ply')
        out_loader.write("data/pointclouds/out.ply")

