"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import os
import numpy as np
from numba import cuda 
import json
import tensorflow as tf
import sys
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')
from ColorTransferLib.ImageProcessing.Image import Image
from .predict import predict
from .utils.utils import calc_mean_score
from .handlers.model_builder import Nima

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ... (NIMA)
# 
#
# Source: https://github.com/idealo/image-quality-assessment
#
# Range [1, 10] Best: 10
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class NIMA:
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
    def apply(img):
        img.resize(224, 224)
        img = img.get_raw()

        img_format = "png"
        base_model_name = "MobileNet"
        weights_file = "Models/NIMA/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"

        # build model and load weights
        nima = Nima(base_model_name, weights=None)
        nima.build()
        nima.nima_model.load_weights(weights_file)

        # get predictions
        predictions = predict(nima.nima_model, img)
        nim = calc_mean_score(predictions[0])

        tf.keras.backend.clear_session()

        return round(nim, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "BCC"
    total_tests = 0
    eval_arr = []

    # check if entries already exist
    # exf = []
    # with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/nima.txt","r") as exfile:
    #     for line in exfile.readlines():
    #         _, exf_src, exf_ref = line.strip().split(" ")
    #         exf.append((exf_src + ".png", exf_ref + ".png"))

    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")

        # if (s_p, r_p) in exf:
        #     continue

        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        out = Image(array=out_img)
        mse = NIMA.apply(out)
        print(mse)
        eval_arr.append(mse)

        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/nima.txt","a") as file2:
            file2.writelines(str(round(mse,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



        # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))

    file1.close()



if __name__ == "__main__":
    main()
