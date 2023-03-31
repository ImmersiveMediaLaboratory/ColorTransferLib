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
import numpy as np
from ColorTransferLib.ImageProcessing.Image import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Mean Square Error (MSE)
# 
#
# Source: 
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MSE:
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
    def apply(src, ref):
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        num_pix = src_img.shape[0] * src_img.shape[1]# + ref_img.shape[0] * ref_img.shape[1]

        mse_c = np.sum(np.power(np.subtract(src_img, ref_img), 2), axis=(0,1)) / num_pix
        mse = np.sum(mse_c) / 3

        # print(mse)
        # psnr_v = 10 * math.log10(math.pow(1,2) / mse)
        # print(psnr_v)

        # difference = cv2.subtract(src.get_raw(), ref.get_raw())
        # b, g, r = cv2.split(difference)
        # if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        #     return 9999

        # psnrval = psnr(src.get_raw(), ref.get_raw())
        # print(psnrval)
        # exit()
        return round(mse, 4)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def apply2(src, ref):
    #     mssim = ssim(src.get_raw(), ref.get_raw(), channel_axis=2)
    #     return round(mssim, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        out = Image(array=out_img)
        mse = MSE.apply(src, out)
        print(mse)
        eval_arr.append(mse)

        with open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/mse.txt","a") as file2:
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
