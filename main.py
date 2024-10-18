"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.MeshProcessing.VolumetricVideo import VolumetricVideo

from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ImageProcessing.Video import Video
from ColorTransferLib.ColorTransfer import ColorTransfer, ColorTransferEvaluation

from ColorTransferLib.DataTypes.LightField import LightField
from ColorTransferLib.DataTypes.GaussianSplatting import GaussianSplatting

import cv2
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  
    #src = Video(file_path='/media/potechius/External/data/Videos/sample-5s.mp4')


    #src = VolumetricVideo(folder_path='/media/potechius/External/data/VolumetricVideos/$volumetric$Human', file_name='Human')

    #src.write("/media/potechius/External/data/VolumetricVideos/out")
    #exit()

    #src.write("/media/potechius/External/data/Videos/out.mp4")

    #frames = src.get_raw()
    # for i, frame in enumerate(frames):
    #     print(frame.dtype)
    #     cv2.imshow(f'Frame {i}', frame)
    #     # Wait for 'interval' seconds
    #     if cv2.waitKey(int(1000)) & 0xFF == ord('q'):
    #         break

    # cv2.destroyAllWindows()


    #exit()

    src_gs = GaussianSplatting(file_path='/home/potechius/Code/ColorTransferLib/testdata/gaussiansplatting/plush.splat')
    src_lf = LightField(file_path='/home/potechius/Code/ColorTransferLib/testdata/lightfields/amethyst.mp4', size=(17, 17))
    src_vv = VolumetricVideo(folder_path='/home/potechius/Code/ColorTransferLib/testdata/volumetricvideos/$volumetric$human', file_name='human')
    src_im = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')
    src_vd = Video(file_path='/home/potechius/Code/ColorTransferLib/testdata/videos/earth.mp4')

    src_array = [
            # {"type": "GaussianSplatting",
            # "data": src_gs},
            {"type": "LightField",
            "data": src_lf},
            # {"type": "VolumetricVideo",
            # "data": src_vv},
            # {"type": "Image",
            # "data": src_im},
            # {"type": "Video",
            # "data": src_vd}
    ]

    ref = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-06.png')  
    #out = Image(file_path='/media/potechius/External/data/Images/out.png')  
    
    #cte = ColorTransferEvaluation(src, ref, out)
    #eva = cte.apply("VSI")
    #print(eva)

    #exit()

    #src = Mesh(file_path='/home/potechius/Downloads/3D/src.ply', datatype="PointCloud")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D/ref.ply', datatype="PointCloud")  

    #src = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Apple.obj', datatype="Mesh")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Pillow.obj', datatype="Mesh")  


    algo = "PDF"
    for src_elem in src_array:
        typeE = src_elem["type"]
        print(typeE)
        src = src_elem["data"]
        ct = ColorTransfer(src, ref, algo)
        out = ct.apply()
        #print(out)
        #exit()

        if out["status_code"] == 0:
            out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/out_" + typeE)
            src.write("/home/potechius/Code/ColorTransferLib/testdata/results/src_" + typeE)
            ref.write("/home/potechius/Code/ColorTransferLib/testdata/results/ref")
        else:
            print("Error: " + out["response"])
        print("Done")

    # Evaluation Example
    # TODO


