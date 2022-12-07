"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from module.Algorithms.GlobalColorTransfer.GlobalColorTransfer import GlobalColorTransfer
from module.Algorithms.PdfColorTransfer.PdfColorTransfer import PdfColorTransfer
from module.Algorithms.NeuralStyleTransfer.NeuralStyleTransfer import NeuralStyleTransfer
from module.Algorithms.HistogramAnalogy.HistogramAnalogy import HistogramAnalogy
from module.Algorithms.CamsTransfer.CamsTransfer import CamsTransfer
from module.Algorithms.DeepPhotoStyleTransfer.DeepPhotoStyleTransfer import DeepPhotoStyleTransfer
from module.Algorithms.TpsColorTransfer.TpsColorTransfer import TpsColorTransfer
from module.Algorithms.FuzzyColorTransfer.FuzzyColorTransfer import FuzzyColorTransfer
from module.Algorithms.GmmEmColorTransfer.GmmEmColorTransfer import GmmEmColorTransfer
from module.Algorithms.Eb3dColorTransfer.Eb3dColorTransfer import Eb3dColorTransfer
from module.Algorithms.PSNetStyleTransfer.PSNetStyleTransfer import PSNetStyleTransfer
from module.Algorithms.MongeKLColorTransfer.MongeKLColorTransfer import MongeKLColorTransfer
from module.Algorithms.FrequencyColorTransfer.FrequencyColorTransfer import FrequencyColorTransfer
from copy import deepcopy
import sys
import json
import os

available_methods = [
    "GlobalColorTransfer",
    "PdfColorTransfer",
    "NeuralStyleTransfer",
    "HistogramAnalogy",
    "DeepPhotoStyleTransfer",
    "CamsTransfer",
    "TpsColorTransfer",
    "FuzzyColorTransfer",
    "GmmEmColorTransfer",
    "Eb3dColorTransfer",
    "PSNetStyleTransfer",
    "MongeKLColorTransfer",
    "FrequencyColorTransfer"
]

available_metrics = [
    "SSIM",
    "PSNR",
    "HistogramIntersection"
]


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Proxy class for all color transfer algorithms within this project. This class allows the call of the algorithms with
# different kind of input data without preprocessing. The following input data can be processed:
#
# ------------------------------
# | Source      | Reference    |
# ------------------------------
# | Image       | Image        |
# | Image       | Pointcloud   |
# | Pointcloud  | Image        |
# | Pointcloud  | Pointcloud   |
# ------------------------------
#
# The following approaches are currently supported:
# global - 2001 - Color Transfer between Images
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ColorTransfer:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, src, ref, options=[]):
        self.__src = src
        self.__src_type = src.get_type()
        self.__ref = ref
        self.__ref_type = ref.get_type()

        self.__out = deepcopy(src)

        self.__options = options

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def apply(self, app):
        options = self.__options
        if app == "GlobalColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = GlobalColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_colors(out_colors)
        elif app == "MongeKLColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = MongeKLColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_colors(out_colors)
        elif app == "PdfColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = PdfColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_colors(out_colors)
        elif app == "NeuralStyleTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = NeuralStyleTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "FrequencyColorTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = FrequencyColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "HistogramAnalogy":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = HistogramAnalogy.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "CamsTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = CamsTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "DeepPhotoStyleTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = DeepPhotoStyleTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "TpsColorTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = TpsColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "FuzzyColorTransfer":
            scale = 300
            self.__src.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            self.__ref.resize(width=scale * self.__ref.get_width() // self.__ref.get_height(), height=scale)
            self.__out.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = FuzzyColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_colors(out_colors)
        elif app == "GmmEmColorTransfer":
            scale = 200
            self.__src.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            self.__ref.resize(width=scale * self.__ref.get_width() // self.__ref.get_height(), height=scale)
            self.__out.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = GmmEmColorTransfer.apply(src_color, ref_color, options)
            self.__out.set_raw(out_colors)
        elif app == "Eb3dColorTransfer":
            out_colors = Eb3dColorTransfer.apply(self.__src, self.__ref, options)
            self.__out.set_colors(out_colors)
        elif app == "PSNetStyleTransfer":
            #out_colors = PSNetStyleTransfer.apply(self.__src, self.__ref, options)
            #self.__out.set_colors(out_colors)
            self.__out = PSNetStyleTransfer.apply(self.__src, self.__ref, options)

        return self.__out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_available_methods():
        av_methods = []
        for met in available_methods:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, "Options/" + met + ".json")

            with open(filename, 'r') as f:
                options = json.load(f)

            av_m = {
                "name": met,
                "options": options,
                "abstract": getattr(sys.modules[__name__], met).get_info()["abstract"],
                "title": getattr(sys.modules[__name__], met).get_info()["title"],
                "year": getattr(sys.modules[__name__], met).get_info()["year"]
            }
            av_methods.append(av_m)
        return av_methods

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_available_metrics():
        return available_metrics
