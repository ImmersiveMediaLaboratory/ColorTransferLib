"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import importlib
from copy import deepcopy
import sys

import json
import os

from ColorTransferLib.Utils.BaseOptions import BaseOptions

from ColorTransferLib.Utils.Helper import get_methods, get_metrics

available_methods = get_methods()
available_metrics = get_metrics()
for m in available_methods:
    exec(m + " = getattr(importlib.import_module('ColorTransferLib.Algorithms."+m+"."+m+"'), '"+m+"')")

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
    def __init__(self, src, ref, approach):
        self.__src = src
        #self.__src_type = src.get_type()
        self.__ref = ref
        #self.__ref_type = ref.get_type()

        self.__out = None#deepcopy(src)
        self.__approach = approach

        with open(os.path.dirname(os.path.abspath(__file__)) + "/Options/" + approach + ".json", 'r') as f:
            options = json.load(f)
            self.__options = BaseOptions(options)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def apply(self):
        self.__out = globals()[self.__approach].apply(self.__src, self.__ref, self.__options)
        return self.__out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def set_option(self, key, value):
        if not key in self.__options.get_keys():
            raise RuntimeError("\033[91m" + "Error: Key <" + key + "> does not exist!" + "\033[0m")
        else:
            self.__options.set_option(key, value)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def set_options(self, options):
        self.__options.set_options(options)

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
