"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Defines a 3-dimensional vertex with its properties, i.e., color and normal.
# normal vectors are normalized
# color values range from 0.0 to 1.0
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Vertex:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, position=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), color=(1.0, 1.0, 1.0, 1.0), label=0, idx=-1):
        self.__idx = idx
        self.__position = position
        self.__normal = normal
        self.__color = color
        self.__label = label

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_position(self):
        return self.__position

    def get_normal(self):
        return self.__normal

    def get_label(self):
        return self.__label

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def get_color(self):
        return self.__color

    def get_idx(self):
        return self.__idx

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_color(self, color):
        self.__color = color

    def set_position(self, position):
        self.__position = position
