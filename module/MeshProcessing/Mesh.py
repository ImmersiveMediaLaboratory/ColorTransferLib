"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Defines a triangle mesh or point cloud, i.e., mesh without faces.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Mesh:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.__type = "Mesh"

        # vertex properties
        self.__vertices_enabled = False
        #self.__vnormals_enabled = False
        #self.__vcolors_enabled = False
        self.__vertices = []
        self.__num_vertices = 0

        # face properties
        self.__faces_enabled = False
        #self.__fnormals_enabled = False
        #self.__fcolors_enabled = False
        self.__faces = []
        self.__num_faces = 0
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vertices(self):
        return self.__vertices

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the colors of all vertices as numpy array with shape (len(vertices), 1, 3). Alpha channel will be
    # ignored.
    # ------------------------------------------------------------------------------------------------------------------
    def get_colors(self):
        list_color = [vertex.get_color()[:-1] for vertex in self.__vertices]
        numpy_color = np.asarray(list_color, dtype=np.float32).reshape(len(list_color), 1, 3)
        return numpy_color

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the normals of all vertices as numpy array with shape (len(vertices), 1, 3).
    # ------------------------------------------------------------------------------------------------------------------
    def get_normals(self):
        list_normal = [vertex.get_normal() for vertex in self.__vertices]
        numpy_normal = np.asarray(list_normal, dtype=np.float32).reshape(len(list_normal), 1, 3)
        return numpy_normal
    
        # ------------------------------------------------------------------------------------------------------------------
    # Returns the positions of all vertices as numpy array with shape (len(vertices), 1, 3).
    # ------------------------------------------------------------------------------------------------------------------
    def get_vertex_positions(self):
        list_pos = [vertex.get_position() for vertex in self.__vertices]
        numpy_pos = np.asarray(list_pos, dtype=np.float32).reshape(len(list_pos), 1, 3)
        return numpy_pos


    # ------------------------------------------------------------------------------------------------------------------
    # returns the color histogram, mean and variance
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_statistic(self):
        color = self.get_colors()
        rgb_c = (color * 255.0).astype(np.int).reshape(color.shape[0], color.shape[2])
        histo_red = np.asarray(np.histogram(rgb_c[:,0], bins=np.arange(257))[0]).reshape(256,1)
        histo_green = np.asarray(np.histogram(rgb_c[:,1], bins=np.arange(257))[0]).reshape(256,1)
        histo_blue = np.asarray(np.histogram(rgb_c[:,2], bins=np.arange(257))[0]).reshape(256,1)
        histo = np.concatenate((histo_red, histo_green, histo_blue), axis=1)
        mean = np.mean(rgb_c, axis=0).astype(np.int)
        std = np.std(rgb_c, axis=0).astype(np.int)
        return histo, mean, std

    def get_faces(self):
        return self.__faces

    def get_num_vertices(self):
        return len(self.__vertices)

    def get_num_faces(self):
        return len(self.__faces)

    def get_type(self):
        return self.__type

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_vertices(self, vertices):
        self.__vertices = vertices
        self.__vertices_enabled = True
        self.__num_vertices = len(vertices)

    # ------------------------------------------------------------------------------------------------------------------
    # <positions> have to be provided in shape (len(vertices), 1, 3)
    # ------------------------------------------------------------------------------------------------------------------
    def set_vertex_positions(self, positions):
        # check if the list <colors> has the same amount of elements as <self.__vertices>
        if len(positions) != len(self.__vertices):
            raise ValueError("len(positions)=" + len(positions) + " is unequal len(self.__vertices)=" + len(self.__vertices))
        for pos, vertex in zip(positions, self.__vertices):
            vertex.set_position(pos.reshape(3))
    # -------------------------------------------------------------
    # -----------------------------------------------------
    # <colors> have to be provided in shape (len(vertices), 1, 3)
    # ------------------------------------------------------------------------------------------------------------------
    def set_colors(self, colors):
        # check if the list <colors> has the same amount of elements as <self.__vertices>
        if len(colors) != len(self.__vertices):
            raise ValueError("len(colors)=" + len(colors) + " is unequal len(self.__vertices)=" + len(self.__vertices))
        for color, vertex in zip(colors, self.__vertices):
            alpha = np.array([[1.0]])
            color = np.append(color, alpha, 1).reshape(4)
            vertex.set_color(color)

    def set_faces(self, faces):
        self.__faces = faces
        self.__faces_enabled = True
        self.__num_faces = len(faces)


