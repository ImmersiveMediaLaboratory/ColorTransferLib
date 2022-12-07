"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
from module.MeshProcessing.Mesh import Mesh
from module.MeshProcessing.Vertex import Vertex
from module.MeshProcessing.Face import Face
import numpy as np
import math
import struct


# Define PLY types
ply_dtypes = dict([
    ('int8', 'i1'),
    ('char', 'i1'),
    ('uint8', 'u1'),
    ('uchar', 'u1'),
    ('int16', 'i2'),
    ('short', 'i2'),
    ('uint16', 'u2'),
    ('ushort', 'u2'),
    ('int32', 'i4'),
    ('int', 'i4'),
    ('uint32', 'u4'),
    ('uint', 'u4'),
    ('float32', 'f4'),
    ('float', 'f4'),
    ('float64', 'f8'),
    ('double', 'f8')
])

class_colors = [
    np.array([0, 0, 1]),        # Blue
    np.array([0, 1, 0]),        # Green
    np.array([0, 1, 1]),        # 
    np.array([1, 0, 0]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1]),
    np.array([0, 0, 0.5]),
    np.array([0, 0.5, 0]),
    np.array([0, 0.5, 0.5]),
    np.array([0.5, 0, 0]),
    np.array([0.5, 0, 0.5]),
    np.array([0.5, 0.5, 0]),
]

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

# Vertex properties
vertex_props = {
    "x" : {"enabled":False, "type":None},
    "y" : {"enabled":False, "type":None},
    "z" : {"enabled":False, "type":None},
    "nx" : {"enabled":False, "type":None},
    "ny" : {"enabled":False, "type":None},
    "nz" : {"enabled":False, "type":None},
    "red" : {"enabled":False, "type":None},
    "green" : {"enabled":False, "type":None},
    "blue" : {"enabled":False, "type":None},
    "alpha" : {"enabled":False, "type":None},
    "label" : {"enabled":False, "type":None}
}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Loads ascii or binary formatted meshes with the following attributes:
# - 3-dim. float vertices
# - 3-dim. float normal per vertex
# - 3 or 4-dim. int color per vertex
# - 1-dim. label scalar
# - triangles faces without normals
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PLYLoader:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None, mesh=None):
        self.vertex_properties = []
        self.face_properties = []

        if file_path is not None:
            self.__mesh = Mesh()
            self.__format = ""
            self.__format_version = 0.0
            self.__read(file_path)
        else:
            self.__mesh = mesh
            self.__format = "ascii"
            self.__format_version = 1.0

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __read(self, file_path):
        #mode = "rb" if self.__isBinary else "r"
        self.__file_ply = open(file_path, "rb")
        self.__read_header_binary()

        if self.__format == "ascii":
            vertices = self.__read_vertices()
            self.__read_faces(vertices)
        else:
            vertices = self.__read_vertices_binary()
            self.__read_faces_binary(vertices)
        self.__file_ply.close()

        return

        vertices = self.__read_vertices()
        self.__read_faces(vertices)
        self.__file_ply.close()
    # ------------------------------------------------------------------------------------------------------------------
    # Read and process PLY header.
    # ------------------------------------------------------------------------------------------------------------------
    def __read_header_binary(self):
        line = self.__file_ply.readline().decode().strip().split(" ")
        while line[0] != "end_header":
            # check if line contains the identifier "ply"
            if line[0] == "ply":
                isValid = True
            # check second line for format (binary of ascii)
            if line[0] == "format":
                self.__format = line[1]
                self.__format_version = line[2]
            # check for vertex numbers
            if line[0] == "element":
                if line[1] == "vertex": self.__num_vertices = int(line[2])
                if line[1] == "face": self.__num_faces = int(line[2])
            if line[0] == "property":
                # Face properties
                if line[1] == "list":
                    self.face_properties.append(("f_num", valid_formats[self.__format] + ply_dtypes[line[2]]))
                    self.face_properties.append(("fx", valid_formats[self.__format] + ply_dtypes[line[3]]))
                    self.face_properties.append(("fy", valid_formats[self.__format] + ply_dtypes[line[3]]))
                    self.face_properties.append(("fz", valid_formats[self.__format] + ply_dtypes[line[3]]))
                # Vertex properties
                else:
                    for key in vertex_props:
                        if line[2] == key: 
                            vertex_props[key]["enabled"] = True
                            vertex_props[key]["type"] = line[1]
                    # example (x, f4)
                    self.vertex_properties.append((line[2], valid_formats[self.__format] + ply_dtypes[line[1]]))

            line = self.__file_ply.readline().decode().strip().split(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Read and process vertices from PLY.
    # ------------------------------------------------------------------------------------------------------------------
    def __read_vertices_binary(self):
        vertex_data = np.fromfile(self.__file_ply, dtype=self.vertex_properties, count=self.__num_vertices)
        vertices = []
        for i in range(self.__num_vertices):
            if vertex_props["nx"]["enabled"] and vertex_props["ny"]["enabled"] and vertex_props["ny"]["enabled"]:
                # get normalization coefficient for normal vectors to achieve a length of one
                norm_c = math.sqrt(pow(float(vertex_data["nx"][i]), 2) + pow(float(vertex_data["ny"][i]), 2) + pow(float(vertex_data["nz"][i]), 2))
                if norm_c != 0.0:
                    vertex_normal = np.array([float(vertex_data["nx"][i])/norm_c, float(vertex_data["ny"][i])/norm_c, float(vertex_data["nz"][i])/norm_c])
                else:
                    vertex_normal = np.array([0.0, 1.0, 0.0])
            else:
                vertex_normal = np.array([0.0, 1.0, 0.0])

            
            if vertex_props["red"]["enabled"] and vertex_props["green"]["enabled"] and vertex_props["blue"]["enabled"]:
                alpha = float(vertex_data["alpha"][i]) / 255.0 if vertex_props["alpha"] else 1.0
                vertex_color = np.array([float(vertex_data["red"][i]) / 255.0, float(vertex_data["green"][i]) / 255.0, float(vertex_data["blue"][i]) / 255.0, alpha])
            else:
                vertex_color = np.array([1.0, 1.0, 1.0, 1.0])

            if vertex_props["label"]["enabled"]:
                vertex_label = vertex_data["label"][i]
            else:
                vertex_label = 0

            vertex = Vertex(position=np.array([vertex_data["x"][i], vertex_data["y"][i], vertex_data["z"][i]]),
                            normal=vertex_normal,
                            color=vertex_color,
                            label=vertex_label,
                            idx=i)
            vertices.append(vertex)

        self.__mesh.set_vertices(np.asarray(vertices))
        return vertices

    # ------------------------------------------------------------------------------------------------------------------
    # Read and process face from PLY.
    # ------------------------------------------------------------------------------------------------------------------
    def __read_faces_binary(self, vertices):
        face_data = np.fromfile(self.__file_ply, dtype=self.face_properties, count=self.__num_faces)

        faces = []
        for i in range(self.__num_faces):
            face = Face((vertices[face_data["fx"][i]], vertices[face_data["fy"][i]], vertices[face_data["fz"][i]]))
            faces.append(face)

        self.__mesh.set_faces(np.asarray(faces))

    # ------------------------------------------------------------------------------------------------------------------
    # Read and process vertices from PLY.
    # ------------------------------------------------------------------------------------------------------------------
    def __read_vertices(self):
        vertices = []
        for i in range(self.__num_vertices):
            vertex_data = str(self.__file_ply.readline().decode("utf-8")).strip().split(" ")

            # positions, vertices and rgb colors have to be available
            # alpha channel and label are optional
            x, y, z, nx, ny, nz, r, g, b, a = vertex_data
            l = "1"
            # TEMPORAILY REMOVED
            # if vertex_props["alpha"]["enabled"]:
            #     if vertex_props["label"]["enabled"]:
            #         x, y, z, nx, ny, nz, r, g, b, a, l = vertex_data
            #     else:
            #         x, y, z, nx, ny, nz, r, g, b, a = vertex_data
            #         l = "1"
            # else:
            #     if vertex_props["label"]["enabled"]:
            #         x, y, z, nx, ny, nz, r, g, b, l = vertex_data
            #         a = "255"
            #     else:
            #         x, y, z, nx, ny, nz, r, g, b = vertex_data
            #         a = "255"
            #         l = "1"

            # get normalization coefficient for normal vectors to achieve a length of one
            norm_c = math.sqrt(pow(float(nx), 2) + pow(float(ny), 2) + pow(float(nz), 2))
            if norm_c != 0.0:
                vertex_normal = np.array([float(nx)/norm_c, float(ny)/norm_c, float(nz)/norm_c])
            else:
                vertex_normal = np.array([0.0, 1.0, 0.0])

            vertex = Vertex(position=np.array([float(x), float(y), float(z)]),
                            normal=vertex_normal,
                            color=np.array([float(r)/255.0, float(g)/255.0, float(b)/255.0, float(a)/255.0]),
                            label= int(l),
                            idx=i)
            vertices.append(vertex)

        self.__mesh.set_vertices(np.asarray(vertices))

        return vertices

    # ------------------------------------------------------------------------------------------------------------------
    # Read and process face from PLY.
    # ------------------------------------------------------------------------------------------------------------------
    def __read_faces(self, vertices):
        faces = []
        for i in range(self.__num_faces):
            _, v0, v1, v2 = str(self.__file_ply.readline().decode("utf-8")).strip().split(" ")
            face = Face((vertices[int(v0)], vertices[int(v1)], vertices[int(v2)]))
            faces.append(face)

        self.__mesh.set_faces(np.asarray(faces))

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, file_path):
        write_file = open(file_path, 'w')
        write_file.write("ply\n")
        write_file.write("format " + "ascii" + " " + str(self.__format_version) + "\n")
        write_file.write("element vertex " + str(self.__mesh.get_num_vertices()) + "\n")
        write_file.write("property float x\n")
        write_file.write("property float y\n")
        write_file.write("property float z\n")
        write_file.write("property float nx\n")
        write_file.write("property float ny\n")
        write_file.write("property float nz\n")
        write_file.write("property uchar red\n")
        write_file.write("property uchar green\n")
        write_file.write("property uchar blue\n")
        write_file.write("property uchar alpha\n")
        write_file.write("property uchar label\n")
        write_file.write("element face " + str(self.__mesh.get_num_faces()) + "\n")
        write_file.write("property list uchar int vertex_indices\n")
        write_file.write("end_header\n")

        for vertex in self.__mesh.get_vertices():
            write_file.write(str(vertex.get_position()[0]) + " " + str(vertex.get_position()[1]) + " " +
                             str(vertex.get_position()[2]) + " ")

            write_file.write(str(vertex.get_normal()[0]) + " " + str(vertex.get_normal()[1]) + " " +
                             str(vertex.get_normal()[2]) + " ")


            write_file.write(str(int(vertex.get_color()[0]*255.0)) + " " + str(int(vertex.get_color()[1]*255.0)) + " " +
                             str(int(vertex.get_color()[2]*255.0)) + " "  + str(int(vertex.get_color()[3]*255.0)) + "\n")
            # TEMPORARILY REMOVED
            #write_file.write(str(int(vertex.get_color()[0]*255.0)) + " " + str(int(vertex.get_color()[1]*255.0)) + " " +
            #                 str(int(vertex.get_color()[2]*255.0)) + " "  + str(int(vertex.get_color()[3]*255.0)) + " ")
            #write_file.write(str(vertex.get_label()) + "\n")

        for face in self.__mesh.get_faces():
            write_file.write("3 " + str(face.get_vertices()[0].get_idx()) + " " + str(face.get_vertices()[1].get_idx()) + " " + 
                             str(face.get_vertices()[2].get_idx()) + " ")

        write_file.close()


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_mesh(self):
        return self.__mesh
