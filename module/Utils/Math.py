"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import math
from numba import cuda
import random


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# DEVICE METHODS
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
@cuda.jit(device=True)
def device_mul_mat3x3_vec3(mat3x3, vec3):
    m00 = mat3x3[0][0] * vec3[0] + mat3x3[0][1] * vec3[1] + mat3x3[0][2] * vec3[2]
    m01 = mat3x3[1][0] * vec3[0] + mat3x3[1][1] * vec3[1] + mat3x3[1][2] * vec3[2]
    m02 = mat3x3[2][0] * vec3[0] + mat3x3[2][1] * vec3[1] + mat3x3[2][2] * vec3[2]

    return m00, m01, m02


# ----------------------------------------------------------------------------------------------------------------------
# Find the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
#
# Args:
#    axis (list): rotation axis of the form [x, y, z]
#    theta (float): rotational angle in radians
#
# Returns:
#    array. Rotation matrix.
#
# Source:
# - https://www.andre-gaschler.com/rotationconverter/
# - https://en.wikipedia.org/wiki/Rotation_matrix
# ----------------------------------------------------------------------------------------------------------------------
def get_3x3rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    # transform <axis> to unit vector
    axis = axis/math.sqrt(np.dot(axis, axis))

    t = 1.0 - math.cos(theta)
    c = math.cos(theta)
    s = math.sin(theta)
    n1 = axis[0]
    n2 = axis[1]
    n3 = axis[2]

    return np.array([[n1*n1*t+c,    n1*n2*t-n3*s, n1*n3*t+n2*s],
                     [n2*n1*t+n3*s, n2*n2*t+c,    n2*n3*t-n1*s],
                     [n3*n1*t-n2*s, n3*n2*t+n1*s, n3*n3*t+c]])


# ----------------------------------------------------------------------------------------------------------------------
# Creates a random 3x3 rotation matrix based on get_3x3rotation_matrix(axis, theta)
# ----------------------------------------------------------------------------------------------------------------------
def get_random_3x3rotation_matrix():
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    z = random.uniform(-1.0, 1.0)
    # check if the the vector is not a zero vector
    if x == 0.0 and y == 0.0 and z == 0.0:
        raise ValueError("Can't create rotation matrix with zero vector.")
    random_vec = [x, y, z]
    random_angle = random.uniform(0.0, 2*math.pi)
    return get_3x3rotation_matrix(random_vec, random_angle)
