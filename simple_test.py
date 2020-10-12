import math
from math import floor, ceil
import random
import numpy as np
from icecream import ic


w = 512
h = 400

x1 = 0
x2 = h
y1 = 0
y2 = w

original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

max_skew_amount = max(w, h)
max_skew_amount = int(ceil(max_skew_amount))
skew_amount = random.randint(1, max_skew_amount)

# Left Tilt
new_plane = [(y1, x1 - skew_amount),  # Top Left
             (y2, x1),  # Top Right
             (y2, x2),  # Bottom Right
             (y1, x2 + skew_amount)]  # Bottom Left

matrix = []

for p1, p2 in zip(new_plane, original_plane):
    matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
    matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

m1 = np.array(matrix)
m2 = np.array(original_plane).reshape(8)

perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(m1), m2)
perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)




