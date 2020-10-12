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

original_plane = np.array([[0, 0],
                           [w, 0],
                           [w, h],
                           [0, h]])

alter_plane = np.array([(0, 0),
                        (w, 0),
                        (w, h),
                        (0, h)])

ic(original_plane)
ic(alter_plane)

