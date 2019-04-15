import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    threshold = 0.001
    iter = 100
    spline_it = RectBivariateSpline(range(It.shape[0]), range(It.shape[1]), It)
    spline_it1 = RectBivariateSpline(range(It1.shape[0]), range(It1.shape[1]), It1)
    p = p0
    rectX = rect[3] - rect[1]
    rectY = rect[2] - rect[0]
    for i in range(iter):

        A = np.stack((spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), dx=0, dy=1, grid=True).flatten(),
                      spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), dx=1, dy=0, grid=True).flatten()
                      )).transpose()
        WIt = spline_it(np.linspace(rect[1], rect[3], rectX), np.linspace(rect[0], rect[2], rectY), grid=True)
        WIt1 = spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), grid=True)
        b = (WIt - WIt1).flatten()

        delta_p = (np.linalg.inv(A.T @ A) @ (A.T @ b))

        p += delta_p
        print(np.sum(delta_p ** 2))
        if np.sum(delta_p ** 2) < threshold:
            break
    return p