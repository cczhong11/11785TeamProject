import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import time


def LucasKanade(It, It1, rect):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    time_start = time.time()

    It = It.squeeze(0)
    It1 = It1.squeeze(0)
    It = It[..., :3] @ torch.Tensor([0.299, 0.587, 0.114])
    It1 = It1[..., :3] @ torch.Tensor([0.299, 0.587, 0.114])

    It = It.cpu().numpy()
    It1 = It1.cpu().numpy()
    rect = rect.cpu().numpy()
    time_cpu = time.time()
    print("time cpu: {}".format(time_cpu - time_start))

    spline_it = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    spline_it1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    time_spline = time.time()
    print("time spline: {}".format(time_spline - time_cpu))

    p = []
    if len(rect.shape) == 1:
        rect = rect.reshape(1, rect.shape[0])
    for rec in rect:
        rec = rec[:4]
        p.append(calculate(spline_it, spline_it1, rec))
    
    p = np.stack(p)
    print("time calculate: {}".format(time.time() - time_spline))
    return p


def calculate(spline_it, spline_it1, rect, p0=np.zeros(2)):
    threshold = 0.001
    iter = 3
    p = p0
    rectX = rect[3] - rect[1]
    rectY = rect[2] - rect[0]
    WIt = spline_it(np.linspace(rect[1], rect[3], rectX), np.linspace(rect[0], rect[2], rectY), grid=True)
    iter_num = 0
    for i in range(iter):
        iter_num += 1
        A = np.stack((spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), dx=0, dy=1, grid=True).flatten(),
                      spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), dx=1, dy=0, grid=True).flatten()
                      )).transpose()
        WIt1 = spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), grid=True)
        b = (WIt - WIt1).flatten()

        delta_p = (np.linalg.inv(A.T @ A) @ (A.T @ b))

        p += delta_p
        if np.sum(delta_p ** 2) < threshold:
            break
    print("iter_num: {}".format(iter_num))
    return p