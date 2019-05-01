from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.interpolate import RectBivariateSpline
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

P = transforms.Compose([transforms.ToPILImage()])


def sobel(x, direction='x'):
    # x: 1 * 1 * H * W
    # print(x.shape)
    a = None
    if direction == 'x':
        a = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().unsqueeze(0).unsqueeze(0)
    elif direction == 'y':
        a = torch.Tensor([[1, 2, 1], [0, 0, 0],[-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0)
    else:
        assert False
    gradient = F.conv2d(x, a, stride=1, padding=1).squeeze(0).squeeze(0)
    return gradient


def test_sobel():
    frames = np.load('../data/carseq.npy')
    img_0 = frames[:, :, 0]
    print(img_0.shape)
    fig, ax = plt.subplots(1)
    ax.imshow(img_0, cmap='gray')
    plt.show()

    gradient_x = sobel(img_0, 'x')
    gradient_y = sobel(img_0, 'y')

    img_x = P(gradient_x)
    img_x.save('grad_x.png')

    img_y = P(gradient_y)
    img_y.save('grad_y.png')


def get_grid(x0, x1, y0, y1, X, Y, imh, imw):
    tx = torch.linspace(x0, x1, X)
    ty = torch.linspace(y0, y1, Y)
    grid_xy = torch.stack(torch.meshgrid(tx, ty), dim=2).unsqueeze(0)

    grid_xy[:, :, :, 0] = (grid_xy[:, :, :, 0] - imh / 2) / (imh / 2)
    grid_xy[:, :, :, 1] = (grid_xy[:, :, :, 1] - imw / 2) / (imw / 2)

    return grid_xy


def LucasKanadeGPU(It, It1, rect, p0=torch.Tensor(np.zeros(2))):
    threshold = 0.0001
    iter = 100
    # print(It.shape, It1.shape)
    p = p0
    rectX = rect[3] - rect[1]
    rectY = rect[2] - rect[0]
    spline_it = RectBivariateSpline(range(It.shape[0]), range(It.shape[1]), It)
    spline_it1 = RectBivariateSpline(range(It1.shape[0]), range(It1.shape[1]), It1)
    img_tensor = torch.Tensor(It).unsqueeze(0).unsqueeze(0)
    img1_tensor = torch.Tensor(It1).unsqueeze(0).unsqueeze(0)
    img_height = img_tensor.size(2)
    img_width = img_tensor.size(3)
    img1_height = img1_tensor.size(2)
    img1_width = img1_tensor.size(3)

    for i in range(iter):
        nx = np.linspace(rect[1] + p[1], rect[3] + p[1], rectX)
        ny = np.linspace(rect[0] + p[0], rect[2] + p[0], rectY)

        grid_xy = get_grid(rect[1] + p[1], rect[3] + p[1],
                           rect[0] + p[0], rect[2] + p[0],
                           rectX, rectY, img1_height, img1_width)
        sample_img = F.grid_sample(img1_tensor, grid=grid_xy)
        # sample_img = torch.Tensor(spline_it1(nx, ny)).unsqueeze(0).unsqueeze(0)

        sp_y = spline_it1(nx, ny, dx=0, dy=1, grid=True)
        sp_x = spline_it1(nx, ny, dx=1, dy=0, grid=True)
        # print(sp_y, sp_x)

        sp_y_sobel = sobel(sample_img, 'y')
        sp_x_sobel = sobel(sample_img, 'x')
        # print(sp_y_sobel, sp_x_sobel)

        A = np.stack((sp_y.flatten(), sp_x.flatten())).transpose()
        # print(A.shape)

        m_A = torch.stack((sp_y_sobel.view(-1), sp_x_sobel.view(-1))).transpose(0, 1)
        # print(m_A.shape)

        WIt = spline_it(np.linspace(rect[1], rect[3], rectX), np.linspace(rect[0], rect[2], rectY), grid=True)
        WIt1 = spline_it1(np.linspace(rect[1] + p[1], rect[3] + p[1], rectX), np.linspace(rect[0] + p[0], rect[2] + p[0], rectY), grid=True)
        b = (WIt - WIt1).flatten()
        # print(b.shape)

        grid_xy0 = get_grid(rect[1], rect[3], rect[0], rect[2], rectX, rectY, img_height, img_width)
        m_WIt = F.grid_sample(img_tensor, grid=grid_xy0)
        m_b = (m_WIt - sample_img).view(-1)
        # print(m_b.shape)

        delta_p = (np.linalg.inv(A.T @ A) @ (A.T @ b))
        delta_pp = torch.inverse(m_A.transpose(0, 1) @ m_A) @ (m_A.transpose(0, 1) @ m_b)
        # print(delta_p, delta_pp)

        p += delta_pp
        # print(np.sum(delta_p ** 2))
        if np.sum(delta_p ** 2) < threshold:
            break
    return p


def test_LucasKanadeGPU():
    frames = np.load('../data/carseq.npy')
    img_0 = frames[:, :, 0]
    img_1 = frames[:, :, 1]
    fig, ax = plt.subplots(1)
    # Create a Rectangle patch
    rec_0 = [59, 116, 145, 151]
    rect = patches.Rectangle((rec_0[0], rec_0[1]), rec_0[2] - rec_0[0], rec_0[3] - rec_0[1], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.imshow(img_0, cmap='gray')
    ax.add_patch(rect)
    plt.show()
    print(LucasKanadeGPU(img_0, img_1, rec_0))
    print(LucasKanade(img_0, img_1, rec_0))


if __name__ == '__main__':
    test_LucasKanadeGPU()