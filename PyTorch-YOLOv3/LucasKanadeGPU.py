from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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


def get_grid(x0, x1, y0, y1, X, Y, imh, imw):
    tx = torch.linspace(x0, x1, X)
    ty = torch.linspace(y0, y1, Y)
    grid_xy = torch.stack(torch.meshgrid(tx, ty), dim=2).unsqueeze(0)
    grid_xy[:, :, :, 0] = (grid_xy[:, :, :, 0] - imh / 2) / (imh / 2)
    grid_xy[:, :, :, 1] = (grid_xy[:, :, :, 1] - imw / 2) / (imw / 2)
    return grid_xy


def calculate(It, It1, rect):
    threshold = 0.001
    iter = 3
    p = torch.zeros(2)
    rectX = int(rect[3] - rect[1])
    rectY = int(rect[2] - rect[0])
    rect = torch.Tensor(rect)

    img_tensor = torch.Tensor(It).float().unsqueeze(0)
    img1_tensor = torch.Tensor(It1).float().unsqueeze(0)

    img_tensor = img_tensor[..., :3] @ torch.Tensor([0.299, 0.587, 0.114])
    img1_tensor = img1_tensor[..., :3] @ torch.Tensor([0.299, 0.587, 0.114])

    img_height = img_tensor.size(2)
    img_width = img_tensor.size(3)
    img1_height = img1_tensor.size(2)
    img1_width = img1_tensor.size(3)

    grid_xy0 = get_grid(rect[0], rect[2], rect[1], rect[3], rectY, rectX, img_width, img_height)
    m_WIt = F.grid_sample(img_tensor, grid=grid_xy0).permute(0, 1, 3, 2)
    for i in range(iter):

        grid_xy = get_grid(rect[0] + p[0], rect[2] + p[0], rect[1] + p[1], rect[3] + p[1], rectY, rectX, img1_width, img1_height)

        sample_img = F.grid_sample(img1_tensor, grid=grid_xy).permute(0, 1, 3, 2)

        sp_y_sobel = sobel(sample_img, 'y')
        sp_x_sobel = sobel(sample_img, 'x')

        m_A = torch.stack((sp_y_sobel.view(-1), sp_x_sobel.view(-1))).transpose(1, 0)

        m_b = (m_WIt - sample_img).contiguous().view(-1)

        #delta_pp = torch.inverse(torch.Tensor(A).permute(1, 0) @ torch.Tensor(A)) @ (torch.Tensor(A).permute(1, 0) @ m_b)
        delta_p = torch.inverse(m_A.permute(1, 0) @ m_A) @ (m_A.permute(1, 0) @ m_b)
        p += delta_p
        if torch.sum(delta_p ** 2) < threshold:
            break
    return p


def LucasKanadeGPU(It, It1, rect):
    p = []
    if len(rect.shape) == 1:
        rect = rect.reshape(1, rect.shape[0])
    for rec in rect:
        rec = rec[:4]
        p.append(calculate(It, It1, rec))

    p = np.stack(p)
    return p