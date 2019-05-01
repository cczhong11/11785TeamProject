# this code is use PSNR to find the move region in the picture
# first use cpu to detect the part of region and plot the change of psnr
# write CNN version
# write detect version
# TODO:find suitable dataset

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--block', type=int, default=20, help='size')
parser.add_argument('--stride', type=int, default=5, help='stride')
parser.add_argument('--path', type=str, default=".", help='image folder')

opt = parser.parse_args()
print(opt)

def get_psnr(img):
    #img is 2D np array
    imax = 255
    d = 1
    
    for l in img.shape:
        d = d*l
    mse = 1/d *np.sum(img**2)
    if mse == 0:
        return 1000
    psnr = 10*np.log10(imax**2/mse)
    return psnr

def split_block(img1_path,img2_path,block,stride):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img1 = np.array(img1)
    img2 = np.array(img2)
    img = img2 - img1
    h,w,_ = img1.shape
    
    block_h_num = int((h-block)/stride)+1
    block_w_num = int((w-block)/stride)+1
    
    psnr = np.zeros((block_h_num,block_w_num))
    for i in range(block_h_num):
        for j in range(block_w_num):
            start_x = i*stride
            start_y = j*stride
            end_x = start_x+block
            end_y = start_y + block
            if end_x > h:
                end_x = h
            if end_y > w:
                end_y = w
            psnr[i,j] = get_psnr(img[start_x:end_x,start_y:end_y,:])
    
    return psnr

def cpu_test():
    rs = []
    blocksize = 100
    step = 5
    for i in range(80):
        p = split_block(Root+"{}.JPEG".format(str(i).zfill(2)),Root+"{}.jpeg".format(str(i+1).zfill(2)),blocksize,step)
        idx = p < 20
    
        p[idx] = 1
        p[~idx]=0
        plt.imshow(p)
        plt.savefig("{}".format(i))
        #rs.append(p[0,10])

def get_psnr_gpu(img,block,stride):
    img = img*img
    img = F.avg_pool2d(img,block,stride)
    img[img==0]=1
    print(img)
    img = 255/img
    img = 10 * torch.log10(img)
    idx = img < 20
    
    img[idx] = 1
    img[~idx]=0
    return img

def gpu_test():
    last_image = None
    for f in os.listdir(opt.path):
        if last_image==None:
            last_image = Image.open(opt.path+f)
            continue
        cur_image = Image.open(opt.path+f)
        img1 = np.array(last_image)
        img2 = np.array(cur_image)
        img = img2 - img1
        img = torch.from_numpy(img).float()
        img = torch.mean(img,dim=2)
        r = get_psnr_gpu(img.unsqueeze(0),opt.block,opt.stride)
        print(torch.sum(r))

gpu_test()