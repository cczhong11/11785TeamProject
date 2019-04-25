#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from utils.datasets import ImageFolder
from torch.autograd import Variable
import argparse
import torch
import time
import datetime
from MobileNetV2 import MobileNetV2  # ref: https://github.com/tonylins/pytorch-mobilenet-v2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    parser.add_argument('--root_dir', type=str, default='../../videos', help='root of the video directory')
    opt = parser.parse_args()
    print(opt)
    cuda = torch.cuda.is_available() and opt.use_cuda

    mnv2 = MobileNetV2(n_class=1000)
    state_dict = torch.load('mobilenet_v2.pth.tar')  # add map_location='cpu' if no gpu
    mnv2.load_state_dict(state_dict)
    mnv2.classifier = Identity()
    if cuda:
        mnv2.cuda()

    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    if cuda:
        model.cuda()

    dirName = "/home/andrewdeeplearningisawesome/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00098000"
    dataloader = DataLoader(ImageFolder(dirName, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    prev_feature = None
    cos = nn.CosineSimilarity(dim=1)
    prev_time = time.time()
    total_time = datetime.timedelta(seconds=0)
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        cur_feature = mnv2(input_imgs)
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        total_time += inference_time
        print('Batch %d, Inference Time: %s' % (batch_i, inference_time))
        prev_time = current_time
        if batch_i == 0:
            prev_feature = cur_feature
            print("keyframe", batch_i)
            continue
        diff = cos(prev_feature, cur_feature)
        print(batch_i, diff)
        if diff < 0.7:
            print("keyframe", batch_i)
            print(batch_i, diff)
            prev_feature = cur_feature
    print("Average time: " + str(total_time / len(dataloader)))
