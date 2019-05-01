from __future__ import division

import argparse
import datetime

from matplotlib.ticker import NullLocator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from models import *
from utils.datasets import *
from utils.utils import *
from utils.iou import *
import os
from MobileNetV2 import MobileNetV2  # ref: https://github.com/tonylins/pytorch-mobilenet-v2

cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_PATH = "utils/sim_result_aligned.txt"
MAP_VID = "utils/map_vid.txt"
iou = IOU(SIM_PATH, MAP_VID, k=2)
os.makedirs('output', exist_ok=True)
cos = nn.CosineSimilarity(dim=1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def init_parser(): 
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
    parser.add_argument('--output_dir', type=str, default='', help='output of the video directory')
    parser.add_argument('--mode', type=str, default='origin', help='cropped or origin version')
    opt = parser.parse_args()
    print(opt)
    return opt

def init_model():
    # Set up model
    opt = init_parser()
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    
    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode
    rs = [model]
    if opt.mode=="cropped" or opt.mode=="psnr" or opt.mode=="net":
        min_model = Darknet(opt.config_path, img_size=320)
        min_model.load_weights(opt.weights_path)
        if cuda:
            min_model.cuda()
        min_model.eval()
        rs.append(min_model)
        if opt.mode == "net":
            mnv2 = MobileNetV2(n_class=1000)
            state_dict = torch.load('../../mobilenet_v2.pth.tar',map_location='cpu')  # add map_location='cpu' if no gpu
            mnv2.load_state_dict(state_dict)
            mnv2.classifier = Identity()
            if cuda:
                mnv2.cuda()
            rs.append(mnv2)
    
    return rs,opt

    
def plot_image(img_i, path,opt, detections,colors,dir_output,classes):
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x
    rs = []
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        object_box = open((dir_output + "/{}.txt").format(img_i), 'w')
        
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            object_box.write("{0}:{1},{2},{3},{4};".format(classes[int(cls_pred)], x1, y1, x2, y2))
            box_h = 0
            box_w = 0
            if opt.mode=="origin":
            # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            elif opt.mode=="cropped" or opt.mode=="psnr":
                box_h = y2 - y1
                box_w = x2 - x1
            rs.append([classes[int(cls_pred)], x1+box_w, x1, y1+box_h, y1])
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                                 edgecolor=color,
                                                 facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                                 bbox={'color': color, 'pad': 0})
        object_box.close()
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(dir_output + '/%d.png' % img_i, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')
    return rs


def psnr_key_frame_detect(img,crop_info,block,stride,threshold):
    # img is h,w,3
    # crop info is  x1,y1,x2,y2
    
    x1,y1,x2,y2 = crop_info
    img[x1:x2,y1:y2,:] = 0
    img = img.to(device)
    img = img.unsqueeze(0)
    img = img*img
    img = F.avg_pool3d(img,(block,block,1),stride)
    img[img==0]=1
    
    img = 255**2/img
    img = 10 * torch.log10(img)
    img = img.squeeze(0)
    img = img.squeeze(0)
    img = img.squeeze(2)
    
    idx = img < threshold
    
    img[idx] = 1
    img[~idx]=0
    size = img.shape[0]*img.shape[1]
    rr = torch.sum(img)/size
    if rr > 0.3:
        print("it is key {}".format(rr))
        return True
    return False

def net_key_frame_detect(f1,f2):
    diff = cos(f1, f2)
    if diff < 0.9:
        print("it is key {}".format(diff))
        return True
    return False

def inference_img_cropped(f,dataloader,model,min_model,opt):
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    print('\nPerforming object detection:')
    prev_time = time.time()
    total_time = datetime.timedelta(seconds=0)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    
    last_detection = None
    
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        
        if batch_i %10 == 0 or last_detection[0] is None:
            h = input_imgs.shape[1]
            w = input_imgs.shape[2]
            
            input_imgs = pad_image(input_imgs[0],opt.img_size)
            input_imgs = torch.unsqueeze(input_imgs,0)
            input_imgs = Variable(input_imgs.type(Tensor))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
                
            detections = transform_to_origin(detections,h,w,opt.img_size)
        else:
            input_imgs, crop_info,nh,nw = crop_image(img_paths,last_detection)
            input_imgs = pad_image(input_imgs,320)
            input_imgs = torch.unsqueeze(input_imgs,0)
            input_imgs = Variable(input_imgs.type(Tensor))
            with torch.no_grad():
                detections = min_model(input_imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            
            detections = transform_to_origin(detections,nh,nw,320,crop_info)
        last_detection = detections
        
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        total_time += inference_time
        prev_time = current_time
        out_string = 'Batch %d, Inference Time: %s' % (batch_i, inference_time)
        print(out_string)
        f.write(out_string + '\n')
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    f.write("Total time: " + str(total_time) + '\n')
    f.write("Average time: " + str(total_time / len(dataloader)))
    return imgs,img_detections


def inference_img_cropped_key(f,dataloader,model,min_model,opt,featuremmodel=None):
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    print('\nPerforming object detection:')
    prev_time = time.time()
    total_time = datetime.timedelta(seconds=0)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    
    last_detection = None
    last_img = None

    crop_info = (0,0,0,0)
    iskey = True
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        
        if last_img is not None:
            if opt.mode=="psnr":
                iskey = psnr_key_frame_detect(torch.abs(input_imgs.float()-last_img.float()),crop_info,20,5,10)
                last_img = input_imgs
            elif opt.mode=="mobilenet":
                # should be 32 times
                input_imgs = pad_image(input_imgs[0],320)
                cur_feature = featuremmodel(input_imgs)

                iskey = net_key_frame_detect(cur_feature,last_img)
                last_img = cur_feature
        
        if batch_i==0 or iskey or last_detection[0] is None:
            h = input_imgs.shape[1]
            w = input_imgs.shape[2]
            input_imgs = pad_image(input_imgs[0],opt.img_size)
            input_imgs = torch.unsqueeze(input_imgs,0)
            input_imgs = Variable(input_imgs.type(Tensor))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            detections = transform_to_origin(detections,h,w,opt.img_size)
        else:
            input_imgs, crop_info,nh,nw = crop_image(img_paths,last_detection)
            input_imgs = pad_image(input_imgs,320)
            input_imgs = torch.unsqueeze(input_imgs,0)
            input_imgs = Variable(input_imgs.type(Tensor))
            with torch.no_grad():
                detections = min_model(input_imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            detections = transform_to_origin(detections,nh,nw,320,crop_info)
        last_detection = detections
        
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        total_time += inference_time
        prev_time = current_time
        out_string = 'Batch %d, Inference Time: %s' % (batch_i, inference_time)
        print(out_string)
        f.write(out_string + '\n')
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    f.write("Total time: " + str(total_time) + '\n')
    f.write("Average time: " + str(total_time / len(dataloader)))
    return imgs,img_detections

def inference_img(f,dataloader,model,opt):
    prev_time = time.time()
    total_time = datetime.timedelta(seconds=0)
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            
            # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        total_time += inference_time
        prev_time = current_time
        out_string = 'Batch %d, Inference Time: %s' % (batch_i, inference_time)
        print(out_string)
        f.write(out_string + '\n')
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        f.write("Total time: " + str(total_time) + '\n')
        f.write("Average time: " + str(total_time / len(dataloader)))
    return imgs,img_detections


def run_model(opt,model):
    rootDir = opt.root_dir
    if len(model) == 1:
        model = model[0]
    elif opt.mode=="cropped" or opt.mode=="psnr":
        min_model = model[1]
        model = model[0]
    elif opt.mode=="net":
        featuremmodel = model[2]
        min_model = model[1]
        model = model[0]
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        if not subdirList:
            dir_output = opt.output_dir+"output/" + '/'.join(dirName.split("/")[-2:])
            if not os.path.exists(dir_output):
                try:
                    os.makedirs(dir_output)
                except OSError:
                    print("Create directory: {} failed".format(dir_output))
                else:
                    print("Create directory: {} successfully".format(dir_output))
            else:
                print("{} already exist.".format(dir_output))

            if opt.mode=="origin":
                dataloader = DataLoader(ImageFolder(dirName, img_size=opt.img_size),
                                    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
            elif opt.mode=="cropped" or opt.mode=="psnr" or opt.mode=="net":
                dataloader = DataLoader(croppedImageFolder(dirName, img_size=opt.img_size),
                                    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
                
            classes = load_classes(opt.class_path)  # Extracts class labels from file
            
            acc = 0
            print('\nPerforming object detection:')
            imgs = None
            img_detections = None


            with open(dir_output + "/infer_time.txt", "w") as f:
                if opt.mode == "origin":
                    imgs, img_detections =  inference_img(f,dataloader,model,opt)
                elif opt.mode=="cropped":
                    imgs, img_detections =  inference_img_cropped(f,dataloader,model,min_model,opt)
                elif opt.mode=="psnr":
                    imgs, img_detections =  inference_img_cropped_key(f,dataloader,model,min_model,opt)
                elif opt.mode=="net":
                    imgs, img_detections =  inference_img_cropped_key(f,dataloader,model,min_model,opt,featuremmodel)

            # Bounding-box colors
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]

            print('\nSaving images:')
            # Iterate through images and save plot of detections
            for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

                print("(%d) Image: '%s'" % (img_i, path))

                # Create plot
                rs = plot_image(img_i, path,opt, detections,colors,dir_output,classes)
                xmlpath = path.replace("Data","Annotations")[:-5]+".xml"

                acc += iou.frame_iou(xmlpath,rs)

                # Save generated image with detections
            with open(dir_output + "/acc.txt", "w") as f:    
                f.write("{}".format(acc/len(dataloader)))

if __name__ == "__main__":
    models, opt = init_model()
    run_model(opt,models)
