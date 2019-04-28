from __future__ import division

import argparse
import datetime

from matplotlib.ticker import NullLocator
from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
from utils.iou import *
import os

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
parser.add_argument('--resize', type=float, default=3, help='resize size')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)
SIM_PATH = "utils/sim_result_aligned.txt"
MAP_VID = "utils/map_vid.txt"
# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
min_model = Darknet(opt.config_path, img_size=320)
model.load_weights(opt.weights_path)
min_model.load_weights(opt.weights_path)

if cuda:
    model.cuda()
    min_model.cuda()
iou = IOU(SIM_PATH, MAP_VID, k=2)
model.eval()  # Set in evaluation mode
min_model.eval()
rootDir = opt.root_dir
for dirName, subdirList, fileList in os.walk(rootDir):
    if not subdirList:
        dir_output = "output/" + '/'.join(dirName.split("/")[-2:])
        if not os.path.exists(dir_output):
            try:
                os.makedirs(dir_output)
            except OSError:
                print("Create directory: {} failed".format(dir_output))
            else:
                print("Create directory: {} successfully".format(dir_output))
        else:
            print("{} already exist.".format(dir_output))

        dataloader = DataLoader(croppedImageFolder(dirName, img_size=opt.img_size),
                                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print('\nPerforming object detection:')
        prev_time = time.time()
        total_time = datetime.timedelta(seconds=0)
        with open(dir_output + "/infer_time.txt", "w") as f:
            last_image_path  = None
            last_detection = None
            for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                # Configure input
                
                if batch_i %10 == 0 or len(last_detection[0])==0:
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

        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print('\nSaving images:')
        acc = 0
        # Iterate through images and save plot of detections
        acc = 0
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
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
                    rs.append([classes[int(cls_pred)], x2, x1, y2, y1])
                    # Rescale coordinates to original dimensions
                    box_h = y2 - y1
                    box_w = x2 - x1
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
            xmlpath = path.replace("Data","Annotations")[:-5]+".xml"
            
            acc+=iou.frame_iou(xmlpath,rs)
            
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(dir_output + '/%d.png' % img_i, bbox_inches='tight', pad_inches=0.0)
            plt.close('all')
        print(acc/len(dataloader))
