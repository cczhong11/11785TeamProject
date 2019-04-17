#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from collections import defaultdict

XML_PATH = "sample.xml"
SIM_PATH = "sim_result_aligned.txt"
MAP_VID = "map_vid.txt"
SAMPLE_DETECT_RES = [['cat', 240.0, 71.0, 372.0, 272.0], ['dog', 51.0, 9.0, 289.0, 251.0]]


class IOU:
    def __init__(self, sim_path, map_vid, k=2):
        """
        :param sim_path: the path of the "sim_result_aligned.txt"
        :param map_vid: the path of the "map_vid.txt"
        :param k: a threshold. If the number of detected objects in two frames' are larger than k. The iou of those
        two frames will be regarded as 0.
        """
        self.k = k
        self.sim_dict = defaultdict(lambda: defaultdict())
        with open(sim_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.split()
                for idx, val in enumerate(info):
                    if idx % 2 == 0:
                        continue
                    # self.sim_dict[info[0]].append([info[idx], float(info[idx + 1])])
                    self.sim_dict[info[0]][info[idx]] = float(info[idx + 1])
        self.str_2_label = defaultdict()
        with open(map_vid, 'r') as f:
            lines = f.readlines()
            for line in lines:
                result = line.split()
                self.str_2_label[result[0]] = result[2]

    @staticmethod
    def bbox_iou(box1, box2):
        """
        Returns the IoU of two bounding boxes
        """
        box1_xmax, box1_xmin, box1_ymax, box1_ymin = box1
        box2_xmax, box2_xmin, box2_ymax, box2_ymin = box2
        if box1_xmin >= box2_xmax or box2_xmin >= box1_xmax:
            return 0
        if box1_ymin >= box2_ymax or box2_ymin >= box1_ymax:
            return 0
        inter_x_max = min(box1_xmax, box2_xmax)
        inter_x_min = max(box1_xmin, box2_xmin)
        inter_y_max = min(box1_ymax, box2_ymax)
        inter_y_min = max(box1_ymin, box2_ymin)
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)
        box2_area = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)
        return inter_area / (box1_area + box2_area - inter_area)

    def frame_iou(self, xml_path: str, detect_res: list) -> float:
        """
        Given the xml path of ILSVRC2015, and a list of detection result of YOLOv3
        xml format see sample.xml.
        detect_res: [[label:xmax,xmin,ymax,ymin], ...]
        :param xml_path: xml path of ILSVRC2015/
        :param detect_res: detection result.
        :return: frame iou. There is no definition of frame iou. High result means high similarity.
        """
        xml_boxes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                for info in child:
                    if info.tag == 'name':
                        xml_boxes.append([self.str_2_label[info.text]])
                    elif info.tag == 'bndbox':
                        for coord in info:
                            xml_boxes[-1].append(float(coord.text))
        if abs(len(xml_boxes) - len(detect_res)) >= self.k:
            return 0.0
        # guarantee xml_boxes has less or equal number of objects than detect_res
        if len(xml_boxes) > len(detect_res):
            xml_boxes, detect_res = detect_res, xml_boxes
        filter_detect_res = []  # find the objects pair with highest iou
        for o1 in xml_boxes:
            max_iou_res = float('-inf')
            for o2 in detect_res:
                cur_res = self.bbox_iou(o1[1:], o2[1:])
                if cur_res > max_iou_res:
                    if max_iou_res == float('-inf'):
                        filter_detect_res.append(o2)
                    else:
                        filter_detect_res[-1] = o2
                    max_iou_res = cur_res
        iou_res = []
        for o1, o2 in zip(xml_boxes, filter_detect_res):
            cur_res = self.bbox_iou(o1[1:], o2[1:])
            sim_weight = self.sim_dict[o1[0]][o2[0]]
            iou_res.append(cur_res * sim_weight)
            print(cur_res, sim_weight)
        return sum(iou_res) / len(iou_res)


if __name__ == '__main__':
    iou = IOU(SIM_PATH, MAP_VID, k=2)
    res = iou.frame_iou(XML_PATH, SAMPLE_DETECT_RES)
    print(res)
