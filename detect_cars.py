import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import coco, register_coco_instances
from detectron2.engine import DefaultPredictor

from constants import IMG_FORMAT, IMG_NAME_FORMAT


def detect_cars(image_dir_path,
                mask_img_path,
                car_predictor,
                score_thres=0.9,
                roi_thres=0.5):

    mask_img = cv2.imread(mask_img_path)
    dilate_kernel = np.ones((5, 5), np.uint8)
    mask_img = cv2.dilate(mask_img, dilate_kernel, iterations=5)
    mask_img = mask_img.astype(np.float32) / 255

    detection_list = []
    img_array = sorted(glob(image_dir_path + "/*." + IMG_FORMAT))
    num_frames = len(img_array)
    for idx in tqdm(range(num_frames)):
        frame_id = idx + 1
        img_name = IMG_NAME_FORMAT % frame_id
        img_path = os.path.join(image_dir_path, img_name)
        img = cv2.imread(img_path)
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        num_instances = 0
        if boxes is not None:
            boxes = boxes.tensor.numpy()
            num_instances = len(boxes)
        if scores is not None:
            scores = scores.numpy()

        frame_detections = []
        if num_instances > 0:
            for i in range(num_instances):
                x1, y1, x2, y2 = boxes[i]
                score = scores[i]
                if score > score_thres:
                    box = (x1, y1, x2, y2, score)
                    masked = mask_img[int(y1):int(y2), int(x1):int(x2), :]
                    mean = np.mean(masked)
                    if mean > roi_thres:
                        frame_detections.append(box)

        detection_list.append(frame_detections)

    return detection_list


def save_detection_result(detection_list, out_file_path):
    with open(out_file_path, 'w') as f:
        for i, bbox_list in enumerate(detection_list):
            for bbox in bbox_list:
                f.write("%d,%f,%f,%f,%f,%f\n" %
                        (i + 1, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("image_dir_path", help="Path to background image directory")
    parser.add_argument("mask_base_path", help="Path to ROI mask")
    parser.add_argument("out_base_path", help="Path to store detection result")
    parser.add_argument("pretrained_model_path", help="Path to pretrained car detection model")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.WEIGHTS = args.pretrained_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    vid_name = os.path.basename(args.image_dir_path)
    mask_img_path = os.path.join(args.mask_base_path, vid_name, "mask.png")
    predictor = DefaultPredictor(cfg)
    detection_list = detect_cars(args.image_dir_path, mask_img_path, predictor)

    out_dir_path = os.path.join(args.out_base_path, "bg_detections")
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)
    out_file_path = os.path.join(out_dir_path, "bg_test_%s.txt" % vid_name)
    save_detection_result(detection_list, out_file_path)
