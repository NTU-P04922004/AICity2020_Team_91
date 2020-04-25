import math
import os
from argparse import ArgumentParser

from tqdm import tqdm

import utils


def get_iou(box1, box2):
    ix1, iy1, ix2, iy2, _ = box1
    x1, y1, x2, y2, _ = box2
    iarea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    xx1 = max(ix1, x1)
    yy1 = max(iy1, y1)
    xx2 = min(ix2, x2)
    yy2 = min(iy2, y2)
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (iarea + area - inter)
    return ovr


def compute_suspected_car_list(detection_list, iou_thres=0.7):
    suspected_car_list = []
    num_frames = len(detection_list)
    for i in tqdm(range(num_frames, 0, -1)):
        frame_id = i + 1
        if frame_id <= len(detection_list) and detection_list[i] is not None:
            frame_detections = detection_list[i]
            for i, detection in enumerate(frame_detections):
                if detection is not None:
                    has_already_existed = False
                    for i, car_info in enumerate(suspected_car_list):
                        x0, y0, x1, y1, start, end = car_info
                        box = (x0, y0, x1, y1, 1)
                        iou = get_iou(box, detection)
                        if iou > iou_thres:
                            if start - frame_id == 1:
                                suspected_car_list[i] = (x0, y0, x1, y1, frame_id, end)
                                has_already_existed = True
                                break

                    if not has_already_existed:
                        x0, y0, x1, y1, score = detection
                        new_car_info = (x0, y0, x1, y1, frame_id, frame_id)
                        suspected_car_list.append(new_car_info)

    return suspected_car_list


def compute_anomaly_duration(suspected_car_list, frame_count, thres=100):
    anomaly_frames = [False] * (frame_count + 1)
    result_list = []
    for car_info in suspected_car_list:
        start = car_info[4]
        end = car_info[5]
        if (end - start) > thres:
            for i in range(start, end+1):
                anomaly_frames[i] = True
    start_idx = -1
    end_idx = -1
    for i, val in enumerate(anomaly_frames):
        if start_idx == -1:
            if val:
                start_idx = i
                end_idx = i
        else:
            if val:
                end_idx = i
            else:
                result_list.append((start_idx, end_idx))
                start_idx = end_idx = -1

    if start_idx != -1 and end_idx != -1:
        result_list.append((start_idx, end_idx))

    return result_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("detection_file_path", help="Path to car detection list file")
    parser.add_argument("image_dir_path", help="Path to background image directory")
    args = parser.parse_args()

    detection_list = []
    img_count = len(os.listdir(args.image_dir_path))
    detection_list = utils.gen_all_frame_detection_list(args.detection_file_path, img_count)

    suspected_car_list = compute_suspected_car_list(detection_list)
    results = compute_anomaly_duration(suspected_car_list, img_count)

    detection_filename = os.path.basename(args.detection_file_path)
    vid = detection_filename[8:-4] # detection_filename has pattern: bg_test_vid.txt
    for interval in results:
        print("%s %d %d" % (vid, interval[0], 1.0))
