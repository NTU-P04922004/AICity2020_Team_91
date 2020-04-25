import os
from argparse import ArgumentParser

import cv2


def gen_all_frame_detection_list(det_file_path, image_count, scale=1):

    all_frame_detection_list = [None] * image_count
    with open(det_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(',')
            frame_id = int(tokens[0])
            x0 = float(tokens[1]) * scale
            y0 = float(tokens[2]) * scale
            x1 = float(tokens[3]) * scale
            y1 = float(tokens[4]) * scale
            score = float(tokens[5])
            detection_info = (x0, y0, x1, y1, score)

            if frame_id <= image_count:
                if all_frame_detection_list[frame_id - 1] is None:
                    all_frame_detection_list[frame_id - 1] = []
                local_frame_detections = all_frame_detection_list[frame_id - 1]
                local_frame_detections.append(detection_info)

    return all_frame_detection_list


def draw_bboxes(img, frame_detection_list, score_thres=0.5):
    HIGH_PROB_COLOR = (0, 255, 0)
    MID_PROB_COLOR = (200, 0, 0)
    LOW_PROB_COLOR = (0, 0, 200)
    for detection in frame_detection_list:
        if detection is not None:
            x0, y0, x1, y1, score = detection
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            color = MID_PROB_COLOR
            if score > score_thres:
                color = HIGH_PROB_COLOR
            elif score < 0.2:
                color = LOW_PROB_COLOR
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)


def visualize_detection_results(img_base_path, out_dir_path, detection_list, score_thres=0.5, img_name_format='frame-%05d.png'):
    num_frames = len(detection_list)
    for i in range(num_frames):
        frame_id = i + 1
        img_name = img_name_format % frame_id
        img_path = os.path.join(img_base_path, img_name)
        img = cv2.imread(img_path)
        if frame_id <= len(detection_list) and detection_list[i] is not None:
            frame_detections = detection_list[i]
            draw_bboxes(img, frame_detections, score_thres)
            out_path = os.path.join(out_dir_path, img_name)
            cv2.imwrite(out_path, img)


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("detection_file_path", help="")
#     parser.add_argument("image_dir_path", help="")
#     args = parser.parse_args()

#     detection_list = []
#     img_count = len(os.listdir(args.image_dir_path))
#     detection_list = gen_all_frame_detection_list(args.detection_file_path, img_count)

#     tmp_path = "/home/kuohsin/workspace/dataset/AIC20_track4/video_prediction/tmp/"
#     dir_name = os.path.basename(args.detection_file_path).replace(".txt", "")
#     out_path = os.path.join(tmp_path, dir_name)
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     visualize_detection_results(args.image_dir_path, out_path, detection_list)
