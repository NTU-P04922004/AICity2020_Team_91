import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np

from constants import IMG_FORMAT, IMG_NAME_FORMAT, FRAME_FPS, FPS_SCALE


def analyze_background(in_dir_path, out_base_path, set_name="test", history_size=120):
    vid_name = os.path.basename(in_dir_path)
    out_bg_path = os.path.join(out_base_path, set_name + "_bg_imgs", vid_name)
    out_fg_path = os.path.join(out_base_path, set_name + "_fg_imgs", vid_name)
    if not os.path.exists(out_bg_path):
        os.makedirs(out_bg_path)
    if not os.path.exists(out_fg_path):
        os.makedirs(out_fg_path)

    bg = cv2.createBackgroundSubtractorMOG2()
    bg.setHistory(history_size // FPS_SCALE)

    img_array = sorted(glob(in_dir_path + "/*." + IMG_FORMAT))
    for i, img_path in enumerate(img_array):
        frame = cv2.imread(img_path)
        frame_id = i + 1

        fg_img = bg.apply(frame)
        fg_result_path = os.path.join(out_fg_path, IMG_NAME_FORMAT % frame_id)
        bg_result_path = os.path.join(out_bg_path, IMG_NAME_FORMAT % (frame_id // FRAME_FPS))
        cv2.imwrite(fg_result_path, fg_img)
        bg_img = bg.getBackgroundImage()
        if frame_id % FRAME_FPS == 0:
            cv2.imwrite(bg_result_path, bg_img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("image_dir_path", help="Path to input image directory")
    parser.add_argument("out_base_path", help="Path for storing the results")
    args = parser.parse_args()

    analyze_background(args.image_dir_path, args.out_base_path)
