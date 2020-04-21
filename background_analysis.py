import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np

IMG_FORMAT = "png"
IMG_NAME_FORMAT = "frame-%05d.png"
ORIGINAL_FPS = 30
FRAME_FPS = 5
FPS_SCALE = ORIGINAL_FPS // FRAME_FPS


def analyze_background(in_dir_path, out_base_path, set_name="test", history_size=120):
    save_bg_path = os.path.join(out_base_path, set_name + "_bg_imgs")
    save_fg_path = os.path.join(out_base_path, set_name + "_fg_imgs")
    if not os.path.exists(save_bg_path):
        os.makedirs(save_bg_path)
    if not os.path.exists(save_fg_path):
        os.makedirs(save_fg_path)

    vid_name = os.path.basename(in_dir_path)
    if True:
        save_bg_path_ = os.path.join(save_bg_path, vid_name)
        save_fg_path_ = os.path.join(save_fg_path, vid_name)
        if not os.path.exists(save_bg_path_):
            os.makedirs(save_bg_path_)
        if not os.path.exists(save_fg_path_):
            os.makedirs(save_fg_path_)

        bg = cv2.createBackgroundSubtractorMOG2()
        bg.setHistory(history_size // FPS_SCALE)

        img_array = sorted(glob(in_dir_path + "/*." + IMG_FORMAT))
        for i, img_path in enumerate(img_array):
            frame = cv2.imread(img_path)
            frame_id = i + 1

            fg_img = bg.apply(frame)
            fg_result_path = os.path.join(save_fg_path_, IMG_NAME_FORMAT % frame_id)
            bg_result_path = os.path.join(save_bg_path_, IMG_NAME_FORMAT % (frame_id // FRAME_FPS))
            cv2.imwrite(fg_result_path, fg_img)
            bg_img = bg.getBackgroundImage()
            if frame_id % FRAME_FPS == 0:
                cv2.imwrite(bg_result_path, bg_img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("image_dir_path", help="Path of input image directory")
    parser.add_argument("out_base_path", help="Base path for storing the results")
    args = parser.parse_args()

    analyze_background(args.image_dir_path, args.out_base_path)
