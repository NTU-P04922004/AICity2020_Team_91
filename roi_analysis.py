import os
from argparse import ArgumentParser

import cv2
import numpy as np

from constants import ORIGINAL_FPS, FPS_SCALE

START_FRAME_IDX = 10 * ORIGINAL_FPS
END_FRAME_IDX = 210 * ORIGINAL_FPS


def analyze_roi(in_image_dir_path, out_dir_path, fps_scale=5, thre=1):
    imgs = os.listdir(in_image_dir_path)
    imgs.sort()

    erode_kernel = np.ones((11, 11), np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)
    w, h, _ = cv2.imread(os.path.join(in_image_dir_path, imgs[0])).shape
    ave_img = np.zeros((w, h, 3))
    for i in range(START_FRAME_IDX // fps_scale, END_FRAME_IDX // fps_scale):
        img = cv2.imread(os.path.join(in_image_dir_path, imgs[i]))
        ave_img += img.astype(np.float32)
    ave_img /= (END_FRAME_IDX - START_FRAME_IDX) // fps_scale
    ret, thresh = cv2.threshold(ave_img.astype(np.uint8), thre, 255,
                                cv2.THRESH_BINARY)
    img_new = cv2.dilate(cv2.erode(thresh, erode_kernel, iterations=1),
                         dilate_kernel,
                         iterations=3)
    cv2.imwrite(os.path.join(out_dir_path, 'mask.png'), img_new)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("image_dir_path", help="Path to foreground image directory")
    parser.add_argument("out_dir_path", help="Path for storing the result ROI mask")
    args = parser.parse_args()

    out_path = args.out_dir_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    analyze_roi(args.image_dir_path, out_path, fps_scale=FPS_SCALE)
