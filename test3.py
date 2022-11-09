# encoding: utf-8
# @author: Evan
# @file: test3.py
# @time: 2022/11/9 11:12
# @desc:
import os

import cv2
import numpy as np


def cal_mean():
    full_fileDir = "./test/"
    # empty_fileDir = "./test/"
    full_list = os.listdir(full_fileDir)
    # empty_list = os.listdir(empty_fileDir)
    img_height = []
    img_width = []

    for full_img in full_list:
        full_img = cv2.imread((full_fileDir + full_img))
        h = full_img.shape[0]
        w = full_img.shape[1]

        img_height.append(h)
        img_width.append(w)

        # for emp_img in empty_list:
        #     emp_img = cv2.imread((empty_fileDir + emp_img))
        #     h = emp_img.shape[0]
        #     w = emp_img.shape[1]

        img_height.append(h)
        img_width.append(w)

    h_mean = int(np.mean(img_height))
    w_mean = int(np.mean(img_width))
    print(h_mean)
    print(w_mean)
    return h_mean, w_mean


def Scale(image):
    h, w = cal_mean()
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    imgs = cv2.imread("./test/0.jpg")
    print(imgs.shape)
    img = Scale(imgs)
    print(img.shape)