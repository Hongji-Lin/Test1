# encoding: utf-8
# @author: Evan
# @file: test.py
# @time: 2022/11/8 11:36
# @desc: 数据增强测试
import math
import os
from random import random

import cv2
import numpy as np
import torch.cuda
from torchvision.transforms import transforms
from cutout import Cutout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 颜色噪声变化 = HSV + 噪声 + 模糊
# HSV变换
# 色域空间增强Augment colorspace：H色调、S饱和度、V亮度
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
    return im_hsv


# 变暗
def Darker(image,percetage=0.5):
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image


# 明亮
def Brighter(image, percetage=1.5):
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image


# 高斯噪声
def gaussian_noise(image):
    h,w,c = image.shape
    for row in range(h):
        for col in range(w):
            # 获取三个高斯随机数
            # 第一个参数：概率分布的均值，对应着整个分布的中心
            # 第二个参数：概率分布的标准差，对应于分布的宽度
            # 第三个参数：生成高斯随机数数量
            s = np.random.normal(0,50,3)
            # 获取每个像素点的bgr值
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            # 给每个像素值设置新的bgr值
            image[row,col,0] = np.clip(b+s[0],0,255)
            image[row,col,1] = np.clip(g+s[1],0,255)
            image[row,col,2] = np.clip(r+s[2],0,255)
    return image


# 高斯模糊
def gaussian_blur(image):
    image = cv2.GaussianBlur(image, (5, 5), 3)
    return image


# 空间几何变换

# 放大缩小
def Scale(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


# 水平翻转
def Horizontal(image):
    return cv2.flip(image, 1, dst=None) # 水平镜像


# 旋转，R可控制图片放大缩小
def Rotate(image, angle=30, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    # rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image


# 平移
def Move(image, x, y):
    img_info = image.shape
    height = img_info[0]
    width = img_info[1]

    mat_translation = np.float32([[1, 0, x], [0, 1, y]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列
    # [[1,0,20],[0,1,50]]   表示平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离。
    img = cv2.warpAffine(image, mat_translation, (width, height))  # 变换函数
    return img


# Cutout
def augment_cutout(image):
    img_tensor = transforms.ToTensor()(image)
    cut = Cutout(n_holes=1, length=100)  # n_holes=1, length=16
    img_cutout = cut(img_tensor)
    img_cutout = img_cutout.mul(255).byte()
    img_cutout = img_cutout.numpy().transpose((1, 2, 0))
    return img_cutout


def data_aug(img_path, save_img):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_path = "./test/"
    save_path = "./test/"
    img_list = os.listdir(save_path)
    print(img_list)
    for file_name in img_list:
        file_i_path = os.path.join(save_path, file_name)
        print(file_i_path)
        img_i = cv2.imread(file_i_path)

        print("数据增强开始")
        img_hsv = augment_hsv(img_i, hgain=0.5, sgain=0.5, vgain=0.5)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_hsv.jpg"),  img_hsv)

        img_dark = Darker(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_dark.jpg"), img_dark)

        img_bright = Brighter(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_bright.jpg"), img_bright)

        img_noise = gaussian_noise(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_noise.jpg"),  img_noise)

        img_blur = gaussian_blur(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_blur.jpg"), img_blur)

        img_horizon = Horizontal(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_horizon.jpg"), img_horizon)

        img_rotate = Rotate(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_rotate.jpg"), img_rotate)

        img_move = Move(img_i, x=120, y=120)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_move.jpg"), img_move)

        img_cutout = augment_cutout(img_i)
        cv2.imwrite(os.path.join(save_path, file_name.split('.')[0] + "_cutout.jpg"), img_cutout)

        print("{}完成".format(file_name))


if __name__ == "__main__":
    img_path = "./test/"
    save_path = "./test/"
    data_aug(img_path, save_path)


