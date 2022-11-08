# encoding: utf-8
# @author: Evan
# @file: randomerasing.py
# @time: 2022/11/8 16:05
# @desc: 随机擦除数据增强


from __future__ import absolute_import
import math
from random import random

from torchvision.transforms import *
import numpy as np
import torch


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability  # 随机概率
        self.mean = mean  # 归一化
        self.sl = sl  # 长
        self.sh = sh  # 高
        self.r1 = r1  # 率

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:  # 随机生成一个实数>概率值
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]  # 长宽

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)  # 随机的像素值代替原始像素值0.3-3.3（黑色-）

            h = int(round(math.sqrt(target_area * aspect_ratio)))  # round上取整，
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:  # 满足遮挡框小于图片大小时
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:  # 3通道RGB 图像范围0-255时，归一化,例cifar
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:  # 单通道归一化,例fashionmnist
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
