# encoding: utf-8
# @author: Evan/Hongji Lin
# @file: TEST2.PY
# @time: 2022/11/8 21:19
# @desc:
import os
import cv2
import numpy as np
import torch





img_path = 'test/'
img_list = os.listdir(img_path)
print(img_list)
full_height = []
full_width = []
empty_height = []
empty_width = []
for img in img_list:
    img_name = cv2.imread((img_path + img))
    height = img_name.shape[0]
    width = img_name.shape[1]

    full_height.append(height)
    full_width.append(width)

# a = torch.tensor(full_height).float()
# full_height_mean = torch.mean(a, 0, True)

mean = np.mean(full_height)
std = np.std(full_height)
print(mean)

print(std)


print(full_height)
print(full_width)