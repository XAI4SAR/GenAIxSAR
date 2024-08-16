
import numpy as np
import cv2
import os
from PIL import Image

def read_dataset_txt(txt_file):
    list_path_label = {}
    f_txt = open(txt_file)
    lines_txt = f_txt.readlines()

    for count, l in enumerate(lines_txt):
        l = l.strip('\n')  # 移除字符串首尾的换行符
        l = l.rstrip()
        l_split = l.split()
        l_path = l_split[0]
        l_label = l_split[1]

        list_path_label[count] = [l_path, int(l_label)]
    return list_path_label


def read_jpg(patch_path):
    patch = Image.open(patch_path).convert('L')
    return patch

def read_npy(patch_path):
    patch = np.load(patch_path)
    return patch