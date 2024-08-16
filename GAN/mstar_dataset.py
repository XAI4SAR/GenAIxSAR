import read_dataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import random
import datetime
from data_transform import OneHot
import math
from PIL import Image
import cv2 as cv


class MSTAR_Dataset(Dataset):
    """
    MSTAR dataset
    """

    def __init__(self, txt_file, transform = None ):
        """
        Args:
            :param txt_file: path to the txt file with ($path$ $label$)
            :param transform: optional transform to be applied on a sample
            
        """
        self.txt_file = txt_file
        self.mstar_path_label = read_dataset.read_dataset_txt(txt_file)

        self.transform = transform

        
    def __len__(self):
        return len(self.mstar_path_label)

    def __getitem__(self, idx):
        """
        Args:
            :param idx: the index of data

            :return:
        """

        patch_path =  self.mstar_path_label[idx][0] 
        
        l_split = patch_path.split('/')
        name_ex = l_split[-1]
        l_name = name_ex.rsplit('.',1)
        name = l_name[0]
        
        data = np.load(patch_path)
        img= abs(data['comp'])
        img = (img-img.min())/(img.max()-img.min())
        img = img*255*3
        img = np.clip(img,0,255)
        img = img.astype(np.uint8)
        # img = np.power(img, 0.5) 
        
        az = abs(data['TargetAz'])
        label = self.mstar_path_label[idx][1]
        sample = {'image' : img, 'name' : name, 'label' : label, 'az' : az }
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

