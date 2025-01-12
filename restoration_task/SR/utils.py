import glob
import os
import copy
import random
from skimage.filters import frangi
from scipy.ndimage import zoom
import cv2
from visualization.view_2D import plot_parallel
from skimage.transform import iradon, radon
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandAffine)

train_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.4, 0.5)),
        RandFlip(prob=0.5),
        RandRotate90(prob=0.5)
    ]
)


class TrainSetLoader_SR(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_SR, self).__init__()

        total_list = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
        random.shuffle(total_list)

        self.file_list = total_list
        print("HDCT_slice number is", len(self.file_list))

    def __getitem__(self, index):
        np_array = np.load(self.file_list[index])["arr_0"]
        np_array = train_transforms(np_array[np.newaxis])[0]

        down_img = cv2.resize(np_array, None, fx=0.25, fy=0.25)

        gt = torch.tensor(np_array[np.newaxis]).to(torch.float).cuda()
        down_img = torch.tensor(np.stack([down_img] * 3, axis=0)).to(torch.float).cuda()
        return down_img, gt

    def __len__(self):
        return len(self.file_list)
