import glob
import os
import copy
import random
from skimage.filters import frangi
from torchvision.transforms import transforms
from analysis.motion import *
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
        translate_range=(32, 32),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.2, 0.2)),
        RandFlip(prob=0.5),
        RandRotate90(prob=0.5)
    ]
)


class TrainSetLoader_motion(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_motion, self).__init__()

        total_list = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
        random.shuffle(total_list)

        self.file_list = total_list
        print("HDCT_slice number is", len(self.file_list))

    def __getitem__(self, index):
        np_array = np.load(self.file_list[index])["arr_0"]
        np_array = train_transforms(np_array[np.newaxis])[0]
        mu1 = np.random.uniform(low=0, high=1)
        mu2 = np.random.uniform(low=0, high=1)

        amplitude = np.random.uniform(low=1, high=5, size=(1, 2))
        fre = np.random.randint(low=1, high=3, size=(1, 2))
        trans = np.random.uniform(low=1, high=5, size=(1, 2))
        rotate = np.random.uniform(low=0.5, high=2.5, size=(1, 2))
        if mu1 < 0.5:
            motion_img = generate_grid_motion(np_array, trans, rotate)
            if mu2 < 0.5:
                motion_img = generate_nongrid_motion(motion_img, amplitude, fre)
        else:
            motion_img = generate_nongrid_motion(np_array, amplitude, fre)
            if mu2 > 0.5:
                motion_img = generate_grid_motion(motion_img, trans, rotate)

        gt = torch.tensor(np_array[np.newaxis]).to(torch.float).cuda()
        motion_img = torch.tensor(np.stack([motion_img] * 3, axis=0)).to(torch.float).cuda()
        return motion_img, gt

    def __len__(self):
        return len(self.file_list)
