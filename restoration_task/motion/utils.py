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

noise_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(128, 128),
        rotate_range=(np.pi / 6, np.pi / 6),
        scale_range=(-0.8, 0.2)),
        RandFlip(prob=0.5),
        RandRotate90(prob=0.5)
    ]
)


class TrainSetLoader_metal(Dataset):
    def __init__(self, device):
        super(TrainSetLoader_metal, self).__init__()
        self.dataset_dir = "/data/Train/metal/paired_data"
        total_list = []
        for ct_name in os.listdir(self.dataset_dir):
            total_list.append(os.path.join(self.dataset_dir, ct_name))
        random.shuffle(total_list)

        self.file_list = total_list
        print("HDCT_slice number is", len(self.file_list))
        self.device = device

    def __getitem__(self, index):
        np_array = np.load(self.file_list[index])["arr_0"]
        index_i = np.random.randint(low=0, high=np_array.shape[0])
        metal = np_array[index_i]
        gt = np_array[-1]

        # np_array = np.clip((np_array * 1600 - 600 + 200) / 500, 0, 1)
        # print(self.file_list[index])
        # np_array = refine_ct(np_array)
        # # noisy_img = simulated_noise(np_array)
        # metal_index = np.random.randint(low=0, high=len(metal_list))
        # metal = np.load(metal_list[metal_index])["arr_0"]
        # metal = zoom(metal, zoom=(1, 2), order=0)
        # # mu = np.random.uniform(low=5, high=40)
        # mu = 10
        #
        # projection = radon(np_array, theta=range(360), circle=False)
        # metal_projection = projection
        # metal_projection[106:106 + 512] += mu * metal
        # metal_recon = iradon(metal_projection, theta=range(360))[106:106 + 512, 106:106 + 512]
        #
        input = train_transforms(np.stack((metal, gt), axis=0))
        mask = 1 - np.array(input[:1] > 0.98, "float32")

        gt = torch.tensor(input[1:]).to(torch.float).to(self.device)

        metal_img = torch.tensor(np.stack([input[0]] * 3, axis=0)).to(torch.float).to(self.device)

        mask = torch.tensor(mask).to(torch.float).to(self.device)

        return metal_img, gt, mask

    def __len__(self):

        return len(self.file_list)
