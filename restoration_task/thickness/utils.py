import glob
import os
import copy
import torch.nn.functional as F
from visualization.view_2D import plot_parallel
from skimage.transform import iradon, radon
from scipy.ndimage import zoom
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from monai.transforms import *

train_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.4, 0.5)),
        RandGridDistortion(
            padding_mode="zeros",
            mode="nearest",
            prob=0.5),
        RandGaussianSharpen(prob=0.1),
        RandGaussianSmooth(prob=0.1),
        RandHistogramShift(num_control_points=10, prob=0.2),
        RandFlip(prob=0.5),
        RandRotate90(prob=0.5)
    ]
)


class TrainSetLoader_thickness(Dataset):
    def __init__(self):
        super(TrainSetLoader_thickness, self).__init__()
        self.dataset_dir = "/data/Train/5mm"
        self.file_list = glob.glob((os.path.join(self.dataset_dir, "*.npz")))

    def __getitem__(self, index):
        # print(self.file_list[index])
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"]
        np_array = train_transforms(np_array)
        raw = torch.tensor(np_array[0:15]).cuda().to(torch.float)
        down = torch.tensor(np_array[15:]).cuda().to(torch.float)
        return down, raw

    def __len__(self):
        return len(self.file_list)
