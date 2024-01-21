import glob
import os
import copy
from torchvision.transforms import transforms
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

noise_dir = "/noise_file"
noise_list = []
for noise_name in os.listdir(noise_dir):
    noise_path = os.path.join(noise_dir, noise_name)
    sub_noise_list = os.listdir(noise_path)
    sub_noise_list = [os.path.join(noise_path, sub_noise_list[x]) for x in range(len(sub_noise_list))]
    noise_list.extend(sub_noise_list)

noise_num = len(noise_list)
print(noise_num)


def refine_noise(noise_array):
    noise_array = noise_array - np.mean(noise_array)
    noise_array = np.clip(noise_array, -0.3, 0.3)

    # mu = np.random.normal(loc=0, scale=0.025)
    # mu = np.random.normal(loc=1, scale=0.1)
    mu = np.random.uniform(low=0.001, high=0.05)

    noise_array = noise_array * mu / np.mean(np.abs(noise_array))

    # if np.random.uniform(low=0, high=1) < 0.5:
    #     noise_array = noise_array * (-1)

    if noise_array.shape != (512, 512):
        # new_array = np.zeros([512, 512])
        h = int((512 - noise_array.shape[0]) / 2)
        w = int((512 - noise_array.shape[1]) / 2)
        new_array = noise_array[h:h + 512, w:w + 512]

        return new_array

    return noise_array

def refine_ct(ct_array):
    window = np.random.uniform(low=0, high=0.9)
    if window < 0.2:
        k = np.random.randint(low=-1200, high=-800)
        mu = np.random.randint(low=1200, high=1600)
        ct_array = np.clip((ct_array - k) / mu, 0, 1)
    elif window < 0.4:
        k = np.random.randint(low=-250, high=-150)
        mu = np.random.randint(low=300, high=600)
        ct_array = np.clip((ct_array - k) / mu, 0, 1)
    elif window < 0.6:
        k = np.random.randint(low=-260, high=-200)
        mu = np.random.randint(low=250, high=400)
        ct_array = np.clip((ct_array - k) / mu, 0, 1)
    elif window < 0.8:
        k = np.random.randint(low=-300, high=-100)
        mu = np.random.randint(low=800, high=1200)
        ct_array = np.clip((ct_array - k) / mu, 0, 1)
    else:
        k = np.random.randint(low=-50, high=-10)
        mu = np.random.randint(low=100, high=140)
        ct_array = np.clip((ct_array - k) / mu, 0, 1)

    return ct_array


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir

        total_list = []
        for i in os.listdir(dataset_dir):
            i_path = os.path.join(dataset_dir, i)
            sub_list = os.listdir(i_path)
            sub_list = [os.path.join(i_path, sub_list[x]) for x in range(len(sub_list))]
            total_list.extend(sub_list)

        self.file_list = total_list
        print("HDCT_slice number is", len(self.file_list))
        self.device = device

    def __getitem__(self, index):
        # print(self.file_list[index])
        raw_array = np.load(self.file_list[index])["arr_0"]
        # raw_array = np.clip((raw_array + 1000) / 1600, 0, 1.5)
        raw_array = refine_ct(raw_array)
        raw_array = np.clip(train_transforms(raw_array[np.newaxis]), 0, 1)[0]

        noise_index = np.random.randint(low=0, high=noise_num)
        noise = refine_noise(np.load(noise_list[noise_index])["arr_0"])
        # print(noise_list[noise_index])

        while np.sum(np.abs(noise)) < 10 ** (-5) or noise.shape != (512, 512):
            noise_index = np.random.randint(low=0, high=noise_num)
            noise = refine_noise(np.load(noise_list[noise_index])["arr_0"])

        noise = noise_transforms(noise[np.newaxis])
        # noise = np.clip(noise, -0.5, 0.5)
        # noisy_array = noise + raw_array
        raw_array = raw_array[np.newaxis]
        if np.random.randint(low=0, high=1) < 0.5:
            noisy_array = noise + raw_array
        else:
            noisy_array = iradon(radon(noise, circle=False)
                                 + radon(raw_array, circle=False))[106:106 + 512, 106:106 + 512]

        raw_tensor = torch.tensor(raw_array).clone().to(torch.float).to(self.device)
        noisy_tensor = torch.tensor(noisy_array).clone().to(torch.float).to(self.device)
        return raw_tensor, noisy_tensor

    def __len__(self):
        return len(self.file_list)


def shift_generator():
    k = np.random.randint(low=0, high=20000)
    root_path = "/pattern_drift"
    file_list = os.listdir(root_path)
    shift_drift = np.load(os.path.join(root_path, file_list[k]))["arr_0"]

    return shift_drift


