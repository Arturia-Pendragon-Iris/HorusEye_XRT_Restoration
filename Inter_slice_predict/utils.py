import glob
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch


def image_transfer(img, gaussian=False):
    up_down = np.random.uniform(1)
    if up_down < 0.25:
        img = img[:, ::-1]
    elif up_down < 0.5:
        img = img[::-1]
    elif up_down < 0.75:
        img = img[::-1, ::-1]

    k = np.random.uniform(0.8, 1.2)
    if gaussian:
        mu = np.random.uniform(0, 1)
        if mu > 0.5:
            k = k * (-1)
        return img * k
    else:
        return img


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = np.sort(os.listdir(dataset_dir))
        self.device = device

    def __getitem__(self, index):
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"]
        raw = np.concatenate((np_array[0:1], np_array[2:3]), axis=0)
        raw = torch.tensor(raw).to(torch.float).to(self.device)
        img = torch.tensor(np_array[1:2]).to(torch.float).to(self.device)

        return raw, img

    def __len__(self):
        return len(self.file_list)


