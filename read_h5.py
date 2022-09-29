import enum
import h5py
import pdb
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import os
import logging
import time
import traceback
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image



cam = "dataset/camera/2016-01-30--11-24-51.h5"
# cam = "dataset/camera/2016-01-30--13-46-00.h5"

log = "dataset/log/2016-01-30--11-24-51.h5"
# log = "dataset/log/2016-01-30--13-46-00.h5"


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

class CommaaiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, cam, log, transform=None):
        """
        Args:
            cam (string): Path to the camera h5 file.
            log (string): Path to the log h5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.camera_h5 = h5py.File(cam, 'r')
        self.log_h5 = h5py.File(log, 'r')
        self.transform = transform
        self.filters = []

        self.angle = []
        steering_angle = self.log_h5["steering_angle"][:]

        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
        self.angle.append(steering_angle[idxs])

        goods = np.abs(self.angle[-1]) <= 200
        self.filters.append(np.argwhere(goods))
        
        
    def __len__(self):
        # this length is incorrect, since the data contains error in angle
        return len(self.filters[0])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # use filter_idx to select from the good angle indexs
        # self.filters is a list of two-dimensional numpy array
        # [-1] to select the numpy array, idx to select the position in the array, [0] to extract only the idx number
        
        filter_idx = self.filters[-1][idx][0]
        image = self.camera_h5["X"][filter_idx]
        
        # image = self.camera_h5["X"][idx]
        sample = (image, self.angle[-1][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_image(data):
    data = data.swapaxes(0, 1)
    data = data.swapaxes(1, 2)
    # pdb.set_trace()

    data = Image.fromarray(data, 'RGB')
    data.save('my.png')


if __name__ == "__main__":

    c5 = h5py.File(cam, 'r')
    t5 = h5py.File(log, 'r')
    # pdb.set_trace()

    angle_array = []
    filters = []
    lastidx = 0
    time_len = 1

    x = c5["X"]
    steering_angle = t5["steering_angle"][:]
    
    idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment

    angle_array.append(steering_angle[idxs])

    # move this to later, you should remove the images and the errorneous label as a pair
    goods = np.abs(angle_array[-1]) <= 200
    filters.append(np.argwhere(goods))
    lastidx += goods.shape[0]
    
    # check for mismatched length bug

    print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], angle_array[-1].shape[0]))
    if x.shape[0] != angle_array[-1].shape[0]:
        raise Exception("bad shape")

    
    comm_dataset = CommaaiDataset(cam, log)
    
    train_length = int(0.8 * len(comm_dataset))
    val_length = len(comm_dataset) - train_length
    trainset, valset = torch.utils.data.random_split(comm_dataset, [train_length, val_length])
    assert (train_length + val_length) == len(comm_dataset), "not match length"

    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True, num_workers=4)

    # pdb.set_trace()
    total = 0
    for i, (imgs, angles) in enumerate(train_loader):
        total += len(imgs)
        if total % 500 == 0:
            print(total)
    
    print(f"Total images: {total}")