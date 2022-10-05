from cmath import nan
import enum
from genericpath import exists
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from my_models import NetworkNvidia
from torch.utils.tensorboard import SummaryWriter
import sys

cam = "dataset/camera/2016-01-30--11-24-51.h5"
log = "dataset/log/2016-01-30--11-24-51.h5"


# cam = "dataset/camera/2016-01-31--19-19-25.h5"
# log = "dataset/log/2016-01-31--19-19-25.h5"


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

        # pdb.set_trace()
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
        sample = (image, self.angle[-1][filter_idx])

        if self.transform:
            sample = self.transform(sample)

        return sample


class CommaaiDatasetSpeed(Dataset):
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
        
        self.speed = []
        all_speed = self.log_h5["speed"][:]

        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
        self.angle.append(steering_angle[idxs])

        self.speed.append(all_speed[idxs])

        # pdb.set_trace()
        # goods = np.abs(self.angle[-1]) <= 200
        goods = np.abs(self.speed[-1]) >= 15 # following yuenan hou's aaai paper
        self.filters.append(np.argwhere(goods))
        
    def __len__(self):

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
        sample = (image, self.angle[-1][filter_idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

def save_image(data, path, output):
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    data = data.swapaxes(0, 1)
    data = data.swapaxes(1, 2)
    data = Image.fromarray(data, 'RGB')
    # data.save('dataset_images/' + str(output) + '.png')
    data.save(path + str(output) + '.png')


if __name__ == "__main__":

    c5 = h5py.File(cam, 'r')
    t5 = h5py.File(log, 'r')
    # pdb.set_trace()

    angle_array = []
    speed_array = []
    filters = []
    lastidx = 0
    time_len = 1

    x = c5["X"]
    steering_angle = t5["steering_angle"][:]
    all_speed = t5["speed"][:]

    speed = t5["speed"]
    
    idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment

    angle_array.append(steering_angle[idxs])

    speed_idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
    speed_array.append(all_speed[speed_idxs])

    # move this to later, you should remove the images and the errorneous label as a pair
    
    # goods = np.abs(angle_array[-1]) <= 200
    goods = np.abs(speed_array[-1]) >= 15

    extracted = angle_array * goods
    good_angles = extracted[extracted != 0]

    mean_angles = np.mean(good_angles)
    std_angles = np.std(good_angles)

    filters.append(np.argwhere(goods))
    # lastidx += goods.shape[0]
    
    # check for mismatched length bug
    print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], angle_array[-1].shape[0]))
    if x.shape[0] != angle_array[-1].shape[0]:
        raise Exception("bad shape")

    # comm_dataset = CommaaiDataset(cam, log)
    comm_dataset = CommaaiDatasetSpeed(cam, log)
    # pdb.set_trace()
    # pdb.set_trace()    
    # for i, ele in enumerate(comm_dataset):
    #     save_image(ele[0], '2016-01-31--19-19-25-image-speed/', str(i))
    # pdb.set_trace()

    print(comm_dataset[0][0].shape)
    train_length = int(0.8 * len(comm_dataset))
    val_length = len(comm_dataset) - train_length
    trainset, valset = torch.utils.data.random_split(comm_dataset, [train_length, val_length])
    assert (train_length + val_length) == len(comm_dataset), "not match length"

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4) # for debug
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    # total = 0
    # for i, (imgs, angles) in enumerate(train_loader):
    #     total += len(imgs)
    #     if total % 500 == 0:
    #         print(total)
    
    # print(f"Total images: {total}")
    # ll = LambdaLayer(lambda x: x ** 2)
    # print(ll)
    # input = torch.tensor([1, 2, 3, 4])
    # out = ll(input)
    # print(out)

    # model = CommaModel()
    model = NetworkNvidia()
    print(model)

    device = 'cuda'
    epochs = 300
    lr = 1e-4
    weight_decay = 1e-5
    # lr = 1e-3

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[65535])
    criterion = nn.MSELoss()

    if len(sys.argv) < 1:
        print("Forgot the add workspace?")
    workspace = str(sys.argv[1])
    # print(workspace)

    writer = SummaryWriter(comment=workspace)

    """Training process."""
    model.to(device)
    for epoch in range(epochs):
        print(f"Starting epochs {epoch}")
        
        # Training
        train_loss = 0
        model.train()

        for local_batch, (imgs, angles) in enumerate(train_loader):
            
            # Transfer to GPU

            imgs = imgs.float().to(device) 
            angles = angles.float().to(device)
            angles_normalized = (angles - mean_angles) / std_angles

            optimizer.zero_grad()
            outputs, _ = model(imgs)

            #angles = angles.reshape(len(imgs), -1)
            angles_normalized = angles_normalized.reshape(len(imgs), -1)
            loss = criterion(outputs, angles_normalized)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print(f"loss after backward {loss}")

            train_loss += loss.data.item()
             
            if local_batch % 20 == 0:
                # print("Training Epoch: {} | RMSE: {} | MAE: {}".format(epoch, train_loss / (local_batch + 1), train_loss_mae / (local_batch + 1)))
                print("Training Epoch: {} | Loss: {} ".format(epoch, train_loss / (local_batch + 1)))

        writer.add_scalar("Loss/train", train_loss / (local_batch + 1), epoch)

        # Validation
        model.eval()
        valid_loss = 0
        valid_loss_mae = 0
        with torch.set_grad_enabled(False):
            for local_batch, (imgs, angles) in enumerate(val_loader):
                
                # Transfer to GPU
                imgs = imgs.float().to(device) 
                angles = angles.float().to(device)
                angles_normalized = (angles - mean_angles) / std_angles
                
                # Model computations
                optimizer.zero_grad()
                outputs, _ = model(imgs)

                #loss = criterion(outputs, angles.unsqueeze(0))
                # pdb.set_trace()
                angles_normalized = angles_normalized.reshape(len(imgs), -1)
                loss = criterion(outputs, angles_normalized)
                
                valid_loss += loss.data.item()

                if local_batch % 20 == 0:
                    # print("Validation RMSE: {} | Validation MAE: {}".format(valid_loss / (local_batch + 1), valid_loss_mae/(local_batch + 1)))
                    print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))
        writer.add_scalar("Loss/validation", valid_loss / (local_batch + 1), epoch)