import torch
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split, DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from my_models import NetworkNvidia
from torch.utils.tensorboard import SummaryWriter
import sys
from utils import CommaaiDatasetSpeed, CommaaiDatasetSpeedMultiWorker
import torch.autograd.profiler as profiler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE=256
NUM_WORKERS=16

# values are manually collected
DATASET_MEAN = -4.6224138113878865
DATASET_STD = 35.17810663695777
DEVICE = 'cuda'


CAM_PATHS = [
    'dataset/camera/2016-01-30--11-24-51.h5',
    'dataset/camera/2016-01-30--13-46-00.h5',
    'dataset/camera/2016-01-31--19-19-25.h5',
    'dataset/camera/2016-02-02--10-16-58.h5',
    'dataset/camera/2016-02-08--14-56-28.h5',
    'dataset/camera/2016-02-11--21-32-47.h5',
    'dataset/camera/2016-03-29--10-50-20.h5',
    'dataset/camera/2016-04-21--14-48-08.h5',
    'dataset/camera/2016-05-12--22-20-00.h5',
    'dataset/camera/2016-06-02--21-39-29.h5',
    'dataset/camera/2016-06-08--11-46-01.h5', 
  ]

LOG_PATHS = [
    'dataset/log/2016-01-30--11-24-51.h5',
    'dataset/log/2016-01-30--13-46-00.h5',
    'dataset/log/2016-01-31--19-19-25.h5',
    'dataset/log/2016-02-02--10-16-58.h5',
    'dataset/log/2016-02-08--14-56-28.h5',
    'dataset/log/2016-02-11--21-32-47.h5',
    'dataset/log/2016-03-29--10-50-20.h5',
    'dataset/log/2016-04-21--14-48-08.h5',
    'dataset/log/2016-05-12--22-20-00.h5',
    'dataset/log/2016-06-02--21-39-29.h5',
    'dataset/log/2016-06-08--11-46-01.h5', 
  ]

def load_datasets(cam_paths, log_paths):
    """Load and concatenate datasets from camera and log paths."""
    # comm_datasets = [CommaaiDatasetSpeed(cam, log) for cam, log in zip(cam_paths, log_paths)]
    # update the loader to support multiple worker
    comm_datasets = [CommaaiDatasetSpeedMultiWorker(cam, log) for cam, log in zip(cam_paths, log_paths)]
    return ConcatDataset(comm_datasets)

def create_data_loaders(dataset, batch_size):
    """Create data loaders for training and validation."""
    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length
    trainset, valset = random_split(dataset, [train_length, val_length])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader

def train(model, train_loader, optimizer, criterion, scheduler, dataset_mean, dataset_std, device, writer, epoch):
    """Training process for one epoch."""
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    for batch_idx, (imgs, angles) in enumerate(train_loader):
        imgs, angles = imgs.float().to(device), angles.float().to(device)
        angles_normalized = (angles - dataset_mean) / dataset_std

        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, angles_normalized.reshape(len(imgs), -1))

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        if batch_idx % 20 == 0:
            logger.debug(f"Epoch {epoch} | Batch {batch_idx} / {total_batches} | Loss: {train_loss / (batch_idx + 1):.6f}")

    writer.add_scalar("Loss/train", train_loss / (batch_idx + 1), epoch)

def profile_train(model, train_loader, optimizer, criterion, scheduler, dataset_mean, dataset_std, device, writer, epoch):
    """Training process for one epoch."""
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    with profiler.profile(use_cuda=True) as prof:

        for batch_idx, (imgs, angles) in enumerate(train_loader):
            imgs, angles = imgs.float().to(device), angles.float().to(device)
            angles_normalized = (angles - dataset_mean) / dataset_std

            optimizer.zero_grad()
            outputs, _ = model(imgs)
            loss = criterion(outputs, angles_normalized.reshape(len(imgs), -1))

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                logger.debug(f"Epoch {epoch} | Batch {batch_idx} / {total_batches} | Loss: {train_loss / (batch_idx + 1):.6f}")
        writer.add_scalar("Loss/train", train_loss / (batch_idx + 1), epoch)
        print(prof.key_averages().table(sort_by="cuda_time_total"))


def validate(model, val_loader, criterion, dataset_mean, dataset_std, device, writer, epoch):
    """Validation process for one epoch."""
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, (imgs, angles) in enumerate(val_loader):
            imgs, angles = imgs.float().to(device), angles.float().to(device)
            angles_normalized = (angles - dataset_mean) / dataset_std

            outputs, _ = model(imgs)
            loss = criterion(outputs, angles_normalized.reshape(len(imgs), -1))
            valid_loss += loss.item()

            if batch_idx % 20 == 0:
                logger.debug(f"Validation Batch {batch_idx} | Loss: {valid_loss / (batch_idx + 1):.6f}")

    writer.add_scalar("Loss/validation", valid_loss / (batch_idx + 1), epoch)


def main():
    assert len(sys.argv) >= 2, "Workspace argument is missing."
    workspace = sys.argv[1]
    writer = SummaryWriter(comment=workspace)

    dataset = load_datasets(CAM_PATHS, LOG_PATHS)
    train_loader, val_loader = create_data_loaders(dataset, BATCH_SIZE)

    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://')

    model = NetworkNvidia().to(DEVICE)
    model = DDP(model)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[65535])
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        logger.info(f"Starting epoch {epoch}")
        train(model, train_loader, optimizer, criterion, scheduler, DATASET_MEAN, DATASET_STD, DEVICE, writer, epoch)
        validate(model, val_loader, criterion, DATASET_MEAN, DATASET_STD, DEVICE, writer, epoch)


if __name__ == "__main__":
    main()