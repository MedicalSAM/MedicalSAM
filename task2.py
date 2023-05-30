import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from nibabel.viewers import OrthoSlicer3D

from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

import nibabel as nib
import imageio
import os

import torchmetrics
from torchmetrics.functional import dice
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

import argparse

# SAM model
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


# Settings
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=device)

num_classes = 14 # or however many classes you have

# sam_model.mask_decoder = torch.nn.Conv3d(
#     in_channels=sam_model.mask_decoder.in_channels,
#     out_channels=num_classes,
#     kernel_size=sam_model.mask_decoder.kernel_size,
#     stride=sam_model.mask_decoder.stride,
#     padding=sam_model.mask_decoder.padding,
#     dilation=sam_model.mask_decoder.dilation,
#     groups=sam_model.mask_decoder.groups,
#     bias=sam_model.mask_decoder.bias,
#     padding_mode=sam_model.mask_decoder.padding_mode
# )


optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters()) 
# loss_fn = dice(average='macro')

# create a SummaryWriter object to write to a log directory
log_dir = './logs'
writer = SummaryWriter(log_dir)


img_path = './data/RawData/Training/img/'
img_filenames = os.listdir(img_path)
label_path = './data/RawData/Training/label/'
label_filenames = os.listdir(label_path)



parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train (default: 10)')
args = parser.parse_args()

num_epochs = args.num_epochs

def loss_fn(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


# Training loop

for epoch in range(num_epochs):
    for img_num in range(24):
        # load image and label
        image_nib = nib.load(img_path + img_filenames[img_num])
        image = image_nib.get_fdata().astype(np.uint8)
        x, y, z = image.shape
        label_nib = nib.load(label_path + label_filenames[img_num])
        label = label_nib.get_fdata().astype(np.uint8)

        # convert to tensor and move to device
        image_tensor = torch.from_numpy(image).to(device=device).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).to(device=device).unsqueeze(0).long()

        # forward pass
        output = sam_model(image_tensor)

        # compute loss and Dice
        loss = loss_fn(output, label_tensor)
        dice_accuracy = dice(output, label_tensor)
        
        # write to SummaryWriter
        writer.add_scalar('train/loss', loss.item(), 24 * epoch + img_num)
        writer.add_scalar('train/dice_accuracy', dice_accuracy, 24 * epoch + img_num)


        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for img_num in range(25, 30):
            # load image and label
            image_nib = nib.load(img_path + img_filenames[img_num])
            image = image_nib.get_fdata().astype(np.uint8)
            x, y, z = image.shape
            label_nib = nib.load(label_path + label_filenames[img_num])
            label = label_nib.get_fdata().astype(np.uint8)

            # convert to tensor and move to device
            image_tensor = torch.from_numpy(image).to(device=device).unsqueeze(0).float()
            label_tensor = torch.from_numpy(label).to(device=device).unsqueeze(0).long()

            # forward pass
            output = sam_model(image_tensor)

            # compute loss and dice
            val_loss += loss_fn(output, label_tensor).item()
            val_dice += dice(F.threshold(F.normalize(output), dim=1, threshold=0.5), label_tensor).item()

        val_loss /= 6
        val_dice /= 6
        
        # draw validation loss and accuracy
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/dice_accuracy', val_dice, epoch)


# save model and close SummaryWriter
writer.close()
torch.save(sam_model.state_dict(), "./checkpoints/fine_tuned_model.pth")