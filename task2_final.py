from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

from nibabel.viewers import OrthoSlicer3D

from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

import nibabel as nib
import imageio
import os

import torch
import torchmetrics
from torchmetrics.functional import dice
from torch.autograd import Variable

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import json

from segment_anything import SamPredictor, sam_model_registry

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate, default=1e-4')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs to train for, default=1000')
args = parser.parse_args()

img_path = './data/RawData/Training/img/'
img_filenames = sorted(os.listdir(img_path))
label_path = './data/RawData/Training/label/'
label_filenames = sorted(os.listdir(label_path))

bbox_coords = {}
ground_truth_mask = {}
images = {}
data_count = 0

# Process data input, assume GPU memory is enough
for img_num in range(0, 30):
    # read image and label
    image_nib = nib.load(img_path + img_filenames[img_num])
    image = image_nib.get_fdata().astype(np.uint8)
    x, y, z = image.shape
    label_nib = nib.load(label_path + label_filenames[img_num])
    label = label_nib.get_fdata().astype(np.uint8)
    
    for h in range(z):
        image0 = image[:, :, h]
        label0 = label[:, :, h]
        
        for i in range(1, 14):
            org_label = np.where(label0 == i, 1, 0)
            nonzero_indices = np.argwhere(org_label == 1)
            if len(nonzero_indices) < 10:   # if the number of non-zero indices is too small, skip
                continue
            x_min = np.min(nonzero_indices[:, 0])
            y_min = np.min(nonzero_indices[:, 1])
            x_max = np.max(nonzero_indices[:, 0])
            y_max = np.max(nonzero_indices[:, 1])
                
            bbox = np.array([[x_min-3, y_min-3, x_max+3, y_max+3]])
            
            bbox_coords[data_count] = bbox
            ground_truth_mask[data_count] = org_label
            images[data_count] = image0
            data_count += 1

# Load model
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=device)
sam_model.train()

# We convert the input images into a format SAM's internal functions expect.
# Preprocess the images
from collections import defaultdict

import torch

from segment_anything.utils.transforms import ResizeLongestSide

transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
    image = images[k]
    image = np.stack([image, image, image], axis=-1)
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size
  
  
# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn1 = torch.nn.MSELoss()
# loss_fn1 = torch.nn.BCELoss()
def loss_fn(pred, target):
  return 1 - dice(pred, target)
keys = list(bbox_coords.keys())


# Running the training loop
from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    for k in keys:
        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']
        
        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            
            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        
        
        low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        )
        

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        #binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        binary_mask = torch.norm(upscaled_masks)
        binary_mask = upscaled_masks.div(binary_mask.expand_as(upscaled_masks))

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask[k], (1, 1, ground_truth_mask[k].shape[0], ground_truth_mask[k].shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        
        loss = loss_fn1(binary_mask, gt_binary_mask)
        
        loss.retain_grad()
        binary_mask.retain_grad()
        upscaled_masks.retain_grad()
        low_res_masks.retain_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        epoch_losses.append(loss.item())
    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')