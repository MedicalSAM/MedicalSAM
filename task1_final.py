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

# SAM model
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import argparse

sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

parser = argparse.ArgumentParser()
parser.add_argument('--points_num', type=int, default=1, help='Number of points as prompt')
parser.add_argument('--if_box', type=int, default=0, help='1: use box as prompt')
args = parser.parse_args()

points_num = args.points_num
if_box = args.if_box


img_path = './data/RawData/Training/img/'
img_filenames = sorted(os.listdir(img_path))
label_path = './data/RawData/Training/label/'
label_filenames = sorted(os.listdir(label_path))

for img_num in range(0, 30):
    # read image and label
    image_nib = nib.load(img_path + img_filenames[img_num])
    image = image_nib.get_fdata().astype(np.uint8)
    x, y, z = image.shape
    label_nib = nib.load(label_path + label_filenames[img_num])
    label = label_nib.get_fdata().astype(np.uint8)
    mDice = []
    
    z = image.shape[2]
    
    for img_z in range(z):
        # get slice of 3D image and label
        image0 = image[:,:,img_z]
        image0 = np.stack((image0,)*3, axis=-1)
        label0 = label[:,:,img_z]
        
        dices = []
        
        predictor = SamPredictor(sam)
        predictor.set_image(image0)
        
        for i in range(1, 14):
            
            org_label = np.where(label0 == i, 1, 0)
            nonzero_indices = np.argwhere(org_label == 1)
            
            # Randomly select 3 indices from the non-zero indices, if no non-zero indices, skip
            if len(nonzero_indices) < points_num:
                continue
            random_indices = np.random.choice(nonzero_indices.shape[0], size=points_num, replace=False)
            
            # Get the coordinates of the randomly selected indices
            input_point = nonzero_indices[random_indices]
            
            input_label = np.array([1]*points_num)
            
            
            # Get the bounding box of the selected indices
            if if_box == 1:
                if len(nonzero_indices) < 10:   # if the number of non-zero indices is too small, skip
                    continue
                x_min = np.min(nonzero_indices[:, 0])
                y_min = np.min(nonzero_indices[:, 1])
                x_max = np.max(nonzero_indices[:, 0])
                y_max = np.max(nonzero_indices[:, 1])
                
                bbox = np.array([[x_min-3, y_min-3, x_max+3, y_max+3]])
            else:
                bbox = None
        
        
            mask_max, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=bbox,
                multimask_output=False,
            )
        
        
            org_label, mask_max = torch.tensor(org_label), torch.tensor(mask_max)
            dices.append(dice(mask_max, org_label))
        if len(dices) > 0:
            # print(img_z, sum(dices)/len(dices), dices)
            mDice.append(sum(dices)/len(dices))

    print(img_num, sum(mDice)/len(mDice))
        
        
        
        
        
        

        