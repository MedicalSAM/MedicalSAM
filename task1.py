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

sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)


img_path = './data/RawData/Training/img/'
img_filenames = os.listdir(img_path)
label_path = './data/RawData/Training/label/'
label_filenames = os.listdir(label_path)

for img_num in range(0, 30):
    # read image and label
    image_nib = nib.load(img_path + img_filenames[img_num])
    image = image_nib.get_fdata().astype(np.uint8)
    x, y, z = image.shape
    label_nib = nib.load(label_path + label_filenames[img_num])
    label = label_nib.get_fdata().astype(np.uint8)
    mDice = []
    
    
    # print("img_num: ", img_num)
    # print(image.shape)
    # print(label.shape)
    # output: (512, 512, 148), (512, 512, 139). Why?
    
    z = min(image.shape[2], label.shape[2])
    
    for img_z in range(z):
        # get slice of 3D image and label
        image0 = image[:,:,img_z]
        image0 = np.stack((image0,)*3, axis=-1)
        label0 = label[:,:,img_z]
        
        sam_result = mask_generator.generate(image0)
        label0 = torch.tensor(label0)
        
        l = len(sam_result)
        # SAM will generate several masks with different performance, choose the best one here
        precison = 0
        for i in range(l):
            s = torch.tensor(sam_result[i]['segmentation'])
            precison = max(dice(s, label0), precison)       # choose the best mask
        mDice.append(precison)
        
    print("img_num: ", img_num, "mDice: ", sum(mDice)/len(mDice))
        