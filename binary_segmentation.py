import os
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import segmentation_models_pytorch as smp

images_dir = os.environ.get("INPUT_DIR")
masks_dir = os.environ.get("MASK_DIR")
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config['device'] = str(device)

class BinaryDataset:

    classes = [
        "sem_vegetacao",
        "com_vegetacao"
        ]
    
    def __init__(
            self,
            images_dir,
            masks_dir,
            txt_file):
        
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        with open(txt_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
    
    def __getitem__(self):
        img_path = os.path.join(self.images_dir, self.image_names)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    pass