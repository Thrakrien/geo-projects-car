import os
import torch
# from dotenv import load_dotenv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from segmentation_models_pytorch import Unet
from torch.utils.data import Dataset, DataLoader

from src.dataset import PatchifySegmentationDataset, PatchifyInference
from src.preprocessing import PreProcessingImage

torch.manual_seed(42)
np.random.seed(42)

img_sat = "/data/integracar/amostras_car_orotofoto/"
img_mask = "/data/integracar/amostras_car_mask/"
train_txt = "data-segments/train_sample.txt"
val_txt = "data-segments/validation_sample.txt"


class UnetImporter():
    def __init__(self, num_classes, loss_function, optimizer):
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def model_information(self):
        model = Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to("cuda")

        if self.loss_function == "bce": 
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss_function == "cross-entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss Function não inserida")
        
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.01,
                weight_decay=0.005,
                momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.01,
                weight_decay=0.005)
        else:
            raise ValueError("Loss Function não inserida")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            # verbose=True
        )

        return model, criterion, optimizer, scheduler
    
    def train_loader(self):

        train_transform = PreProcessingImage(transform_type="train")

        train_dataset = PatchifySegmentationDataset(
            txt_file=train_txt,
            images_dir=img_sat,
            masks_dir=img_mask,
            transform=train_transform.transformations()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        return train_loader

    def validation_loader(self):
        validation_transform = PreProcessingImage(transform_type="validation")

        validation_dataset = PatchifySegmentationDataset(
            txt_file=val_txt,
            images_dir=img_sat,
            masks_dir=img_mask,
            transform=validation_transform.transformations()
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        return validation_loader