import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import json
import logging
import csv
from datetime import datetime
from patchify import patchify, unpatchify

logger_here = logging.getLogger(__name__)


class PatchifySegmentationDataset(Dataset):
    """
    Dataset para segmentação semântica com suporte a patchify
    Divide imagens 1024x1024 em patches 512x512
    """
    def __init__(self, txt_file, images_dir, masks_dir, 
                 image_size=1024, patch_size=512, 
                 transform=None, use_patches=True):
        """
        Args:
            txt_file (string): Caminho para o arquivo .txt com os nomes das imagens
            images_dir (string): Diretório com as imagens de entrada
            masks_dir (string): Diretório com as máscaras/labels
            image_size (int): Tamanho da imagem original (1024)
            patch_size (int): Tamanho dos patches (512)
            transform (bool): Aplica transformações como rotações e color jitter
            use_patches (bool): Se True, divide em patches. Se False, usa imagem completa
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.use_patches = use_patches
        
        with open(txt_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        self.patches_per_row = image_size // patch_size
        self.patches_per_image = self.patches_per_row ** 2
        
        if use_patches:
            self.total_patches = len(self.image_names) * self.patches_per_image
        else:
            self.total_patches = len(self.image_names)
        
        print(f"Dataset: {len(self.image_names)} imagens")
        if use_patches:
            print(f"Patches por imagem: {self.patches_per_image} ({self.patches_per_row}x{self.patches_per_row})")
            print(f"Total de patches: {self.total_patches}")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        if self.use_patches:
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
            
            img_name = self.image_names[img_idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            image_patches = patchify(image_np, (self.patch_size, self.patch_size, 3), step=self.patch_size)
            mask_patches = patchify(mask_np, (self.patch_size, self.patch_size), step=self.patch_size)
            
            patch_row = patch_idx // self.patches_per_row
            patch_col = patch_idx % self.patches_per_row
            
            image_patch = image_patches[patch_row, patch_col, 0]
            mask_patch = mask_patches[patch_row, patch_col]
            
            image_patch = Image.fromarray(image_patch.astype('uint8'))
            mask_patch = Image.fromarray(mask_patch.astype('uint8'))
            
        else:
            img_name = self.image_names[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)
            
            image_patch = Image.open(img_path).convert('RGB')
            mask_patch = Image.open(mask_path).convert('L')
        
        # Aplicar transformações
        if self.transform:
            image_patch = self.transform(image_patch)
        else:
            image_patch = transforms.ToTensor()(image_patch)
        
        mask_patch = torch.from_numpy(np.array(mask_patch)).long()
        
        return image_patch, mask_patch


# ===================== INFERÊNCIA COM PATCHIFY =====================
class PatchifyInference:
    """
    Classe para fazer inferência em imagens grandes usando patches
    e reconstruir a imagem completa
    """
    def __init__(self, model, device, image_size=1024, patch_size=512, num_classes=1):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patches_per_row = image_size // patch_size
        
    def predict_image(self, image_path, binary_classifier=True, transform=None):
        """
        Faz predição em uma imagem completa usando patches
        
        Args:
            image_path: Caminho da imagem
            transform: Transformações para aplicar nos patches
            
        Returns:
            prediction: Máscara predita (numpy array)
            probability_map: Mapa de probabilidades (opcional)
        """
        # Carregar imagem
        image = Image.open(image_path).convert('RGB')
        
        # Redimensionar se necessário
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        image_np = np.array(image)
        
        # Dividir em patches
        image_patches = patchify(image_np, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        
        # Preparar array para predições
        pred_patches = np.zeros((
            self.patches_per_row, 
            self.patches_per_row, 
            self.patch_size, 
            self.patch_size
        ), dtype=np.uint8)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.patches_per_row):
                for j in range(self.patches_per_row):
                    # Extrair patch
                    patch = image_patches[i, j, 0]
                    patch_pil = Image.fromarray(patch.astype('uint8'))
                    
                    # Aplicar transformações
                    if transform:
                        patch_tensor = transform(patch_pil)
                    else:
                        patch_tensor = transforms.ToTensor()(patch_pil)
                    
                    # Adicionar batch dimension
                    patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
                    
                    # Predição
                    output = self.model(patch_tensor)
                    if binary_classifier:
                        probs = torch.sigmoid(output)
                        pred = (probs > 0.5).float()
                        pred = pred.squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    
                    # Armazenar predição
                    pred_patches[i, j] = pred
        
        reconstructed = unpatchify(pred_patches, (self.image_size, self.image_size))
        
        return reconstructed
    
    def predict_batch(self, image_paths, transform=None, save_dir=None):
        """
        Faz predição em múltiplas imagens
        """
        predictions = []
        
        for img_path in tqdm(image_paths, desc="Predicting images"):
            pred = self.predict_image(img_path, transform)
            predictions.append(pred)
            
            # Salvar se necessário
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                img_name = os.path.basename(img_path)
                save_path = os.path.join(save_dir, img_name)
                Image.fromarray(pred.astype('uint8')).save(save_path)
        
        return predictions