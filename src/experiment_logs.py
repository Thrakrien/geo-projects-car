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


class ExperimentLogger:
    """
    Logger para registrar experimentos de mestrado com múltiplos backends
    """
    def __init__(self, experiment_name, config, log_dir='experiments', use_wandb=False):
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb
        
        # Criar timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # Criar diretório de experimento
        self.exp_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Salvar configuração
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Inicializar CSV para métricas
        self.metrics_csv = os.path.join(self.exp_dir, 'metrics.csv')
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_pixel_acc', 
                            'val_loss', 'val_iou', 'val_pixel_acc', 'learning_rate'])
        
        # Inicializar Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=experiment_name,
                name=self.run_name,
                config=config
            )
        
        print(f"📊 Experimento iniciado: {self.run_name}")
        print(f"📁 Logs salvos em: {self.exp_dir}")
    
    def log_epoch(self, epoch, metrics):
        """Registra métricas de uma época"""
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                metrics.get('train_loss', ''),
                metrics.get('train_iou', ''),
                metrics.get('train_pixel_acc', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_iou', ''),
                metrics.get('val_pixel_acc', ''),
                metrics.get('learning_rate', '')
            ])
        
        if self.use_wandb:
            wandb.log(metrics, step=epoch)
    
    def log_model(self, model, optimizer, epoch, metrics, filename='best_model.pth'):
        """Salva checkpoint do modelo"""
        checkpoint_path = os.path.join(self.exp_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)
        
        if self.use_wandb:
            wandb.save(checkpoint_path)
    
    def log_figure(self, fig, name):
        """Salva figura matplotlib"""
        fig_path = os.path.join(self.exp_dir, f"{name}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({name: wandb.Image(fig_path)})
    
    def log_patch_comparison(self, original_img, patches, reconstructed, epoch):
        """
        Visualiza processo de patchify/unpatchify
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagem original
        axes[0].imshow(original_img)
        axes[0].set_title('Original (1024x1024)')
        axes[0].axis('off')
        
        # Mostrar alguns patches
        patch_grid = np.vstack([np.hstack(patches[i:i+2]) for i in range(0, 6, 2)])
        axes[1].imshow(patch_grid)
        axes[1].set_title('Patches (6x 512x512)')
        axes[1].axis('off')
        
        # Imagem reconstruída
        axes[2].imshow(reconstructed)
        axes[2].set_title('Reconstructed (1024x1024)')
        axes[2].axis('off')
        
        plt.tight_layout()
        self.log_figure(fig, f'patchify_process_epoch_{epoch}')
        plt.close()
    
    def save_summary(self, summary_dict):
        """Salva resumo final do experimento"""
        summary_path = os.path.join(self.exp_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=4)
    
    def finish(self):
        """Finaliza o logging"""
        if self.use_wandb:
            wandb.finish()
        print(f"✅ Experimento finalizado: {self.run_name}")