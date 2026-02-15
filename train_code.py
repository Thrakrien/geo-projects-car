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
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp
import json
import csv
from datetime import datetime
from patchify import patchify, unpatchify
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.classification import multiclass_accuracy
# import wandb  # Opcional: pip install wandb

# ===================== DATASET COM PATCHIFY =====================
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
            transform: Transformações para as imagens
            use_patches (bool): Se True, divide em patches. Se False, usa imagem completa
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.use_patches = use_patches
        
        # Ler os nomes dos arquivos do txt
        with open(txt_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        # Calcular quantos patches por imagem
        self.patches_per_row = image_size // patch_size
        self.patches_per_image = self.patches_per_row ** 2
        
        # Total de patches no dataset
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
            # Calcular qual imagem e qual patch
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
            
            # Carregar imagem completa
            img_name = self.image_names[img_idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Redimensionar para tamanho esperado se necessário
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            
            # Converter para numpy
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Dividir em patches usando patchify
            # patchify retorna (n_patches_h, n_patches_w, patch_h, patch_w, channels)
            image_patches = patchify(image_np, (self.patch_size, self.patch_size, 3), step=512)
            mask_patches = patchify(mask_np, (self.patch_size, self.patch_size), step=512)
            
            # Calcular posição do patch
            patch_row = patch_idx // self.patches_per_row
            patch_col = patch_idx % self.patches_per_row
            
            # Extrair o patch específico
            image_patch = image_patches[patch_row, patch_col, 0]
            mask_patch = mask_patches[patch_row, patch_col]
            
            # Converter de volta para PIL para aplicar transforms
            image_patch = Image.fromarray(image_patch.astype('uint8'))
            mask_patch = Image.fromarray(mask_patch.astype('uint8'))
            
        else:
            # Usar imagem completa (sem patches)
            print('utilizando imagem completa')
            img_name = self.image_names[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)
            
            image_patch = Image.open(img_path).convert('RGB')
            mask_patch = Image.open(mask_path).convert('L')
        
        # Aplicar transformações
        # if self.transform:
        #     image_patch = self.transform(image_patch)
        # else:
        #     image_patch = transforms.ToTensor()(image_patch)
        image_patch = transforms.ToTensor()(image_patch)
        
        mask_patch = torch.from_numpy(np.array(mask_patch)).long()
        
        return image_patch, mask_patch


# ===================== INFERÊNCIA COM PATCHIFY =====================
class PatchifyInference:
    """
    Classe para fazer inferência em imagens grandes usando patches
    e reconstruir a imagem completa
    """
    def __init__(self, model, device, image_size=1024, patch_size=512, num_classes=2):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patches_per_row = image_size // patch_size
        
    def predict_image(self, image_path, transform=None):
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
        
        # Predizer cada patch
        self.model.eval()
        with torch.no_grad():
            for i in range(self.patches_per_row):
                for j in range(self.patches_per_row):
                    # Extrair patch
                    patch = image_patches[i, j, 0]
                    patch_pil = Image.fromarray(patch.astype('uint8'))
                    
                    # Aplicar transformações
                    # if transform:
                    #     patch_tensor = transform(patch_pil)
                    # else:
                    #     patch_tensor = transforms.ToTensor()(patch_pil)

                    patch_tensor = transforms.ToTensor()(patch_pil)
                    
                    # Adicionar batch dimension
                    patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
                    
                    # Predição
                    output = self.model(patch_tensor)
                    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                    # probs = torch.sigmoid(output)
                    # pred = (probs > 0.5).float()
                    # pred = (probs > 0.5).long().squeeze(1)
                    # pred = pred.squeeze(0).squeeze(0).cpu().numpy()
                    
                    # Armazenar predição
                    pred_patches[i, j] = pred
        
        # Reconstruir imagem completa usando unpatchify
        # unpatchify espera (n_patches_h, n_patches_w, patch_h, patch_w)
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


# ===================== LOGGER DE EXPERIMENTOS =====================
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
            # writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_pixel_acc', 
            #                 'val_loss', 'val_iou', 'val_pixel_acc', 'learning_rate'])

            writer.writerow(['epoch', 'train_loss', 'train_iou', 
                            'val_loss', 'val_iou', 'learning_rate'])
        
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
                # metrics.get('train_pixel_acc', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_iou', ''),
                # metrics.get('val_pixel_acc', ''),
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
        axes[1].set_title('Patches (4x 512x512)')
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


# ===================== MÉTRICAS =====================
def calculate_metrics(pred, target, num_classes):
    """
    Calcula IoU e Pixel Accuracy usando TorchMetrics (Otimizado).
    
    Args:
        pred: Tensor ou numpy array de predições (B, H, W) ou (B, 1, H, W)
        target: Tensor ou numpy array de targets (B, H, W)
        num_classes: Inteiro definindo número total de classes
    """
    # Converter numpy arrays para tensors se necessário
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Garantir que estão no mesmo device
    if pred.device != target.device:
        pred = pred.to(target.device)
    
    # Remove dimensão extra de canal se existir
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    
    # Garante que target também não tem dimensão extra
    if target.ndim == 4:
        target = target.squeeze(1)
    
    # Garantir tipo correto (long/int para classificação)
    pred = pred.long()
    target = target.long()
    
    # 1. Calcular IoU
    iou_per_class = mean_iou(
        preds=pred, 
        target=target, 
        num_classes=num_classes, 
        per_class=True
    )

    valid_classes_mask = iou_per_class >= 0
    if valid_classes_mask.sum() > 0:
        m_iou = iou_per_class[valid_classes_mask].mean()
    else:
        m_iou = torch.tensor(0.0, device=pred.device)

    # 2. Calcular Pixel Accuracy
    # pixel_acc = multiclass_accuracy(
    #     preds=pred, 
    #     target=target, 
    #     num_classes=num_classes, 
    #     average='micro'
    # )

    return (
        m_iou.item(),
        # pixel_acc.item(),
        iou_per_class.tolist()
    )

# ===================== TREINAMENTO =====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    """Treina o modelo por uma época"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    iou_per_class = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        # loss = criterion(outputs, masks.unsqueeze(1).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        running_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        # probs = torch.sigmoid(outputs)
            ## pred = (probs > 0.5).float()
        # pred = (probs > 0.5).long().squeeze(1)
        # pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        mean_iou, iou_per_class = calculate_metrics(pred, masks, num_classes)
        running_iou += mean_iou
        # running_pixel_acc += pixel_acc

    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    # epoch_pixel_acc = running_pixel_acc / len(dataloader)
    
    return epoch_loss, epoch_iou #, epoch_pixel_acc


def validate(model, dataloader, criterion, device, num_classes):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    # running_pixel_acc = 0.0
    iou_per_class = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            # loss = criterion(outputs, masks.unsqueeze(1).float())
            
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            # probs = torch.sigmoid(outputs)
            ## pred = (probs > 0.5).float()
            # pred = (probs > 0.5).long().squeeze(1)
            # pred = pred.squeeze(0).squeeze(0).cpu().numpy()
            mean_iou, iou_per_class = calculate_metrics(pred, masks, num_classes) 
            #pixel_acc, _ = calculate_metrics(pred, masks, num_classes)
            running_iou += mean_iou
            # running_pixel_acc += pixel_acc
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    # epoch_pixel_acc = running_pixel_acc / len(dataloader)
    
    return epoch_loss, epoch_iou #, epoch_pixel_acc


# ===================== FUNÇÃO PRINCIPAL =====================
def main():
    # ========== CONFIGURAÇÕES ==========
    config = {
        # Dados
        'train_txt': 'data-segments/train_sample.txt',
        'val_txt': 'data-segments/validation_sample.txt',
        'images_dir': '/data/integracar/amostras_car_orotofoto/',#'/data/integracar/satellite_sample_2058/', 
        'masks_dir': '/data/integracar/amostras_car_mask/',#'/data/integracar/amostras_car_mask_2058/', 
        
        # Patchify
        'use_patches': True,          # Se True, usa patches de 512x512
        'image_size': 1024,           # Tamanho original da imagem
        'patch_size': 512,            # Tamanho dos patches
        
        # Modelo
        'architecture': 'Unet',
        'encoder_name': 'resnet50',
        'encoder_weights': 'imagenet',
        'num_classes': 2,
        'activation': None,
        
        # Treinamento
        'batch_size': 8, # 5 9
        'num_epochs': 10, # 20
        'learning_rate': 0.01,       # 0.001, 0,01
        'weight_decay':  0.0005,   #1e-5,
        
        # Otimizador
        'optimizer': 'SGD',  #'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        
        # Loss
        # 'loss_function': 'DiceLoss',
        
        # Logging
        'experiment_name': 'testando-sobreposicao-das-imagens',
        'use_wandb': False,
        
        # Sistema
        'num_workers': 4,
        'seed': 42
    }
    
    # Seed para reprodutibilidade
    torch.manual_seed(config['seed']) # config['seed']
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = str(device)
    print(f"Using device: {device}")
    
    # ========== LOGGER ==========
    logger = ExperimentLogger(
        experiment_name=config['experiment_name'],
        config=config,
        use_wandb=config['use_wandb']
    )
    
    # ========== TRANSFORMAÇÕES ==========
    # As transformações agora são aplicadas nos patches de 512x512
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # testar sem o color jitter/parametros - agressivos
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # pensar em rever os pesos, visto que são imagenet based
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========== DATASETS E DATALOADERS ==========
    train_dataset = PatchifySegmentationDataset(
        txt_file=config['train_txt'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        transform=train_transform,
        use_patches=config['use_patches']
    )
    
    val_dataset = PatchifySegmentationDataset(
        txt_file=config['val_txt'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        transform=val_transform,
        use_patches=config['use_patches']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ========== MODELO (Segmentation Models PyTorch) ==========
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=3,
        classes=config['num_classes'],
        activation=config['activation']
    ).to(device)
    
    print(f"Modelo: {config['architecture']} com encoder {config['encoder_name']}")
    print(f"Entrada do modelo: patches de {config['patch_size']}x{config['patch_size']}")
    
    # ========== LOSS E OPTIMIZER ==========
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    # criterion = nn.BCEWithLogitsLoss()

    # print('Usando CrossEntropyLoss')

    # criterion = smp.losses.DiceLoss('binary', from_logits=True)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        momentum=0.9
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        # verbose=True
    )
    
    # ========== TREINAMENTO ==========
    best_val_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Treinar
        # train_loss, train_iou, train_pixel_acc = train_one_epoch(
        #     model, train_loader, criterion, optimizer, device, config['num_classes']
        # )

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config['num_classes']
        )
        # Validar
        # val_loss, val_iou, val_pixel_acc = validate(
        #     model, val_loader, criterion, device, config['num_classes']
        # )

        val_loss, val_iou = validate(
            model, val_loader, criterion, device, config['num_classes']
        )
        
        # Atualizar learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Salvar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        # Log
        metrics = {
            'train_loss': train_loss,
            'train_iou': train_iou,
            # 'train_pixel_acc': train_pixel_acc,
            'val_loss': val_loss,
            'val_iou': val_iou,
            # 'val_pixel_acc': val_pixel_acc,
            'learning_rate': current_lr
        }
        logger.log_epoch(epoch, metrics)
        
        # print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Pixel Acc: {train_pixel_acc:.4f}")
        # print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Pixel Acc: {val_pixel_acc:.4f}")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Salvar melhor modelo
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            logger.log_model(model, optimizer, epoch, metrics, 'best_model.pth')
            print(f"✓ Melhor modelo salvo com Val IoU: {val_iou:.4f}")
    
    # ========== DEMONSTRAÇÃO DE INFERÊNCIA COM PATCHIFY ==========
    print(f"\n{'='*60}")
    print("Testando inferência com patchify/unpatchify...")
    print(f"{'='*60}")
    
    # Criar objeto de inferência
    patchify_inference = PatchifyInference(
        model=model,
        device=device,
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes']
    )
    
    # Ler lista de imagens de validação
    with open(config['val_txt'], 'r') as f:
        val_images = [line.strip() for line in f.readlines()]
    
    # Fazer predição em algumas imagens de exemplo
    predictions_dir = os.path.join(logger.exp_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    for img_name in val_images[:3]:
        img_path = os.path.join(config['images_dir'], img_name)
        
        print(f"Predizindo: {img_name}")
        pred_mask = patchify_inference.predict_image(img_path, val_transform)
        
        # Salvar predição
        pred_save_path = os.path.join(predictions_dir, f"pred_{img_name}")
        Image.fromarray(pred_mask.astype('uint8')).save(pred_save_path)
        
        # Carregar imagens
        original = Image.open(img_path)
        gt_mask = np.array(Image.open(os.path.join(config['masks_dir'], img_name)))
        
        # Criar visualização com overlays usando matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        cmap = ListedColormap(['gray', 'green'])
        
        # Ground Truth Overlay
        axes[0].imshow(original)
        axes[0].imshow(gt_mask, cmap='tab20', alpha=0.9, interpolation='none')
        axes[0].set_title('Input + Ground Truth Overlay', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction Overlay
        axes[1].imshow(original)
        axes[1].imshow(pred_mask, cmap='tab20', alpha=0.9, interpolation='none')
        axes[1].set_title('Input + Prediction Overlay', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        logger.log_figure(fig, f'overlay_comparison_{img_name.split(".")[0]}')
        plt.close()

    
    # ========== PLOTAR RESULTADOS ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(train_ious, label='Train IoU')
    axes[0, 1].plot(val_ious, label='Val IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Resumo textual
    summary_text = f"""
    Experimento: {config['experiment_name']}
    
    Modelo: {config['architecture']} + {config['encoder_name']}
    Patchify: {config['use_patches']}
    Image Size: {config['image_size']}x{config['image_size']}
    Patch Size: {config['patch_size']}x{config['patch_size']}
    
    Melhor Val IoU: {best_val_iou:.4f}
    Final Train Loss: {train_losses[-1]:.4f}
    Final Val Loss: {val_losses[-1]:.4f}
    
    Épocas: {config['num_epochs']}
    Batch Size: {config['batch_size']}
    Learning Rate: {config['learning_rate']}
    """
    axes[1, 0].text(0.1, 0.5, summary_text, fontsize=10, family='monospace')
    axes[1, 0].axis('off')
    
    # Informações sobre patches
    patches_info = f"""
    PATCHIFY CONFIGURATION:
    
    Original Image: {config['image_size']}x{config['image_size']}
    Patch Size: {config['patch_size']}x{config['patch_size']}
    Patches per Image: {(config['image_size']//config['patch_size'])**2}
    
    Training Patches: {len(train_dataset)}
    Validation Patches: {len(val_dataset)}
    
    Durante treinamento:
    - Divide 1024x1024 em 4 patches 512x512
    - Treina em cada patch individualmente
    
    Durante inferência:
    - Divide imagem em patches
    - Prediz cada patch
    - Reconstrói com unpatchify
    """
    axes[1, 1].text(0.1, 0.5, patches_info, fontsize=9, family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    logger.log_figure(fig, 'training_summary')
    
    # ========== SALVAR RESUMO FINAL ==========
    summary = {
        'best_val_iou': best_val_iou,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_iou': train_ious[-1],
        'final_val_iou': val_ious[-1],
        'total_epochs': config['num_epochs'],
        'patchify_enabled': config['use_patches'],
        'patches_per_image': (config['image_size']//config['patch_size'])**2,
        'config': config
    }
    logger.save_summary(summary)
    
    logger.finish()
    print(f"\n{'='*60}")
    print("✅ Treinamento concluído!")
    print(f"📊 Resultados salvos em: {logger.exp_dir}")
    print(f"🔍 Predições de exemplo em: {predictions_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Tempo total de execução: {timedelta(seconds=elapsed)}")