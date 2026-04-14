import argparse
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import csv
from datetime import datetime
from src.dataset import create_segmentation_dataloaders
from src.inference import PatchifyInference
from src.models import build_model
from src.training_config import load_config, model_config
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.classification import multiclass_accuracy
# import wandb  # Opcional: pip install wandb


def compute_class_weights(
        train_txt,
        masks_dir,
        num_classes,
        ignore_index=5,
        dev_limit=None,
        normalize=True,
        save_csv_path=None,
        save_npy_path=None):

    with open(train_txt, "r") as f:
        names = [line.strip() for line in f.readlines()]

    if dev_limit is not None:
        names = names[:dev_limit]

    counts = np.zeros(num_classes, dtype=np.int64)

    for name in names:
        mask_path = os.path.join(masks_dir, name)
        mask = np.array(Image.open(mask_path))  # se já é máscara indexada, não precisa convert('L')
        mask = mask.astype(np.int64)

        if ignore_index is not None:
            mask = mask[mask != ignore_index]

        if mask.size == 0:
            continue

        binc = np.bincount(mask.ravel(), minlength=num_classes)
        counts += binc[:num_classes]

    M = counts.sum()
    C = num_classes

    denominador = C * counts
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = M / denominador.astype(np.float64)
        weights[np.isinf(weights)] = 0.0
        weights = np.nan_to_num(weights)

    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()

    if save_csv_path:
        df = pd.DataFrame({"count_pixels": counts, "weight": weights})
        df.to_csv(save_csv_path, sep=";", decimal=",", encoding="utf-8", index=False)

    if save_npy_path:
        np.save(save_npy_path, weights)

    return weights 

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML or JSON experiment config path."
    )
    return parser.parse_args()


# ===================== FUNÇÃO PRINCIPAL =====================
def main() -> None:
    # ========== CONFIGURAÇÕES ==========
    args = parse_args()
    config = load_config(args.config)
    
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
        # CORRECAO: evitar flips nao pareados (imagem x mascara).
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # pensar em rever os pesos, visto que são imagenet based
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========== DATASETS E DATALOADERS ==========
    train_dataset, val_dataset, train_loader, val_loader = (
        create_segmentation_dataloaders(
            config=config,
            train_transform=train_transform,
            val_transform=val_transform
        )
    )
    
    # ========== MODELO ==========
    model = build_model(model_config(config)).to(device)
    
    print(f"Modelo: {config['architecture']} com encoder {config['encoder_name']}")
    print(f"Entrada do modelo: patches de {config['patch_size']}x{config['patch_size']}")
    
    # ========== LOSS E OPTIMIZER ==========
    # criterion = nn.CrossEntropyLoss(ignore_index=2) # para binario 2 e para full 14

    class_weights = None
    if config.get("use_class_weights", True):
        w = compute_class_weights(
            train_txt=config["train_txt"],
            masks_dir=config["masks_dir"],
            num_classes=config["num_classes"],
            ignore_index=config["ignore_index"], # para binario 2 e para full 14
            dev_limit=config.get("weights_dev_limit", None),
            normalize=True,
            # save_csv_path=os.path.join(logger.exp_dir, "loss_weights.csv"),
            save_npy_path=os.path.join(logger.exp_dir, "loss_weights.npy"),
        )
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        # CORRECAO: evita expressao constante ambigua e warning de sintaxe.
        weight=class_weights, ignore_index=config["ignore_index"])
    # criterion = nn.BCEWithLogitsLoss()

    # print('Usando CrossEntropyLoss')

    # criterion = smp.losses.DiceLoss('binary', from_logits=True)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        momentum=0.9
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        gamma=config["gamma"],
        milestones=config["milestones"]
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
    print("Testando inferência com sliding window e overlap...")
    print(f"{'='*60}")
    
    # Criar objeto de inferência
    patchify_inference = PatchifyInference(
        model=model,
        device=device,
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        step=config['patch_step']
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
        pred_mask = patchify_inference.predict_image(
            img_path,
            transform=val_transform,
            window_size=config['patch_size'],
            stride=config['inference_stride']
        )
        
        # Salvar predição
        pred_save_path = os.path.join(predictions_dir, f"pred_{img_name}")
        Image.fromarray(pred_mask.astype('uint8')).save(pred_save_path)
        
        # Carregar imagens
        original = Image.open(img_path).convert('RGB')
        gt_mask_pil = Image.open(os.path.join(config['masks_dir'], img_name))
        # CORRECAO: alinhar tamanhos para evitar overlay visualmente "bonito" mas deslocado.
        if original.size != (config['image_size'], config['image_size']):
            original = original.resize((config['image_size'], config['image_size']), Image.BILINEAR)
        if gt_mask_pil.size != (config['image_size'], config['image_size']):
            gt_mask_pil = gt_mask_pil.resize((config['image_size'], config['image_size']), Image.NEAREST)
        gt_mask = np.array(gt_mask_pil)
        
        # Criar visualização com overlays usando matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # cmap = ListedColormap(['gray', 'green'])


        cmap = ListedColormap([
            "#DC143C",  # 0 Infraestrutura
            "#FFD700",  # 1 Agropastoril
            "#006400",  # 2 Vegetação Nativa
            "#8F9779",  # 3 Macega
            "#1E90FF"  # 4 Massa D'água
        ])
        
        # Ground Truth Overlay
        axes[0].imshow(original)
        axes[0].imshow(gt_mask, cmap=cmap, alpha=0.6, interpolation='none')
        axes[0].set_title('Input + Ground Truth Overlay', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction Overlay
        axes[1].imshow(original)
        axes[1].imshow(pred_mask, cmap=cmap, alpha=0.6, interpolation='none')
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
    Train Patch Step: {config['patch_step']}
    Inference Stride: {config['inference_stride']}
    Patches per Image: {(((config['image_size'] - config['patch_size']) // config['patch_step']) + 1) ** 2}
    
    Training Patches: {len(train_dataset)}
    Validation Patches: {len(val_dataset)}
    
    Durante treinamento:
    - Treina em cada patch individualmente
    
    Durante inferência:
    - Usa sliding window com overlap
    - Prediz cada janela
    - Agrega logits por média antes do argmax
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
        'patches_per_image': (((config['image_size'] - config['patch_size']) // config['patch_step']) + 1) ** 2,
        'inference_stride': config['inference_stride'],
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
