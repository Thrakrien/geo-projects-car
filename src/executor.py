import torch
from tqdm import tqdm
from src.unet_run import UnetImporter
        

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
        # loss = criterion(outputs, masks)
        loss = criterion(outputs, masks.unsqueeze(1).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        running_loss += loss.item()
        # pred = torch.argmax(outputs, dim=1)
        probs = torch.sigmoid(outputs)
            # pred = (probs > 0.5).float()
        pred = (probs > 0.5).long().squeeze(1)
        pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        mean_iou, iou_per_class = calculate_metrics(pred, masks, num_classes)
        running_iou += mean_iou
        # running_pixel_acc += pixel_acc

    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    # epoch_pixel_acc = running_pixel_acc / len(dataloader)
    
    return epoch_loss, epoch_iou #, epoch_pixel_acc

model = UnetImporter()

best_val_iou = 0.0
train_losses, val_losses = [], []
train_ious, val_ious = [], []

for epoch in range(20):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{20}")
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