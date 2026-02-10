import torch
import numpy as np
from torchmetrics.functional.segmentation import mean_iou


def calculate_metrics(pred, target, num_classes):
    """
    Calcula IoU e Pixel Accuracy usando TorchMetrics built-in.
    
    Args:
        pred: Tensor ou numpy array de predições (B, H, W) ou (B, 1, H, W)
        target: Tensor ou numpy array de targets (B, H, W)
        num_classes: Valor inteiro definindo número total de classes
    """
 
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    if pred.device != target.device:
        pred = pred.to(target.device)
    
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    
    if target.ndim == 4:
        target = target.squeeze(1)
    
    pred = pred.long()
    target = target.long()
    
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
        m_iou = torch.tensor(0.0, device='cuda')

    # Accuracy comentada devido estarmos olhando apenas o iou e mean iou

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