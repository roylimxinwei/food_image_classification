"""
Shared utilities for food classification experiments.
CNN vs ViT vs Hybrid comparison on Food-101 and SGFood233.
"""

import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, Any
import json
import os


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Shared configuration for fair model comparison."""
    
    # Image settings
    IMAGE_SIZE = 224
    
    # ImageNet normalization (used by pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Training settings (same for all models)
    BATCH_SIZE = 64
    NUM_WORKERS = 8  # Set to 0 on Windows if issues
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 10
    EARLY_STOP_PATIENCE = 3
    
    # Random seed for reproducibility
    SEED = 42
    
    # Model architectures
    MODELS = {
        'cnn': 'resnet50',
        'vit': 'vit_base_patch16_224',
        'hybrid': 'convnext_base',
    }

    HYBRID_MODELS = {
        'coatnet': 'coatnet_2_rw_224',      # Conv + Attention interleaved
        'maxvit': 'maxvit_base_tf_224',      # Multi-axis attention + MBConv
        'levit': 'levit_384',                # Conv stem + attention stages  
        'efficientformer': 'efficientformer_l7',
    }


def set_seed(seed: int = Config.SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


# ============================================================================
# Data Transforms
# ============================================================================

def get_train_transforms(image_size: int = Config.IMAGE_SIZE):
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = Config.IMAGE_SIZE):
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])


# ============================================================================
# Dataset Wrappers
# ============================================================================

class HFImageDataset(Dataset):
    """Wrapper for Hugging Face image datasets."""
    
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        label = item["label"]
        
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class LocalImageDataset(Dataset):
    """Dataset for local image folders (e.g., SGFood233)."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Path to dataset root
            split: 'train', 'val', or 'test'
            transform: Image transforms
        """
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        split_dir = os.path.join(root_dir, split)
        
        # Build class mapping
        classes = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all image paths
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((
                        os.path.join(cls_dir, img_name),
                        self.class_to_idx[cls]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============================================================================
# Lightning Module
# ============================================================================

class FoodClassifier(pl.LightningModule):
    """PyTorch Lightning module for food classification."""
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        lr: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model with pretrained weights
        self.model = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=num_classes,
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # For tracking predictions
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'preds': preds,
            'targets': y,
        })
        return loss

    def on_validation_epoch_end(self):
        # Calculate macro F1 score
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Per-class accuracy (for macro F1 approximation)
        num_classes = self.hparams.num_classes
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        for c in range(num_classes):
            mask = all_targets == c
            if mask.sum() > 0:
                class_correct[c] = ((all_preds == c) & mask).sum().float()
                class_total[c] = mask.sum().float()
        
        # Macro accuracy (approximation of macro F1 for balanced comparison)
        valid_classes = class_total > 0
        macro_acc = (class_correct[valid_classes] / class_total[valid_classes]).mean()
        self.log("val_macro_acc", macro_acc, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


# ============================================================================
# Training Utilities
# ============================================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """Create train and validation dataloaders."""
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_dl, val_dl


def get_trainer_callbacks():
    """Get standard callbacks for training."""
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="{epoch:02d}-{val_acc:.4f}",
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=Config.EARLY_STOP_PATIENCE,
        verbose=True,
    )
    
    return [checkpoint_cb, early_stop_cb]


def save_class_mapping(id2label: Dict[int, str], filepath: str):
    """Save class label mapping to JSON."""
    with open(filepath, 'w') as f:
        json.dump(id2label, f, indent=2)


def load_class_mapping(filepath: str) -> Dict[int, str]:
    """Load class label mapping from JSON."""
    with open(filepath, 'r') as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


# ============================================================================
# Feature Extraction for Embedding Analysis
# ============================================================================

class FeatureExtractor:
    """
    Extract penultimate layer features from timm models for embedding analysis.

    Works with ResNet50, ViT, ConvNeXt, and other timm architectures.
    Uses forward hooks to capture activations before the classifier head.

    Example:
        >>> model = timm.create_model('resnet50', pretrained=True, num_classes=233)
        >>> extractor = FeatureExtractor(model, 'resnet50')
        >>> features = extractor.extract(image_tensor)  # shape: (batch, feature_dim)
    """

    def __init__(self, model: nn.Module, model_name: str):
        """
        Args:
            model: A timm model (can be wrapped in FoodClassifier or standalone)
            model_name: The timm model name (e.g., 'resnet50', 'vit_base_patch16_224')
        """
        self.model_name = model_name
        self.features = None
        self.hook = None

        # Handle FoodClassifier wrapper
        if hasattr(model, 'model'):
            self.model = model.model
        else:
            self.model = model

        # Register hook on the appropriate layer
        self._register_hook()

    def _get_target_layer(self) -> nn.Module:
        """Get the penultimate layer based on model architecture."""
        model = self.model

        # ResNet family
        if 'resnet' in self.model_name.lower():
            return model.global_pool

        # Vision Transformer family
        elif 'vit' in self.model_name.lower() or 'deit' in self.model_name.lower():
            # ViT uses fc_norm or norm before the head
            if hasattr(model, 'fc_norm'):
                return model.fc_norm
            elif hasattr(model, 'norm'):
                return model.norm

        # ConvNeXt family
        elif 'convnext' in self.model_name.lower():
            # ConvNeXt: stages[-1] -> norm_pre -> head
            if hasattr(model, 'head') and hasattr(model.head, 'norm'):
                return model.head.norm
            elif hasattr(model, 'norm_pre'):
                return model.norm_pre

        # CoAtNet, MaxViT
        elif 'coatnet' in self.model_name.lower() or 'maxvit' in self.model_name.lower():
            if hasattr(model, 'head') and hasattr(model.head, 'global_pool'):
                return model.head.global_pool

        # EfficientNet family
        elif 'efficientnet' in self.model_name.lower():
            return model.global_pool

        # Fallback: try common patterns
        if hasattr(model, 'global_pool'):
            return model.global_pool
        elif hasattr(model, 'head') and hasattr(model.head, 'global_pool'):
            return model.head.global_pool

        raise ValueError(f"Cannot find penultimate layer for model: {self.model_name}")

    def _register_hook(self):
        """Register forward hook to capture features."""
        target_layer = self._get_target_layer()

        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                # Flatten spatial dimensions if present
                if output.dim() == 4:  # (B, C, H, W)
                    output = output.mean(dim=[2, 3])
                elif output.dim() == 3:  # (B, N, C) for ViT
                    output = output[:, 0]  # CLS token
                self.features = output.detach()
            elif isinstance(output, tuple):
                out = output[0]
                if out.dim() == 4:
                    out = out.mean(dim=[2, 3])
                elif out.dim() == 3:
                    out = out[:, 0]
                self.features = out.detach()

        self.hook = target_layer.register_forward_hook(hook_fn)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __del__(self):
        self.remove_hook()


def extract_all_features(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
) -> tuple:
    """
    Extract features from all images in a dataloader.

    Args:
        model: The model to extract features from
        model_name: Model architecture name
        dataloader: DataLoader with images
        device: Device to run on
        max_samples: Maximum number of samples to extract (for memory efficiency)

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    import numpy as np

    extractor = FeatureExtractor(model, model_name)
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []
    n_samples = 0

    for images, labels in dataloader:
        if max_samples and n_samples >= max_samples:
            break

        images = images.to(device)
        features = extractor.extract(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

        n_samples += images.size(0)

    extractor.remove_hook()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]

    return features, labels
