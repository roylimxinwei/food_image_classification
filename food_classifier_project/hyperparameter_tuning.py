"""
Hyperparameter Tuning Experiments for ConvNeXt on SGFood233

This module implements systematic hyperparameter tuning following the sequential
dependency approach: each phase uses the best values from previous phases.

Hyperparameters studied:
1. Learning Rate: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
2. Weight Decay: [0, 1e-5, 1e-4, 1e-3, 1e-2]
3. Batch Size: [16, 32, 64, 128]
4. LR Schedule: [constant, step, cosine, cosine_warmup]
5. Data Augmentation: [none, basic, moderate, strong, autoaugment]
6. Label Smoothing: [0, 0.05, 0.1, 0.15, 0.2]
7. Optimizer: [SGD, Adam, AdamW]

Usage:
    from hyperparameter_tuning import HyperparameterExperiment

    experiment = HyperparameterExperiment(
        dataset_root='path/to/sgfood233',
        results_dir='hp_results'
    )

    # Run all experiments sequentially
    results = experiment.run_all_experiments()

    # Or run individual phases
    lr_results = experiment.run_learning_rate_sweep()
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import timm
from torchvision import transforms
from torchvision.transforms import autoaugment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    Config, set_seed, LocalImageDataset,
    get_train_transforms, get_val_transforms,
    create_dataloaders
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""

    # Model
    model_name: str = 'convnext_base'
    num_classes: int = 233

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 15

    # Optimizer
    optimizer: str = 'AdamW'  # SGD, Adam, AdamW

    # Scheduler
    scheduler: str = 'cosine'  # constant, step, cosine, cosine_warmup, onecycle
    warmup_epochs: int = 1

    # Regularization
    label_smoothing: float = 0.0

    # Augmentation
    augmentation: str = 'basic'  # none, basic, moderate, strong, autoaugment

    # Misc
    seed: int = 42
    num_workers: int = 4


@dataclass
class ExperimentResult:
    """Results from a single training experiment."""

    config: Dict[str, Any]
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    final_train_acc: float
    final_train_loss: float
    training_time_minutes: float
    history: Dict[str, List[float]] = field(default_factory=dict)


# ============================================================================
# Data Augmentation Strategies
# ============================================================================

def get_augmentation_transforms(aug_type: str, image_size: int = 224) -> transforms.Compose:
    """
    Get augmentation transforms based on strategy type.

    Args:
        aug_type: One of 'none', 'basic', 'moderate', 'strong', 'autoaugment'
        image_size: Target image size

    Returns:
        Composed transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if aug_type == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    elif aug_type == 'basic':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])

    elif aug_type == 'moderate':
        # RandAugment with moderate settings
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            autoaugment.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])

    elif aug_type == 'strong':
        # Strong augmentation with RandAugment + random erasing
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            autoaugment.RandAugment(num_ops=2, magnitude=15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])

    elif aug_type == 'autoaugment':
        # ImageNet AutoAugment policy
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


# ============================================================================
# Lightning Module with Configurable Options
# ============================================================================

class TunableFoodClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for food classification with configurable hyperparameters.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create model
        self.model = timm.create_model(
            config.model_name,
            pretrained=True,
            num_classes=config.num_classes,
        )

        # Loss function with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Tracking
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        self.training_step_outputs.append({'loss': loss.detach(), 'acc': acc.detach()})
        return loss

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=1).indices
        top5_correct = (top5_preds == y.unsqueeze(1)).any(dim=1).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_top5_acc', top5_correct, prog_bar=False, on_step=False, on_epoch=True)

        self.validation_step_outputs.append({
            'preds': preds,
            'targets': y,
            'loss': loss.detach(),
        })
        return loss

    def on_validation_epoch_end(self):
        # Calculate macro F1 approximation
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

        num_classes = self.config.num_classes
        class_correct = torch.zeros(num_classes, device=self.device)
        class_total = torch.zeros(num_classes, device=self.device)

        for c in range(num_classes):
            mask = all_targets == c
            if mask.sum() > 0:
                class_correct[c] = ((all_preds == c) & mask).sum().float()
                class_total[c] = mask.sum().float()

        valid_classes = class_total > 0
        macro_acc = (class_correct[valid_classes] / class_total[valid_classes]).mean()
        self.log('val_macro_acc', macro_acc, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Select optimizer
        if self.config.optimizer == 'SGD':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'Adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Select scheduler
        if self.config.scheduler == 'constant':
            return optimizer

        elif self.config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.1,
            )

        elif self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=1e-6,
            )

        elif self.config.scheduler == 'cosine_warmup':
            # Cosine with linear warmup
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return epoch / self.config.warmup_epochs
                else:
                    progress = (epoch - self.config.warmup_epochs) / (
                        self.config.max_epochs - self.config.warmup_epochs
                    )
                    return 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        elif self.config.scheduler == 'onecycle':
            # Note: OneCycleLR requires steps_per_epoch, will be set in trainer
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.max_epochs,
                steps_per_epoch=1,  # Will be overridden
                pct_start=0.1,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }

        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# ============================================================================
# Experiment Runner
# ============================================================================

class HyperparameterExperiment:
    """
    Manages hyperparameter tuning experiments for ConvNeXt on SGFood233.

    Follows sequential dependency approach: each phase uses best values
    from previous phases.
    """

    def __init__(
        self,
        dataset_root: str,
        results_dir: str = 'hp_results',
        device: str = 'auto',
    ):
        """
        Args:
            dataset_root: Path to SGFood233 dataset
            results_dir: Directory to save results
            device: 'cuda', 'cpu', or 'auto'
        """
        self.dataset_root = dataset_root
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Best hyperparameters (updated as experiments progress)
        self.best_config = ExperimentConfig()

        # All results
        self.all_results: Dict[str, List[ExperimentResult]] = {}

    def _create_dataloaders(
        self,
        batch_size: int,
        augmentation: str,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders with specified augmentation."""

        train_transform = get_augmentation_transforms(augmentation)
        val_transform = get_val_transforms()

        train_dataset = LocalImageDataset(
            self.dataset_root,
            split='train',
            transform=train_transform,
        )

        val_dataset = LocalImageDataset(
            self.dataset_root,
            split='val',
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        return train_loader, val_loader

    def _run_single_experiment(
        self,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str,
    ) -> ExperimentResult:
        """Run a single training experiment."""

        set_seed(config.seed)

        # Create model
        model = TunableFoodClassifier(config)

        # Callbacks
        checkpoint_cb = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename=f'{experiment_name}-{{epoch:02d}}-{{val_acc:.4f}}',
            dirpath=os.path.join(self.results_dir, 'checkpoints'),
        )

        early_stop_cb = EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=5,
            verbose=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Logger
        logger = CSVLogger(
            save_dir=self.results_dir,
            name='logs',
            version=experiment_name,
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )

        # Train
        start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        training_time = (time.time() - start_time) / 60  # minutes

        # Collect results
        best_val_acc = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else 0.0

        # Load training history from CSV logs
        metrics_file = os.path.join(
            self.results_dir, 'logs', experiment_name, 'metrics.csv'
        )
        history = {}
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            for col in df.columns:
                if col != 'step' and col != 'epoch':
                    history[col] = df[col].dropna().tolist()

        result = ExperimentResult(
            config=asdict(config),
            best_val_acc=best_val_acc,
            best_val_loss=trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item(),
            best_epoch=checkpoint_cb.best_model_path.split('-')[1] if checkpoint_cb.best_model_path else 0,
            final_train_acc=trainer.callback_metrics.get('train_acc', torch.tensor(0.0)).item(),
            final_train_loss=trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item(),
            training_time_minutes=training_time,
            history=history,
        )

        return result

    def run_learning_rate_sweep(
        self,
        learning_rates: List[float] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 1: Sweep learning rate values.

        Args:
            learning_rates: List of LR values to test

        Returns:
            List of experiment results
        """
        if learning_rates is None:
            learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

        print("\n" + "="*60)
        print("Phase 1: Learning Rate Sweep")
        print("="*60)

        results = []
        train_loader, val_loader = self._create_dataloaders(
            batch_size=self.best_config.batch_size,
            augmentation=self.best_config.augmentation,
        )

        for lr in learning_rates:
            config = ExperimentConfig(
                learning_rate=lr,
                weight_decay=self.best_config.weight_decay,
                batch_size=self.best_config.batch_size,
                scheduler=self.best_config.scheduler,
                augmentation=self.best_config.augmentation,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'lr_{lr:.0e}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        # Update best config
        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.learning_rate = best_result.config['learning_rate']
        print(f"\nBest LR: {self.best_config.learning_rate} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['learning_rate'] = results
        self._save_phase_results('learning_rate', results)

        return results

    def run_weight_decay_sweep(
        self,
        weight_decays: List[float] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 2: Sweep weight decay values (using best LR from Phase 1).
        """
        if weight_decays is None:
            weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]

        print("\n" + "="*60)
        print("Phase 2: Weight Decay Sweep")
        print(f"Using best LR: {self.best_config.learning_rate}")
        print("="*60)

        results = []
        train_loader, val_loader = self._create_dataloaders(
            batch_size=self.best_config.batch_size,
            augmentation=self.best_config.augmentation,
        )

        for wd in weight_decays:
            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=wd,
                batch_size=self.best_config.batch_size,
                scheduler=self.best_config.scheduler,
                augmentation=self.best_config.augmentation,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'wd_{wd:.0e}' if wd > 0 else 'wd_0'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.weight_decay = best_result.config['weight_decay']
        print(f"\nBest WD: {self.best_config.weight_decay} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['weight_decay'] = results
        self._save_phase_results('weight_decay', results)

        return results

    def run_batch_size_sweep(
        self,
        batch_sizes: List[int] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 3: Sweep batch size values.
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128]

        print("\n" + "="*60)
        print("Phase 3: Batch Size Sweep")
        print(f"Using best LR: {self.best_config.learning_rate}, WD: {self.best_config.weight_decay}")
        print("="*60)

        results = []

        for bs in batch_sizes:
            train_loader, val_loader = self._create_dataloaders(
                batch_size=bs,
                augmentation=self.best_config.augmentation,
            )

            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=self.best_config.weight_decay,
                batch_size=bs,
                scheduler=self.best_config.scheduler,
                augmentation=self.best_config.augmentation,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'bs_{bs}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.batch_size = best_result.config['batch_size']
        print(f"\nBest BS: {self.best_config.batch_size} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['batch_size'] = results
        self._save_phase_results('batch_size', results)

        return results

    def run_scheduler_sweep(
        self,
        schedulers: List[str] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 4: Sweep learning rate schedulers.
        """
        if schedulers is None:
            schedulers = ['constant', 'step', 'cosine', 'cosine_warmup']

        print("\n" + "="*60)
        print("Phase 4: LR Scheduler Sweep")
        print(f"Using best config so far...")
        print("="*60)

        results = []
        train_loader, val_loader = self._create_dataloaders(
            batch_size=self.best_config.batch_size,
            augmentation=self.best_config.augmentation,
        )

        for sched in schedulers:
            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=self.best_config.weight_decay,
                batch_size=self.best_config.batch_size,
                scheduler=sched,
                augmentation=self.best_config.augmentation,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'sched_{sched}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.scheduler = best_result.config['scheduler']
        print(f"\nBest Scheduler: {self.best_config.scheduler} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['scheduler'] = results
        self._save_phase_results('scheduler', results)

        return results

    def run_augmentation_sweep(
        self,
        augmentations: List[str] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 5: Sweep data augmentation strategies.
        """
        if augmentations is None:
            augmentations = ['none', 'basic', 'moderate', 'strong', 'autoaugment']

        print("\n" + "="*60)
        print("Phase 5: Data Augmentation Sweep")
        print("="*60)

        results = []

        for aug in augmentations:
            train_loader, val_loader = self._create_dataloaders(
                batch_size=self.best_config.batch_size,
                augmentation=aug,
            )

            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=self.best_config.weight_decay,
                batch_size=self.best_config.batch_size,
                scheduler=self.best_config.scheduler,
                augmentation=aug,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'aug_{aug}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.augmentation = best_result.config['augmentation']
        print(f"\nBest Augmentation: {self.best_config.augmentation} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['augmentation'] = results
        self._save_phase_results('augmentation', results)

        return results

    def run_label_smoothing_sweep(
        self,
        label_smoothings: List[float] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 6: Sweep label smoothing values.
        """
        if label_smoothings is None:
            label_smoothings = [0, 0.05, 0.1, 0.15, 0.2]

        print("\n" + "="*60)
        print("Phase 6: Label Smoothing Sweep")
        print("="*60)

        results = []
        train_loader, val_loader = self._create_dataloaders(
            batch_size=self.best_config.batch_size,
            augmentation=self.best_config.augmentation,
        )

        for ls in label_smoothings:
            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=self.best_config.weight_decay,
                batch_size=self.best_config.batch_size,
                scheduler=self.best_config.scheduler,
                augmentation=self.best_config.augmentation,
                label_smoothing=ls,
                optimizer=self.best_config.optimizer,
            )

            experiment_name = f'ls_{ls}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.label_smoothing = best_result.config['label_smoothing']
        print(f"\nBest Label Smoothing: {self.best_config.label_smoothing} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['label_smoothing'] = results
        self._save_phase_results('label_smoothing', results)

        return results

    def run_optimizer_sweep(
        self,
        optimizers: List[str] = None,
    ) -> List[ExperimentResult]:
        """
        Phase 7: Sweep optimizer choices (final validation).
        """
        if optimizers is None:
            optimizers = ['SGD', 'Adam', 'AdamW']

        print("\n" + "="*60)
        print("Phase 7: Optimizer Sweep (Final Validation)")
        print("="*60)

        results = []
        train_loader, val_loader = self._create_dataloaders(
            batch_size=self.best_config.batch_size,
            augmentation=self.best_config.augmentation,
        )

        for opt in optimizers:
            config = ExperimentConfig(
                learning_rate=self.best_config.learning_rate,
                weight_decay=self.best_config.weight_decay,
                batch_size=self.best_config.batch_size,
                scheduler=self.best_config.scheduler,
                augmentation=self.best_config.augmentation,
                label_smoothing=self.best_config.label_smoothing,
                optimizer=opt,
            )

            experiment_name = f'opt_{opt.lower()}'
            print(f"\nRunning experiment: {experiment_name}")

            result = self._run_single_experiment(
                config, train_loader, val_loader, experiment_name
            )
            results.append(result)

            print(f"  Best Val Acc: {result.best_val_acc:.4f}")

        best_result = max(results, key=lambda r: r.best_val_acc)
        self.best_config.optimizer = best_result.config['optimizer']
        print(f"\nBest Optimizer: {self.best_config.optimizer} (acc: {best_result.best_val_acc:.4f})")

        self.all_results['optimizer'] = results
        self._save_phase_results('optimizer', results)

        return results

    def run_all_experiments(self) -> Dict[str, List[ExperimentResult]]:
        """
        Run all hyperparameter tuning phases sequentially.

        Returns:
            Dictionary mapping phase names to results
        """
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING FOR CONVNEXT ON SGFOOD233")
        print("="*70)
        print(f"Dataset: {self.dataset_root}")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.device}")

        # Run all phases
        self.run_learning_rate_sweep()
        self.run_weight_decay_sweep()
        self.run_batch_size_sweep()
        self.run_scheduler_sweep()
        self.run_augmentation_sweep()
        self.run_label_smoothing_sweep()
        self.run_optimizer_sweep()

        # Save final results
        self._save_final_results()

        return self.all_results

    def _save_phase_results(self, phase_name: str, results: List[ExperimentResult]):
        """Save results from a single phase to JSON."""
        filepath = os.path.join(self.results_dir, f'{phase_name}_results.json')

        # Convert to serializable format
        data = []
        for r in results:
            d = {
                'config': r.config,
                'best_val_acc': r.best_val_acc,
                'best_val_loss': r.best_val_loss,
                'best_epoch': r.best_epoch,
                'final_train_acc': r.final_train_acc,
                'final_train_loss': r.final_train_loss,
                'training_time_minutes': r.training_time_minutes,
            }
            data.append(d)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {phase_name} results to {filepath}")

    def _save_final_results(self):
        """Save final summary and best configuration."""

        # Best configuration
        best_config_path = os.path.join(self.results_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(asdict(self.best_config), f, indent=2)

        # Summary table
        summary = []
        for phase_name, results in self.all_results.items():
            best = max(results, key=lambda r: r.best_val_acc)
            summary.append({
                'phase': phase_name,
                'best_value': self._get_phase_value(phase_name, best.config),
                'best_val_acc': best.best_val_acc,
                'training_time_min': best.training_time_minutes,
            })

        summary_path = os.path.join(self.results_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print("="*60)
        print(f"Best configuration saved to: {best_config_path}")
        print(f"\nBest hyperparameters:")
        for key, value in asdict(self.best_config).items():
            print(f"  {key}: {value}")

    def _get_phase_value(self, phase_name: str, config: Dict) -> Any:
        """Get the varied hyperparameter value for a phase."""
        mapping = {
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'batch_size': 'batch_size',
            'scheduler': 'scheduler',
            'augmentation': 'augmentation',
            'label_smoothing': 'label_smoothing',
            'optimizer': 'optimizer',
        }
        return config.get(mapping.get(phase_name, phase_name))


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_learning_rate_results(results: List[ExperimentResult], save_path: str = None):
    """Plot learning rate sweep results."""
    lrs = [r.config['learning_rate'] for r in results]
    accs = [r.best_val_acc for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax1 = axes[0]
    x_labels = [f'{lr:.0e}' for lr in lrs]
    bars = ax1.bar(x_labels, accs, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Learning Rate vs Validation Accuracy', fontsize=14)
    ax1.set_ylim(min(accs) - 0.02, max(accs) + 0.02)

    # Highlight best
    best_idx = accs.index(max(accs))
    bars[best_idx].set_color('darkgreen')

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # Training curves (if history available)
    ax2 = axes[1]
    for r in results:
        if 'val_acc' in r.history:
            label = f"LR={r.config['learning_rate']:.0e}"
            ax2.plot(r.history['val_acc'], label=label)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Training Curves by Learning Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_ablation_study(results_dir: str, save_path: str = None):
    """
    Plot ablation study showing incremental gains from each hyperparameter.
    """
    summary_path = os.path.join(results_dir, 'summary.json')

    if not os.path.exists(summary_path):
        print("Summary file not found. Run experiments first.")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    phases = [s['phase'] for s in summary]
    accs = [s['best_val_acc'] for s in summary]

    # Calculate improvements
    baseline = 0.8345  # Your baseline accuracy
    improvements = []
    cumulative = baseline

    for acc in accs:
        improvements.append(acc - cumulative)
        cumulative = acc

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(phases))
    bars = ax.bar(x, accs, color='steelblue', edgecolor='black')

    ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline: {baseline:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, ha='right')
    ax.set_xlabel('Optimization Phase', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Incremental Improvement from Hyperparameter Optimization', fontsize=14)
    ax.legend()

    # Add improvement annotations
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        if imp > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'+{imp:.3f}', ha='center', va='bottom', fontsize=9, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def create_results_table(results_dir: str) -> pd.DataFrame:
    """
    Create a summary table of all hyperparameter tuning results.
    """
    all_data = []

    for phase in ['learning_rate', 'weight_decay', 'batch_size', 'scheduler',
                  'augmentation', 'label_smoothing', 'optimizer']:
        filepath = os.path.join(results_dir, f'{phase}_results.json')

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results = json.load(f)

            for r in results:
                all_data.append({
                    'Phase': phase,
                    'Value': r['config'].get(phase, 'N/A'),
                    'Val Accuracy': r['best_val_acc'],
                    'Train Accuracy': r['final_train_acc'],
                    'Best Epoch': r['best_epoch'],
                    'Time (min)': r['training_time_minutes'],
                })

    df = pd.DataFrame(all_data)
    return df


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for ConvNeXt on SGFood233')
    parser.add_argument('--dataset', type=str, required=True, help='Path to SGFood233 dataset')
    parser.add_argument('--results-dir', type=str, default='hp_results', help='Results directory')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'lr', 'wd', 'bs', 'sched', 'aug', 'ls', 'opt'],
                       help='Which phase to run')

    args = parser.parse_args()

    experiment = HyperparameterExperiment(
        dataset_root=args.dataset,
        results_dir=args.results_dir,
    )

    if args.phase == 'all':
        experiment.run_all_experiments()
    elif args.phase == 'lr':
        experiment.run_learning_rate_sweep()
    elif args.phase == 'wd':
        experiment.run_weight_decay_sweep()
    elif args.phase == 'bs':
        experiment.run_batch_size_sweep()
    elif args.phase == 'sched':
        experiment.run_scheduler_sweep()
    elif args.phase == 'aug':
        experiment.run_augmentation_sweep()
    elif args.phase == 'ls':
        experiment.run_label_smoothing_sweep()
    elif args.phase == 'opt':
        experiment.run_optimizer_sweep()
