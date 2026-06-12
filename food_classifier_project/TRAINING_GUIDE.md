# ConvNeXt Model Training Guide for SGFood233

This document explains how the ConvNeXt model training works in `06_hybrid_sgfood233.ipynb`.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Data Augmentation](#data-augmentation)
4. [Model Architecture](#model-architecture)
5. [Training Configuration](#training-configuration)
6. [PyTorch Lightning Module](#pytorch-lightning-module)
7. [Optimization Strategy](#optimization-strategy)
8. [Callbacks and Logging](#callbacks-and-logging)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Model Export](#model-export)

---

## Overview

The notebook trains a **ConvNeXt-Base** model (a hybrid CNN-Transformer architecture) on the **SGFood233** dataset for Singaporean food classification. The training uses:

- **Framework**: PyTorch Lightning for structured training
- **Model Library**: `timm` (PyTorch Image Models) for pretrained backbones
- **Precision**: Mixed precision (FP16) training for faster computation
- **Hardware**: NVIDIA GPU with CUDA support

---

## Dataset

### SGFood233 Dataset

| Property | Value |
|----------|-------|
| Total Classes | 233 |
| Training Samples | 153,138 |
| Validation Samples | 51,167 |
| Image Format | JPG, PNG, WEBP |

### Dataset Structure

```
data/sgfood233/
├── train/
│   ├── Alcoholic Beverage/
│   ├── Apple/
│   ├── Ban Mian/
│   └── ... (233 class folders)
└── val/
    ├── Alcoholic Beverage/
    └── ...
```

### Dataset Loading

The `LocalImageDataset` class (`utils.py:120-162`) handles data loading:

```python
class LocalImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        # Scans directory structure to build class mappings
        # class_to_idx: {'Alcoholic Beverage': 0, 'Apple': 1, ...}
        # idx_to_class: {0: 'Alcoholic Beverage', 1: 'Apple', ...}
```

Key features:
- Automatically discovers classes from folder names
- Builds bidirectional class-index mappings
- Supports JPG, JPEG, PNG, and WEBP formats
- Converts all images to RGB

---

## Data Augmentation

### Training Transforms

Training uses data augmentation to improve generalization (`utils.py:70-79`):

```python
transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to model input size
    transforms.RandomHorizontalFlip(),        # 50% chance horizontal flip
    transforms.RandomRotation(15),            # ±15 degree rotation
    transforms.ColorJitter(                   # Color perturbation
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),                    # Convert to tensor [0, 1]
    transforms.Normalize(                     # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Validation Transforms

Validation uses minimal transforms for consistent evaluation (`utils.py:82-88`):

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Why ImageNet Normalization?

The pretrained ConvNeXt model was trained on ImageNet with these specific mean/std values. Using the same normalization ensures the input distribution matches what the model expects.

---

## Model Architecture

### ConvNeXt-Base

ConvNeXt is a "modernized" CNN that incorporates design elements from Vision Transformers while maintaining a purely convolutional architecture.

| Property | Value |
|----------|-------|
| Model Name | `convnext_base` |
| Parameters | ~87.8 Million |
| Input Size | 224 × 224 × 3 |
| Pretrained On | ImageNet-1K |

### Key ConvNeXt Innovations

1. **Patchify Stem**: Uses 4×4 non-overlapping convolution (like ViT patch embedding)
2. **Inverted Bottleneck**: Wider hidden dimensions (similar to Transformer MLPs)
3. **Depthwise Convolutions**: 7×7 kernels for spatial mixing
4. **Layer Normalization**: Used instead of Batch Normalization
5. **GELU Activation**: Used instead of ReLU
6. **Fewer Activations**: Single activation per block (like Transformers)

### Model Creation

The model is created using `timm` (`utils.py:183-187`):

```python
self.model = timm.create_model(
    backbone_name='convnext_base',  # Model architecture
    pretrained=True,                 # Load ImageNet weights
    num_classes=233,                 # Replace classifier head
)
```

The `num_classes` parameter automatically replaces the final classification head (originally 1000 classes for ImageNet) with a new head for 233 classes.

---

## Training Configuration

All hyperparameters are centralized in the `Config` class (`utils.py:22-48`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMAGE_SIZE` | 224 | Input image dimensions |
| `BATCH_SIZE` | 64 | Samples per batch |
| `NUM_WORKERS` | 8 | Data loading workers |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `MAX_EPOCHS` | 10 | Maximum training epochs |
| `EARLY_STOP_PATIENCE` | 3 | Epochs without improvement before stopping |
| `SEED` | 42 | Random seed for reproducibility |

### Reproducibility

The `set_seed()` function ensures reproducible results:

```python
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
```

---

## PyTorch Lightning Module

The `FoodClassifier` class (`utils.py:169-263`) encapsulates the entire training logic.

### Forward Pass

```python
def forward(self, x):
    return self.model(x)  # Returns logits [batch_size, num_classes]
```

### Training Step

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)  # CrossEntropyLoss
    acc = (logits.argmax(dim=1) == y).float().mean()

    self.log("train_loss", loss, prog_bar=True, on_epoch=True)
    self.log("train_acc", acc, prog_bar=True, on_epoch=True)
    return loss
```

### Validation Step

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)
    preds = logits.argmax(dim=1)
    acc = (preds == y).float().mean()

    # Store predictions for macro accuracy calculation
    self.validation_step_outputs.append({'preds': preds, 'targets': y})
    return loss
```

### Loss Function

**CrossEntropyLoss** combines LogSoftmax and NLLLoss:

```
Loss = -log(exp(x[class]) / Σ exp(x[j]))
```

This is the standard loss for multi-class classification.

---

## Optimization Strategy

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    self.parameters(),
    lr=1e-4,
    weight_decay=1e-4,  # Decoupled weight decay
)
```

AdamW uses decoupled weight decay (L2 regularization applied directly to weights, not gradients), which works better with adaptive learning rates.

### Learning Rate Schedule: Cosine Annealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10,      # Number of epochs
    eta_min=1e-6,  # Minimum learning rate
)
```

The learning rate follows a cosine curve:

```
lr_t = eta_min + 0.5 * (lr_0 - eta_min) * (1 + cos(π * t / T_max))
```

| Epoch | Learning Rate |
|-------|---------------|
| 0 | 1e-4 (initial) |
| 5 | ~5e-5 (midpoint) |
| 10 | 1e-6 (minimum) |

This gradual decay helps fine-tune the model without overshooting optimal weights.

### Mixed Precision Training

```python
trainer = pl.Trainer(
    precision="16-mixed",  # FP16 for forward/backward, FP32 for optimizer
)
```

Benefits:
- ~2x faster training on modern GPUs with Tensor Cores
- Lower memory usage (allows larger batch sizes)
- Maintains accuracy through dynamic loss scaling

---

## Callbacks and Logging

### Model Checkpointing

```python
checkpoint_cb = ModelCheckpoint(
    dirpath="checkpoints/sgfood233_convnext_base",
    monitor="val_acc",      # Metric to watch
    mode="max",             # Higher is better
    save_top_k=1,           # Keep only best model
    filename="{epoch:02d}-{val_acc:.4f}",
)
```

Saves the model with highest validation accuracy. Example filename: `epoch=09-val_acc=0.8345.ckpt`

### Early Stopping

```python
early_stop_cb = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=3,  # Stop if no improvement for 3 epochs
    verbose=True,
)
```

Prevents overfitting by stopping training when validation accuracy plateaus.

### TensorBoard Logging

```python
logger = TensorBoardLogger("logs", name="sgfood233_convnext_base")
```

Logs training/validation metrics for visualization. View with:
```bash
tensorboard --logdir logs
```

---

## Evaluation Metrics

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `train_loss` | CrossEntropy loss on training set |
| `train_acc` | Top-1 accuracy on training set |
| `val_loss` | CrossEntropy loss on validation set |
| `val_acc` | Top-1 accuracy on validation set |
| `val_macro_acc` | Macro-averaged accuracy across all classes |

### Macro Accuracy Calculation

Calculated at epoch end (`utils.py:223-244`):

```python
def on_validation_epoch_end(self):
    all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
    all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

    # Per-class accuracy
    for c in range(num_classes):
        mask = all_targets == c
        class_correct[c] = ((all_preds == c) & mask).sum()
        class_total[c] = mask.sum()

    # Average across classes (not weighted by class size)
    macro_acc = (class_correct / class_total).mean()
```

Macro accuracy treats all classes equally, important when class sizes are imbalanced.

### Final Results

| Metric | Value |
|--------|-------|
| Validation Loss | 0.8451 |
| Validation Accuracy | 83.45% |
| Macro Accuracy | 82.45% |

---

## Model Export

### ONNX Export

After training, the model is exported to ONNX format for deployment:

```python
torch.onnx.export(
    best_model,
    dummy_input,                    # torch.randn(1, 3, 224, 224)
    "sg233convnext.onnx",
    export_params=True,             # Include trained weights
    opset_version=14,               # ONNX operator version
    do_constant_folding=True,       # Optimize constant operations
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={                  # Support variable batch size
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)
```

### ONNX Benefits

- **Framework agnostic**: Run in ONNX Runtime, TensorRT, OpenVINO
- **Optimized inference**: Graph optimizations and quantization
- **Edge deployment**: Mobile and embedded devices
- **Web deployment**: ONNX.js for browser inference

### Class Mapping

The label mapping is saved for inference:

```python
save_class_mapping(id2label, "id2label.json")
# {0: "Alcoholic Beverage", 1: "Apple", ...}
```

---

## Training Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. SETUP                                                    │
│  • Set random seed (42)                                      │
│  • Load pretrained ConvNeXt-Base from timm                  │
│  • Replace classifier head (1000 → 233 classes)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. DATA LOADING                                             │
│  • Load SGFood233 from local folders                        │
│  • Apply augmentations (train) / normalize (val)            │
│  • Create DataLoaders (batch=64, workers=8)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TRAINING LOOP (per epoch)                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  For each batch in train_loader:                      │  │
│  │    • Forward pass (FP16)                              │  │
│  │    • Compute CrossEntropyLoss                         │  │
│  │    • Backward pass                                    │  │
│  │    • Update weights (AdamW)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  For each batch in val_loader:                        │  │
│  │    • Forward pass (no grad)                           │  │
│  │    • Compute loss and accuracy                        │  │
│  └───────────────────────────────────────────────────────┘  │
│  • Update LR (CosineAnnealing)                              │
│  • Check for early stopping                                 │
│  • Save checkpoint if best val_acc                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. EXPORT                                                   │
│  • Load best checkpoint                                      │
│  • Export to ONNX format                                    │
│  • Save class mapping (id2label.json)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Example

### Training

```python
from utils import Config, FoodClassifier, LocalImageDataset

# Load dataset
train_ds = LocalImageDataset("data/sgfood233", split='train', transform=get_train_transforms())
val_ds = LocalImageDataset("data/sgfood233", split='val', transform=get_val_transforms())

# Create model
model = FoodClassifier(
    backbone_name='convnext_base',
    num_classes=233,
    lr=1e-4,
)

# Train
trainer = pl.Trainer(max_epochs=10, precision="16-mixed")
trainer.fit(model, train_dl, val_dl)
```

### Inference

```python
from PIL import Image
import torch

# Load model
model = FoodClassifier.load_from_checkpoint("best_model.ckpt")
model.eval()

# Preprocess image
img = Image.open("food.jpg").convert("RGB")
x = get_val_transforms()(img).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(x)
    pred_class = logits.argmax(dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

print(f"Predicted: {id2label[pred_class]} ({confidence:.1%})")
```
