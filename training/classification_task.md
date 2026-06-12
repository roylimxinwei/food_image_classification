# End-to-End Training Steps (CNN vs ViT vs Hybrid)

## 0. Experiment Design
- Define task: single-label food image classification
- Datasets: Food-101, SGFood233
- Models: CNN, ViT, Hybrid CNN–ViT
- Fix comparison constraints:
  - Input resolution (e.g., 224×224)
  - Training budget (epochs or steps)
  - Optimizer family (e.g., AdamW)
  - Augmentation policy
  - Evaluation metrics (Top-1, Macro-F1)
  - Random seeds

---

## 1. Dataset Preparation
### 1.1 Data Splits
- Food-101: use official train/test split
- SGFood233: stratified train/val/test (e.g., 70/15/15), fixed seed

### 1.2 Preprocessing
- Resize and crop images
- Normalize (ImageNet mean/std)
- Verify label consistency

### 1.3 Data Augmentation (shared)
- RandomResizedCrop
- HorizontalFlip
- ColorJitter
- Optional: Mixup/CutMix (same settings for all models)

---

## 2. Model Definition
### 2.1 CNN Baseline
- ImageNet-pretrained CNN (e.g., ResNet, EfficientNet)
- Replace final classification layer

### 2.2 ViT Baseline
- ImageNet-pretrained ViT (e.g., ViT-B/16, DeiT)
- Linear classification head

### 2.3 Hybrid CNN–ViT
- CNN stem for local feature extraction
- Patch/token embedding
- Transformer encoder
- Classification head

---

## 3. Baseline Training (Fair Comparison)
### 3.1 Shared Training Setup
- Optimizer: AdamW
- Learning rate schedule: cosine decay + warmup
- Same batch size and epochs/steps
- Same early stopping rule (or none)

### 3.2 Train on Food-101
- Train → validate → test
- Save best checkpoint
- Log metrics and training time

### 3.3 Train on SGFood233
- Repeat identical procedure

---

## 4. Transfer Learning
### 4.1 Food-101 → SGFood233 Finetuning
- Initialize models with Food-101 weights
- Replace classifier head
- Finetune on SGFood233
- Compare with training from scratch

### 4.2 (Optional) Zero-shot Evaluation
- Train on Food-101 → test on SGFood233
- Train on SGFood233 → test on Food-101

---

## 5. Model Optimization
### 5.1 Light Tuning (All Models)
- Tune learning rate, weight decay, dropout
- Same number of trials per model

### 5.2 Deep Tuning (Final Model Only)
- Further optimize best-performing model (Hybrid)
- Stronger augmentation or longer training
- Clearly label as final selected system

---

## 6. Evaluation & Analysis
### 6.1 Performance Metrics
- Top-1 Accuracy
- Macro-F1
- Mean ± std over seeds
- Per-class recall

### 6.2 Efficiency Metrics
- Number of parameters
- FLOPs
- Inference latency
- Training time

### 6.3 Error Analysis
- Confusion matrices
- Common failure classes
- Visual inspection of misclassifications

---

## 7. Reporting
- In-domain results (Food-101, SGFood233)
- Transfer learning results
- Efficiency comparison
- Discussion on why Hybrid CNN–ViT performs best
