# Food Classifier: CNN vs ViT vs Hybrid Comparison

This project compares three model architectures for food image classification:

| Model | Architecture | Parameters |
|-------|-------------|------------|
| **CNN** | ResNet50 | ~25M |
| **ViT** | ViT-Base/16 | ~86M |
| **Hybrid** | ConvNeXt-Base | ~89M |

## Datasets

### Food-101 (Hugging Face)
- **Source:** `ethz/food101`
- **Classes:** 101 food categories
- **Images:** ~101,000 (75K train, 25K test)
- **Access:** Automatic download via `datasets` library

### SGFood233 (Local)
- **Source:** [NUS Food(lg)](https://foodlg.comp.nus.edu.sg/)
- **Classes:** 233 Singaporean dishes
- **Images:** ~210,000
- **Access:** Requires registration

## Project Structure

```
food_classifier_project/
├── utils.py                    # Shared utilities (Config, datasets, model)
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── 01_cnn_food101.ipynb       # CNN on Food-101
├── 02_vit_food101.ipynb       # ViT on Food-101
├── 03_hybrid_food101.ipynb    # ConvNeXt on Food-101
│
├── 04_cnn_sgfood233.ipynb     # CNN on SGFood233
├── 05_vit_sgfood233.ipynb     # ViT on SGFood233
├── 06_hybrid_sgfood233.ipynb  # ConvNeXt on SGFood233
│
├── checkpoints/               # Saved models (created during training)
├── logs/                      # TensorBoard logs (created during training)
└── data/                      # Local datasets (for SGFood233)
    └── sgfood233/
        ├── train/
        ├── val/
        └── test/
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks

For **Food-101** (auto-downloads from Hugging Face):
```bash
# Open in VSCode or Jupyter
jupyter notebook 01_cnn_food101.ipynb
```

For **SGFood233** (requires manual download):
1. Register at https://foodlg.comp.nus.edu.sg/
2. Download and extract to `data/sgfood233/`
3. Organize into `train/`, `val/`, `test/` folders
4. Run notebooks 04-06

## Configuration

All training parameters are in `utils.py`:

```python
class Config:
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 10
    SEED = 42
```

Modify these for your hardware/needs.

## Training Tips

### GPU Memory Issues
- Reduce `BATCH_SIZE` in `utils.py`
- Use gradient accumulation (modify trainer)

### Windows Users
- Set `NUM_WORKERS = 0` in `utils.py` if you get multiprocessing errors

### Monitoring Training
```bash
# Start TensorBoard
tensorboard --logdir=logs/
```

## Experiment Comparison

After training all 6 models, compare:

| Dataset | CNN (ResNet50) | ViT-Base | ConvNeXt |
|---------|---------------|----------|----------|
| Food-101 | ? | ? | ? |
| SGFood233 | ? | ? | ? |

## Files Reused from Colab

Your existing Colab code components that were incorporated:
- ✅ `HFImageDataset` wrapper
- ✅ `FoodClassifier` Lightning module  
- ✅ Train/val transforms (ImageNet normalization)
- ✅ ONNX export code
- ✅ AdamW + ReduceLROnPlateau → upgraded to CosineAnnealing

## Next Steps (from your markdown plan)

- [ ] Transfer learning: Food-101 → SGFood233
- [ ] Confusion matrix analysis
- [ ] Inference latency benchmarking
- [ ] FLOPs comparison
