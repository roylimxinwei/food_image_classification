# Building a food recognition and classification model for calorie estimation (Final Year Project)

## Presentation
- [Final Report](https://dr.ntu.edu.sg/entities/publication/50f6beb6-d83c-4f08-9f13-f5ce74cecd6c)
- [Download PowerPoint](./slides/presentation.pptx)

Food image classifier for 233 Singaporean food categories, trained on ~203K images. Best model (ConvNeXt-Base) achieves 84.55% validation accuracy on the cleaned dataset.

## Repository Layout

```
food image classification/
├── food_classifier_project/     # Core training pipeline (PyTorch Lightning + timm)
├── reactnative_frontend/foodai/ # Expo React Native mobile app
├── fastapi_backend/             # FastAPI inference server (WIP)
├── training/                    # [Legacy] Early fastai experiments (superseded)
├── main.py                      # [Legacy] Streamlit app using fastai (superseded)
├── convert_checkpoint_to_onnx.py
├── foodsg233_labels.json        # Class ID ↔ label mapping (233 classes)
└── requirements.txt
```

Datasets (not tracked in git):
```
foodsg-233/                      # Raw dataset (204,305 images)
foodsg-233_cleaned/              # After duplicate/mislabel removal (203,655 images)
foodsg-233_train_val_test_split/ # 75/15/10 split used for training
```

## Components

### food_classifier_project/ — Training Pipeline

PyTorch Lightning training notebooks using timm pretrained backbones. All notebooks import shared code from `utils.py`.

| Notebooks | Purpose |
|-----------|---------|
| `00_data_cleaning.ipynb` | Duplicate and near-duplicate detection |
| `01–03_*_food101.ipynb` | Food-101 baseline experiments (ResNet50, ViT, ConvNeXt) |
| `04–06_*_sgfood233.ipynb` | SGFood-233 experiments |
| `09*_deep_analysis.ipynb` | Embedding analysis, t-SNE, UMAP, explainability |
| `10_mislabel_detection.ipynb` | Multi-model mislabel flagging |
| `11_post_cleaning_evaluation.ipynb` | Retrain on cleaned data |
| `12_hyperparameter_tuning.ipynb` | LR sensitivity across backbones |
| `13_convnext_deep_tuning.ipynb` | ConvNeXt HP optimization + final evaluation |

Best checkpoints in `checkpoints/`:
| Checkpoint | Val Acc | Notes |
|-----------|---------|-------|
| `sgfood233_convnext_base/epoch=09-val_acc=0.8345.ckpt` | 83.45% | Trained on dirty 75/25 split |
| `sgfood233_convnext_base_cleaned/epoch=09-val_acc=0.8455.ckpt` | 84.55% | Trained on clean 75/15/10 split |

### reactnative_frontend/ — Mobile App

Expo React Native app (TypeScript + NativeWind). Sends images to the FastAPI backend for inference and displays predictions.

The backend for this app lives in a separate repo: **SAM3**.

### training/ — Legacy (fastai)

Earlier Food-101 experiments using the fastai library. Superseded by the organized notebooks in `food_classifier_project/`. Not actively maintained.

### main.py — Legacy (fastai + Streamlit)

Streamlit web app for comparing multiple models, using fastai for inference. Mostly superseded by the FastAPI + React Native stack.

## Setup

### Activate Virtual Environment

**Command Prompt:**
```
.venv\Scripts\activate.bat
```

**PowerShell:**
```
.venv\Scripts\Activate.ps1
```

**Git Bash:**
```
source .venv/Scripts/activate
```

### Install PyTorch (CUDA 12.8)

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### Install other dependencies

```bash
pip install -r requirements.txt
```

## Mobile App — Connecting to Backend

The React Native app calls the FastAPI backend (SAM3 repo) running in WSL2.

**Initial setup — run as Administrator in PowerShell:**
```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-Command', 'netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>; netsh advfirewall firewall add rule name="WSL2 Port 8000" dir=in action=allow protocol=TCP localport=8000'
```

**If the Windows IP changes:**
1. Find new IP: `ipconfig | findstr "IPv4"`
2. Update `BASE_URL` in `reactnative_frontend/foodai/config/`:
   ```
   BASE_URL: "http://YOUR_NEW_IP:8000"
   ```

**If the WSL IP changes:**
```powershell
# Get new WSL IP
wsl ip addr show eth0 | findstr "inet "

# Update port forwarding (run as Admin)
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<NEW_WSL_IP>
```
