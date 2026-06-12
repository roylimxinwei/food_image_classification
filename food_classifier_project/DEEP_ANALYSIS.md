# Deep Analysis Methodology for SGFood233 Classification

This document provides detailed documentation for the comprehensive analysis comparing CNN (ResNet50), Vision Transformer (ViT), and ConvNeXt architectures on the SGFood233 dataset.

## Overview

The deep analysis notebook (`09_deep_analysis.ipynb`) performs four types of analysis to provide insights beyond simple accuracy metrics:

1. **Error Analysis** - Understand where and why models fail
2. **GradCAM Explainability** - Visualize what models focus on
3. **Efficiency Analysis** - Compare computational costs
4. **Embedding Space Analysis** - Analyze learned representations

---

## Analysis 1: Error Analysis

### Purpose
Error analysis reveals the failure patterns unique to each architecture, helping us understand their inherent biases and limitations.

### Methodology

#### 1.1 Per-Class Metrics
- **Precision**: Of all predictions for class X, what fraction are correct?
- **Recall**: Of all actual class X samples, what fraction did we find?
- **F1-Score**: Harmonic mean of precision and recall

```python
from sklearn.metrics import classification_report, precision_recall_fscore_support
```

For 233 classes, we focus on:
- Classes with F1 < 0.5 (struggling classes)
- Classes with F1 > 0.9 (easy classes)
- Cross-architecture comparison heatmap

#### 1.2 Confusion Analysis
With 233 classes (54,289 possible pairs), we identify the **top 15 most confused pairs** per model:

```python
# Sort off-diagonal elements of confusion matrix
confusion_pairs = []
for i in range(n_classes):
    for j in range(n_classes):
        if i != j and cm[i, j] > threshold:
            confusion_pairs.append((classes[i], classes[j], cm[i, j]))
```

**Key Insight**: Different architectures confuse different class pairs due to their inductive biases:
- **CNN (ResNet50)**: Confuses classes with similar textures/colors
- **ViT**: Confuses classes with similar global structure
- **ConvNeXt**: More balanced, fewer systematic confusions

#### 1.3 Failure Case Visualization
Display images where models fail with highest confidence (worst mistakes):

```python
# Find high-confidence errors
wrong_mask = predictions != labels
loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
hardest_indices = loss_per_sample[wrong_mask].argsort(descending=True)[:10]
```

---

## Analysis 2: GradCAM Explainability

### Purpose
Gradient-weighted Class Activation Mapping (GradCAM) reveals which image regions drive predictions, exposing architectural differences in visual attention.

### Methodology

#### Library
```python
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
```

#### Target Layers by Architecture

| Model | Target Layer | Rationale |
|-------|--------------|-----------|
| ResNet50 | `model.layer4[-1]` | Last residual block, richest features |
| ViT | `model.blocks[-1].norm1` | Last transformer block normalization |
| ConvNeXt | `model.stages[-1].blocks[-1]` | Last ConvNeXt block |

#### Expected Observations

**CNN (ResNet50)**
- Focuses on local texture patterns (grain, color gradients)
- Attention spreads across textured regions
- May miss overall dish structure

**Vision Transformer (ViT)**
- Attends to global dish structure
- May focus on dish boundaries and key components
- Can miss fine-grained details important for distinguishing similar foods

**ConvNeXt**
- Balanced local and global attention
- Attends to both texture AND structure
- Explains superior accuracy through complementary attention patterns

### Visualization Grid
Create comparison figure:
- Rows: Sample images (correct, misclassified, edge cases)
- Columns: Original | ResNet50 | ViT | ConvNeXt

---

## Analysis 3: Efficiency Analysis

### Purpose
Beyond accuracy, model selection requires understanding computational trade-offs for deployment scenarios.

### Metrics

| Metric | Measurement Method | Importance |
|--------|-------------------|------------|
| **Parameter Count** | `sum(p.numel() for p in model.parameters())` | Memory footprint |
| **Model Size (MB)** | Checkpoint file size | Storage requirements |
| **FLOPs** | fvcore FlopCountAnalysis | Computational cost |
| **Inference Speed** | Average over 100+ images (with warmup) | Latency |
| **GPU Memory** | `torch.cuda.max_memory_allocated()` | Deployment constraints |

### Implementation

```python
from fvcore.nn import FlopCountAnalysis

# Parameter count
params = sum(p.numel() for p in model.parameters())

# FLOPs
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(model, dummy_input)
total_flops = flops.total()

# Inference speed (with warmup)
for _ in range(10):  # Warmup
    _ = model(dummy_input)

times = []
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = model(dummy_input)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_inference_ms = sum(times) / len(times)
```

### Expected Results

| Model | Params | FLOPs | Speed (img/s) | Memory |
|-------|--------|-------|---------------|--------|
| ResNet50 | ~25M | ~4.1G | Fastest | Low |
| ViT-Base | ~86M | ~17.6G | Slowest | High |
| ConvNeXt-Base | ~89M | ~15.4G | Medium | High |

### Pareto Analysis
Plot **Accuracy vs Efficiency** scatter plot to identify optimal trade-offs:
- X-axis: Inference speed (images/second)
- Y-axis: Validation accuracy
- Ideal models appear in upper-right quadrant

---

## Analysis 4: Embedding Space Analysis

### Purpose
Visualizing the learned representation space reveals how well models separate food classes and identifies clustering quality.

### Methodology

#### 4.1 Feature Extraction
Extract features from the penultimate layer (before classifier head):

```python
from utils import FeatureExtractor, extract_all_features

extractor = FeatureExtractor(model, model_name)
features, labels = extract_all_features(
    model, model_name, val_loader,
    device='cuda', max_samples=5000  # Subsample for efficiency
)
```

Feature dimensions:
- ResNet50: 2048
- ViT-Base: 768
- ConvNeXt-Base: 1024

#### 4.2 Dimensionality Reduction

**t-SNE** (t-distributed Stochastic Neighbor Embedding)
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(features)
```

**UMAP** (Uniform Manifold Approximation and Projection)
```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = reducer.fit_transform(features)
```

#### 4.3 Quantitative Evaluation

**Silhouette Score**: Measures cluster separation quality
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(embeddings_2d, labels)
# Range: -1 to 1, higher is better
```

**Interpretation**:
- > 0.5: Strong cluster structure
- 0.25-0.5: Reasonable structure
- < 0.25: Weak/overlapping clusters

### Expected Insights

1. **ConvNeXt** should show tightest, most separated clusters (best representations)
2. **ViT** may show good global structure but some class overlap
3. **ResNet50** may show more overlap between visually similar food classes

### Visualization
- 3 side-by-side scatter plots (one per model)
- Points colored by class (use categorical colormap)
- For 233 classes, consider grouping into food categories for clearer visualization

---

## Dependencies

Install required packages:
```bash
pip install grad-cam fvcore umap-learn seaborn scikit-learn
```

## File Structure

```
food_classifier_project/
├── 09_deep_analysis.ipynb     # Main analysis notebook
├── DEEP_ANALYSIS.md           # This documentation
├── utils.py                   # Updated with FeatureExtractor
└── checkpoints/
    ├── sgfood233_resnet50/
    ├── sgfood233_vit_base_patch16_224/
    └── sgfood233_convnext_base/
```

## Expected Outputs

After running the notebook, you will have:

1. **Tables**:
   - Per-model accuracy, macro-F1, top confused pairs
   - Efficiency metrics (params, FLOPs, speed, memory)

2. **Figures**:
   - Per-class F1 heatmap (3 models x 233 classes)
   - GradCAM comparison grid (sample images x 3 models)
   - Efficiency bar charts and Pareto plot
   - t-SNE/UMAP embedding visualizations

3. **Key Findings**:
   - Why ConvNeXt achieves best accuracy
   - Which food classes are inherently difficult
   - Deployment recommendations based on accuracy vs efficiency trade-offs

---

## References

- **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
- **t-SNE**: van der Maaten & Hinton, "Visualizing Data using t-SNE" (JMLR 2008)
- **UMAP**: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018)
- **ConvNeXt**: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
