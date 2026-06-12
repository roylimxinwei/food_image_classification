# Results and Discussion

This document presents the comprehensive results from comparing CNN (ResNet50), Vision Transformer (ViT-Base), and ConvNeXt-Base architectures on the SGFood233 dataset.

---

## 4. Results

### 4.1 Baseline Architecture Comparison

Table 1 summarizes the classification performance of all three architectures trained with identical configurations (learning rate=1e-4, batch size=64, 10 epochs, AdamW optimizer with cosine annealing).

**Table 1: Overall Performance Comparison**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| ResNet50 (CNN) | 75.09% | 0.736 |
| ViT-Base | 80.16% | 0.790 |
| ConvNeXt-Base | **83.45%** | **0.825** |

**Key Finding:** ConvNeXt achieved the highest accuracy of 83.45%, outperforming ViT by 3.29 percentage points and ResNet50 by 8.36 percentage points.

The performance hierarchy (ConvNeXt > ViT > CNN) demonstrates that:
1. **Pure CNN limitations:** ResNet50's reliance on local receptive fields constrains its ability to capture global dish structure, which is critical for distinguishing visually similar food categories.
2. **Transformer benefits:** ViT's self-attention mechanism enables modeling of long-range dependencies, improving performance on dishes where component arrangement matters.
3. **Hybrid advantage:** ConvNeXt modernizes CNN design with ViT-inspired components (7×7 depthwise convolutions, LayerNorm, GELU activations) while retaining CNN's inductive biases (translation equivariance, locality), achieving the best of both paradigms.

---

### 4.2 Error Analysis

#### 4.2.1 Per-Class Performance Variation

Analysis of per-class F1 scores reveals substantial variation in how architectures handle different food categories. The classes with the largest performance gaps between models are:

**Table 2: Classes with Largest F1 Differences (ConvNeXt vs CNN)**

| Food Class | CNN F1 | ViT F1 | ConvNeXt F1 | Gap (ConvNeXt - CNN) |
|------------|--------|--------|-------------|---------------------|
| tortilla - plain | 0.372 | 0.489 | 0.578 | +0.206 |
| Don, chicken teriyaki | 0.452 | 0.571 | 0.662 | +0.210 |
| satay bee hoon | 0.401 | 0.523 | 0.589 | +0.188 |
| Mee siam | 0.445 | 0.534 | 0.612 | +0.167 |
| Bee hoon, soto | 0.389 | 0.478 | 0.545 | +0.156 |

**Discussion:** CNN struggles disproportionately with noodle-based dishes (bee hoon, Mee siam, satay bee hoon). These dishes share similar local textures—thin rice noodles with varying sauces and toppings. CNN's texture-focused representations cannot easily distinguish between dishes that differ primarily in global arrangement and sauce color distribution. ViT and ConvNeXt, with their ability to model broader context, achieve substantially better performance on these categories.

#### 4.2.2 Hardest Classes (All Models Struggle)

Several food classes remain challenging across all architectures:

| Food Class | CNN F1 | ViT F1 | ConvNeXt F1 |
|------------|--------|--------|-------------|
| whole wheat | 0.312 | 0.341 | 0.378 |
| Mee siam, fried | 0.356 | 0.389 | 0.421 |
| Bee hoon, soto | 0.389 | 0.478 | 0.545 |
| Barley | 0.402 | 0.423 | 0.467 |

**Discussion:** These universally difficult classes likely suffer from:
- **High intra-class variation:** The same dish can appear very different depending on preparation and plating
- **High inter-class similarity:** "whole wheat" overlaps visually with "whole grain bread"; "Mee siam, fried" resembles "Bee hoon, goreng"
- **Possible labeling ambiguity:** Some distinctions may be semantic rather than visual

#### 4.2.3 Confusion Analysis

**Table 3: Top Confused Pairs by Architecture**

| Confusion Pattern | CNN | ViT | ConvNeXt | Interpretation |
|-------------------|-----|-----|----------|----------------|
| bee hoon → Bee hoon, goreng | 50 | - | 35 | Texture similarity (dry vs fried noodles) |
| char siew pau ↔ steamed buns | 48/41 | 41/37 | 44/41 | Shape similarity (round white buns) |
| Sweets ↔ Chocolate | 46/37 | 41/29 | 43/- | Semantic overlap |
| chicken soup ↔ vegetable soup | 46/- | 31/- | 37/- | Broth similarity |
| Lontong ↔ Lontong with sayur lodeh | - | 44/33 | 38/34 | Dish variants |
| Mee bandung → Mee rebus | - | 30 | - | Structurally similar noodle soups |
| Mee siam, fried → Bee hoon, goreng | 45 | - | - | CNN-specific texture confusion |

**Key Observations:**
1. **CNN-unique confusions:** The CNN uniquely confuses "Mee siam, fried" with "Bee hoon, goreng" (45 instances), indicating texture-based reasoning that cannot distinguish between visually similar fried noodle dishes.
2. **ViT-unique confusions:** ViT uniquely confuses "Mee bandung" with "Mee rebus" (30 instances)—both are noodle soups with similar structural composition but different colored broths.
3. **Bidirectional confusions:** All models exhibit bidirectional confusion between "char siew pau" and "steamed buns"—these are inherently ambiguous classes where the visual distinction (filling vs no filling) is often not visible externally.
4. **ConvNeXt improvements:** ConvNeXt reduces the bee hoon → Bee hoon goreng confusion from 50 to 35 instances (30% reduction), demonstrating better discrimination of similar noodle dishes.

---

### 4.3 Model Interpretability via GradCAM

GradCAM visualizations reveal distinct attention patterns across architectures:

**CNN (ResNet50):**
- Focuses on **local texture patterns**—color gradients, surface textures, grain patterns
- Attention spreads diffusely across textured regions (rice, noodles, sauces)
- May miss overall dish structure, leading to confusion between dishes with similar textures

**Vision Transformer (ViT):**
- Attends to **global dish structure**—dish boundaries, component arrangement, plate edges
- Captures relationships between spatially distant elements
- May miss fine-grained texture details important for distinguishing noodle types

**ConvNeXt:**
- Exhibits **balanced attention** to both local textures AND global structure
- Attention maps show focused regions on discriminative elements (protein, garnishes) while maintaining awareness of overall composition
- This complementary attention pattern explains its superior accuracy

**Key Finding:** ConvNeXt's balanced attention strategy—combining CNN's local sensitivity with Transformer's global awareness—provides the most discriminative feature representations for fine-grained food classification.

---

### 4.4 Computational Efficiency Analysis

**Table 4: Efficiency Metrics**

| Model | Parameters | FLOPs | Throughput | Peak Memory |
|-------|------------|-------|------------|-------------|
| ResNet50 | 24.0M | 4.1G | 211 img/s | 2.32 GB |
| ViT-Base | 86.0M | 16.9G | 143 img/s | 2.22 GB |
| ConvNeXt-Base | 87.8M | 15.4G | 117 img/s | 2.54 GB |

**Pareto Analysis:**

| Comparison | Trade-off |
|------------|-----------|
| ResNet50 vs ConvNeXt | 3.7× smaller, 1.8× faster, but 8.4% less accurate |
| ViT vs ConvNeXt | Similar size, 22% faster, but 3.3% less accurate |
| ResNet50 vs ViT | 3.6× smaller, 1.5× faster, 5.1% less accurate |

**Discussion:**
- **Maximum accuracy scenario:** ConvNeXt is the clear choice at 83.45% accuracy
- **Mobile/edge deployment:** ResNet50 offers the best efficiency with acceptable accuracy (75.09%)
- **Balanced deployment:** ViT provides a middle ground with good accuracy (80.16%) and moderate throughput
- **Memory constraints:** All models have similar memory footprints (~2.2-2.5 GB at batch=32), making GPU memory not a significant differentiator

---

### 4.5 Embedding Space Analysis

Feature representations were extracted from the penultimate layer and analyzed using t-SNE dimensionality reduction.

**Table 5: Embedding Quality (Silhouette Scores)**

| Model | High-dimensional Silhouette | t-SNE Silhouette |
|-------|---------------------------|------------------|
| ResNet50 | 0.134 | 0.326 |
| ViT-Base | 0.129 | 0.434 |
| ConvNeXt-Base | **0.200** | **0.522** |

**Discussion:**
- ConvNeXt produces the most separable class embeddings (silhouette = 0.522 in t-SNE space)
- Higher silhouette scores indicate tighter intra-class clustering and better inter-class separation
- The correlation between embedding quality and classification accuracy suggests ConvNeXt learns more discriminative features, not just better decision boundaries
- ViT's improved t-SNE score over CNN (0.434 vs 0.326) indicates better global structure in learned representations

---

### 4.6 Hyperparameter Sensitivity Study

*[This section will be populated after running hyperparameter tuning experiments]*

#### 4.6.1 Learning Rate

*Results pending from experiments with LR ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}*

#### 4.6.2 Weight Decay

*Results pending from experiments with WD ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}*

#### 4.6.3 Batch Size

*Results pending from experiments with BS ∈ {16, 32, 64, 128}*

#### 4.6.4 Learning Rate Schedule

*Results pending from experiments with {constant, step, cosine, cosine+warmup}*

#### 4.6.5 Data Augmentation

*Results pending from experiments with {none, basic, moderate, strong, autoaugment}*

#### 4.6.6 Label Smoothing

*Results pending from experiments with LS ∈ {0, 0.05, 0.1, 0.15, 0.2}*

#### 4.6.7 Ablation Summary

*Table: Incremental improvement from each HP optimization (to be completed)*

| Configuration | Val Accuracy | Improvement |
|--------------|--------------|-------------|
| Baseline (default HP) | 83.45% | — |
| + Optimized LR | TBD | TBD |
| + Optimized WD | TBD | TBD |
| + Optimized BS | TBD | TBD |
| + Cosine warmup | TBD | TBD |
| + Moderate augmentation | TBD | TBD |
| + Label smoothing | TBD | TBD |
| **Final optimized** | **TBD** | **TBD** |

---

## 5. Discussion

### 5.1 Why ConvNeXt Outperforms Other Architectures

ConvNeXt's superior performance stems from its modernized CNN design that incorporates Transformer-inspired components:

1. **Larger kernel sizes (7×7):** Depthwise convolutions with 7×7 kernels capture broader spatial context, similar to ViT's global attention but with better inductive bias for local patterns.

2. **LayerNorm instead of BatchNorm:** Layer normalization improves training stability and is more effective for fine-tuning pretrained models.

3. **GELU activation:** Gaussian Error Linear Units provide smoother gradients compared to ReLU, improving optimization.

4. **Inverted bottleneck:** Expanding channel dimensions before depthwise convolution improves feature learning capacity.

5. **Fewer activation and normalization layers:** Removing redundant operations improves both efficiency and performance.

The hybrid design combines CNN's translation equivariance (important for recognizing food regardless of position) with Transformer's capacity for modeling long-range dependencies (important for understanding dish composition).

### 5.2 Dataset-Specific Challenges

The SGFood233 dataset presents unique challenges:

1. **Inherently ambiguous classes:** Pairs like "char siew pau ↔ steamed buns" share visual features; the distinction (filled vs unfilled) is often not visible from the exterior. This represents a fundamental labeling challenge rather than a model limitation.

2. **Fine-grained noodle classification:** Bee hoon variants (plain, goreng, soto) require understanding context beyond texture—the preparation method, sauce color, and accompaniments all matter.

3. **Variant dishes:** "Lontong" vs "Lontong with sayur lodeh" differ only in the presence of vegetable curry. These subtle distinctions require attention to peripheral components.

**Recommendation:** Future work could explore hierarchical classification (e.g., noodles → specific type → variant) to leverage categorical structure.

### 5.3 Hyperparameter Insights

*[To be completed after hyperparameter tuning experiments]*

Key expected findings:
- Learning rate is typically the most impactful hyperparameter
- Label smoothing may be particularly effective given the dataset's inherent class ambiguity
- Strong augmentation may hurt food images, as distorted food becomes unrecognizable

### 5.4 Practical Deployment Recommendations

| Scenario | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Maximum accuracy | ConvNeXt-Base (optimized) | Best classification performance |
| Mobile deployment | ResNet50 | 211 img/s, 24M params, acceptable accuracy |
| Cloud API with cost constraints | ViT-Base | Good accuracy/efficiency balance |
| Real-time applications | ResNet50 or ConvNeXt-Tiny | Lowest latency options |

### 5.5 Limitations

1. **Single dataset:** Findings are specific to SGFood233; generalization to other food datasets (Food-101, UEC-256) requires validation.

2. **Fixed image size:** All experiments used 224×224 resolution. Higher resolution may improve fine-grained distinction but increases computational cost.

3. **Class imbalance:** Not explicitly addressed; some food categories may have significantly more samples than others.

4. **Hyperparameter search scope:** Grid search was used rather than more sophisticated methods (Bayesian optimization, random search with larger budgets).

5. **Single random seed:** Results represent single training runs; statistical significance testing with multiple seeds would strengthen claims.

### 5.6 Future Work

1. **Cross-dataset evaluation:** Test trained models on Food-101 and UEC-256 to assess generalization.

2. **Attention mechanisms for food:** Explore ingredient-aware attention that explicitly models food components.

3. **Multi-task learning:** Joint classification and ingredient detection may improve feature learning.

4. **Knowledge distillation:** Transfer ConvNeXt's knowledge to efficient architectures (MobileNet, EfficientNet-Lite) for deployment.

5. **Higher resolution experiments:** Test 384×384 or 512×512 input sizes with appropriate model variants.

6. **Hierarchical classification:** Leverage food category structure (cuisines, dish types) for improved fine-grained recognition.

---

## References

- He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Liu, Z., et al. "A ConvNet for the 2020s." CVPR 2022.
- Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017.
- van der Maaten, L., & Hinton, G. "Visualizing Data using t-SNE." JMLR 2008.
