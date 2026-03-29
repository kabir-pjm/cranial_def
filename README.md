# CranialAI — 3D Deep Learning for Cranial Deformity Classification

> Automated detection and severity grading of cranial deformities from 3D skull meshes using volumetric CNNs, with Monte Carlo Dropout uncertainty estimation and Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![trimesh](https://img.shields.io/badge/trimesh-3D-teal)
![License](https://img.shields.io/badge/License-MIT-gray)

---

## Problem Statement

Cranial deformities (Brachycephaly, Plagiocephaly, Dolichocephaly) affect ~47% of infants. Early detection is critical — untreated cases can lead to developmental issues. Current diagnosis relies on subjective visual assessment and manual caliper measurements, introducing inter-observer variability.

This system automates the diagnostic pipeline: 3D skull scan → volumetric deep learning → classified deformity type + severity grade + model confidence + visual explanation of the decision.

### Clinical Relevance

| Condition | Cephalic Index | Description | Prevalence |
|-----------|---------------|-------------|------------|
| Normocephaly | 75–81 | Normal skull proportions | ~53% |
| Brachycephaly | >81 | Short, wide skull | ~25% |
| Plagiocephaly | — | Asymmetric flattening | ~15% |
| Dolichocephaly | <75 | Long, narrow skull | ~7% |

### Why This Architecture Matters Beyond Medicine

The pipeline pattern — **3D data ingestion → feature engineering → multi-model comparison → uncertainty quantification → explainability** — applies directly to:

| Domain | 3D Input | Classification | Uncertainty Use |
|--------|----------|---------------|-----------------|
| **This Project** | Skull meshes | Deformity type | Flag low-confidence for review |
| **Fraud Detection** | Transaction graphs | Fraud/legitimate | Flag uncertain cases for analysts |
| **Credit Risk** | Multi-dimensional features | Default probability | Confidence-based approval tiers |
| **AML Compliance** | Network topology | Suspicious activity | Prioritize investigation queue |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DATA LAYER                             │
│  ┌─────────────┐                                         │
│  │ OBJ Meshes  │  modelsSynth/                           │
│  │ (3D skulls) │  ├── N/   (Normal)                      │
│  └──────┬──────┘  ├── BP/  (Brachycephaly)               │
│         │         ├── P/   (Plagiocephaly)                │
│         ▼         └── D/   (Dolichocephaly)               │
│  ┌─────────────────┐                                     │
│  │ MESH PROCESSOR   │                                     │
│  │ • Voxelization   │→ 32×32×32 binary occupancy grid    │
│  │ • Cephalic Index │→ Clinical severity grading         │
│  │ • Asymmetry      │→ Left-right imbalance ratio        │
│  └────────┬─────────┘                                     │
│           ▼                                               │
│  ┌────────────────────────────────────────┐               │
│  │    3-MODEL COMPARISON FRAMEWORK        │               │
│  │  ┌────────┐ ┌────────┐ ┌───────────┐  │               │
│  │  │3D CNN  │ │3D      │ │3D         │  │               │
│  │  │(base)  │ │ResNet  │ │DenseNet   │  │               │
│  │  │~50K    │ │(skip   │ │(feature   │  │               │
│  │  │params  │ │connect)│ │reuse)     │  │               │
│  │  └────┬───┘ └───┬────┘ └────┬──────┘  │               │
│  │       └─────────┼───────────┘          │               │
│  └─────────────────┼──────────────────────┘               │
│                    ▼                                      │
│  ┌─────────────────────────────────────┐                  │
│  │    SAFETY & EXPLAINABILITY LAYER     │                  │
│  │                                     │                  │
│  │  MC Dropout ──→ Epistemic           │                  │
│  │  (N=50 passes)  Uncertainty         │                  │
│  │                                     │                  │
│  │  3D Grad-CAM ──→ Attention          │                  │
│  │  (gradient viz)  Heatmap            │                  │
│  │                                     │                  │
│  │  Clinical ────→ CI + Severity       │                  │
│  │  Metrics        + Risk Level        │                  │
│  └─────────────────┬───────────────────┘                  │
│                    ▼                                      │
│  ┌──────────────────────────────────────────────────┐     │
│  │           CLINICAL DIAGNOSTIC REPORT              │     │
│  │  Prediction │ Confidence │ Attention │ Risk Level │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### 1. Voxelization (OBJ → 32×32×32 Grid)
3D CNNs can't process triangular meshes directly. Voxelization converts continuous surfaces into discrete binary occupancy grids — each cell is 1 (inside mesh) or 0 (outside). The pitch is computed as `max_extent / 32`, normalizing all skulls to the same grid size regardless of absolute dimensions.

### 2. Three-Architecture Comparison
- **3D CNN**: Lightweight baseline. Fast training, interpretable. Risk: shallow depth may miss subtle morphological patterns.
- **3D ResNet**: Skip connections solve vanishing gradients in deeper networks. Better feature extraction but more parameters.
- **3D DenseNet**: Each layer receives features from ALL preceding layers, maximizing feature reuse and reducing parameters. Best accuracy but slowest training.

### 3. Monte Carlo Dropout — Uncertainty Quantification
Standard softmax is NOT a calibrated probability. A model can output 99% confidence and be completely wrong. In medical AI, this is dangerous.

MC Dropout runs N=50 stochastic forward passes with Dropout active at inference. The variance across passes estimates **epistemic uncertainty** — uncertainty due to limited data, not inherent noise. High variance → flag for human review.

### 4. 3D Grad-CAM — Explainability
A regulatory requirement for medical AI (FDA SaMD, EU MDR). Shows WHERE the model is looking — which skull regions influenced the classification. Anterior/posterior/left/right attention distribution reveals whether the model learned clinically meaningful features.

### 5. Cephalic Index — Clinical Validation
The automated prediction is cross-validated against the quantitative Cephalic Index (CI = width/length × 100), a standard craniometric measurement. Agreement between ML prediction and CI-based classification builds clinical confidence.

---

## Quick Start

```bash
git clone https://github.com/[you]/cranial-deformity.git
cd cranial-deformity
pip install -r requirements.txt

# Train on dataset
python src/pipeline.py --dataset path/to/modelsSynth/ --epochs 20

# Open dashboard
open dashboard/index.html

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
cranial-deformity/
├── src/
│   ├── pipeline.py         # Full pipeline: mesh processing, models, training
│   └── explainability.py   # 3D Grad-CAM implementation
├── dashboard/
│   └── index.html           # Interactive clinical dashboard
├── tests/
│   └── test_pipeline.py     # Unit tests
├── data/                    # Dataset (not tracked in git)
├── models/                  # Saved model weights
├── docs/                    # Additional documentation
├── requirements.txt
└── README.md
```

---

## Results

| Model | Val Accuracy | Parameters | Training Time |
|-------|-------------|------------|---------------|
| 3D CNN | ~82% | ~50K | Fast |
| 3D ResNet | ~88% | ~200K | Medium |
| 3D DenseNet | ~91% | ~150K | Slow |

---

## Future Enhancements

- [ ] Point cloud processing (PointNet++) as mesh-native alternative to voxelization
- [ ] Aleatoric uncertainty via heteroscedastic loss
- [ ] DICOM integration for clinical CT scan input
- [ ] Federated learning for multi-hospital training without data sharing
- [ ] ONNX export for edge deployment on clinic hardware

---

## License

MIT
