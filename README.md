# 🧠 Cranial Deformity Classification — Clinical-Grade 3D Deep Learning

A **clinical-grade diagnostic pipeline** that classifies cranial (skull) deformities from 3D `.obj` mesh files using volumetric deep learning. The system converts 3D skull models into voxel grids and classifies them using three architectures — **3D CNN**, **3D ResNet**, and **3D DenseNet** — augmented with **quantitative clinical metrics** and **epistemic uncertainty estimation** for safe, interpretable medical AI.

> This project is designed as a **Software as a Medical Device (SaMD)** prototype, demonstrating how deep learning can be responsibly deployed in pediatric craniofacial diagnostics.

---

## 🏗️ Architecture Overview

```
.obj Mesh Files
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Trimesh     │────▶│  Voxelization    │────▶│  Label Encoding    │
│  Loader      │     │  (32×32×32)      │     │  (One-Hot)         │
└─────────────┘     └──────────────────┘     └────────────────────┘
                                                      │
                          ┌───────────────────────────┤
                          ▼               ▼           ▼
                    ┌──────────┐   ┌──────────┐  ┌──────────┐
                    │  3D CNN  │   │ 3D ResNet│  │3D DenseNet│
                    └──────────┘   └──────────┘  └──────────┘
                          │               │           │
                          ▼               ▼           ▼
                    ┌─────────────────────────────────────┐
                    │      Clinical Diagnostic Report     │
                    │  • Cephalic Index (from 3D mesh)    │
                    │  • MC Dropout Uncertainty Score     │
                    │  • Per-class probability ± std      │
                    └─────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔬 Core ML Pipeline
- **3D Mesh → Voxel Conversion**: Converts `.obj` skull models into 32×32×32 binary voxel grids using `trimesh`
- **Multi-Model Training**: Trains and compares 3D CNN, 3D ResNet (with residual blocks), and 3D DenseNet
- **Comprehensive Evaluation**: Confusion matrices, accuracy/loss curves, PCA projections, voxel slice visualizations

### 📊 Feature 1 — Quantitative Clinical Metrics (Cephalic Index)
- Extracts **3D bounding box extents** from skull meshes
- Computes the **Cephalic Index (CI)** = `(biparietal_width / anteroposterior_length) × 100`
- Classifies severity:
  - **CI > 81** → Brachycephaly (abnormally wide skull)
  - **CI < 75** → Dolichocephaly (abnormally elongated skull)
  - **75 ≤ CI ≤ 81** → Normal

### 🎯 Feature 2 — Epistemic Uncertainty Estimation (Monte Carlo Dropout)
- Implements **MC Dropout** (Gal & Ghahramani, 2016) for calibrated confidence scores
- Runs **N=50 stochastic forward passes** with Dropout active at inference
- Reports **per-class probability distributions** (mean ± std)
- Generates clinical recommendations:
  - **Confidence < 75%** → `⚠️ Manual Clinical Review Required`
  - **Confidence ≥ 75%** → `✅ High Confidence Prediction`

### 📋 Clinical Diagnostic Report
- Combines both features into a unified **SaMD-style diagnostic report**
- Reports true label, model prediction, Cephalic Index, confidence score, and per-class uncertainty
- Demonstrates on randomized test samples

---

## 📁 Dataset

The project uses the **modelsSynth** dataset — a collection of synthetic 3D skull meshes in `.obj` format, organized by deformity class.

**Expected structure after extraction:**
```
dataset/modelsSynth/
├── N/          # Normal skulls
├── BP/         # Brachycephaly / Plagiocephaly
├── ...         # Other deformity classes
```

> ⚠️ Place `modelsSynth.zip` in the working directory before running.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn trimesh tensorflow scikit-learn
```

| Package        | Purpose                              |
|----------------|--------------------------------------|
| `trimesh`      | 3D mesh loading & voxelization       |
| `tensorflow`   | Deep learning models (3D CNN/ResNet/DenseNet) |
| `scikit-learn` | Label encoding, train/test split, PCA, confusion matrices |
| `matplotlib`   | Training curves, voxel visualizations |
| `seaborn`      | Class distribution plots              |
| `pandas`       | Feature dataframe construction        |
| `numpy`        | Numerical operations                  |

---

## 🚀 How to Run

This script is designed to run inside **Google Colab** with the dataset uploaded:

1. Upload `modelsSynth.zip` to `/content/`
2. Run all cells in order — the script handles extraction, preprocessing, training, and evaluation
3. At the end, a **Clinical Diagnostic Report** is printed for 3 random test samples

---

## 📈 Model Architectures

| Model | Architecture | Key Properties |
|-------|-------------|----------------|
| **3D CNN** | Conv3D(32) → Pool → Conv3D(64) → Pool → Dense(128) → Dropout(0.3) → Softmax | Lightweight baseline |
| **3D ResNet** | Conv3D(64, 7×7×7) → Pool → 2 Residual Blocks → GAP → Dense(128) → Softmax | Skip connections prevent vanishing gradients |
| **3D DenseNet** | Conv3D(24) → 3 Dense Blocks (4 convs each, k=12) → Transitions → GAP → Softmax | Feature reuse via concatenation |

---

## 📊 Outputs & Visualizations

- Training & validation **accuracy/loss curves** (CNN vs ResNet vs DenseNet)
- **Confusion matrices** for each model
- **Voxel slice visualizations** (axial cross-sections of 3D grids)
- **PCA projection** of flattened voxel data
- **Class distribution** bar charts
- **Clinical Diagnostic Reports** with Cephalic Index + MC Dropout scores

---

## 📄 License

This project is for academic and research purposes.

---

## 👤 Author

**Kabir** — [GitHub](https://github.com/kabir-pjm)
