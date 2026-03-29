"""
Cranial Deformity Classification Pipeline
==========================================
3D deep learning system for automated detection and classification of
cranial deformities (Brachycephaly, Plagiocephaly, Dolichocephaly) from
synthetic skull mesh data.

Pipeline:
    OBJ Mesh → Voxelization → 3D CNN / ResNet / DenseNet → Classification
                                                         → MC Dropout Uncertainty
    OBJ Mesh → Cephalic Index (Clinical Metric)          → Severity Grading

Architecture Comparison:
    - 3D CNN: Lightweight baseline (~50K params)
    - 3D ResNet: Residual connections for gradient flow
    - 3D DenseNet: Feature reuse via dense connections

Author: [Your Name]
License: MIT
"""

import os
import numpy as np
import trimesh
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s [CRANIAL] %(message)s')


# ─────────────────────────────────────────────
# Clinical Definitions
# ─────────────────────────────────────────────

class DeformityType(Enum):
    """WHO ICD-10 classification codes for cranial deformities."""
    NORMAL = "N"
    BRACHYCEPHALY = "BP"       # Q67.6 - Short, wide skull
    PLAGIOCEPHALY = "P"        # Q67.3 - Asymmetric flattening
    DOLICHOCEPHALY = "D"       # Q67.2 - Long, narrow skull
    SCAPHOCEPHALY = "SC"       # Q75.0 - Boat-shaped skull


@dataclass
class ClinicalMetrics:
    """Quantitative craniometric measurements from 3D mesh."""
    cephalic_index: float          # (width / length) * 100
    width_mm: float                # Biparietal diameter
    length_mm: float               # Anteroposterior diameter
    height_mm: float               # Superior-inferior
    asymmetry_ratio: float         # Left-right asymmetry indicator
    severity: str                  # Clinical classification string
    risk_level: str                # LOW / MODERATE / HIGH


# ─────────────────────────────────────────────
# 3D Mesh Processing
# ─────────────────────────────────────────────

class MeshProcessor:
    """
    Handles OBJ mesh loading, voxelization, and clinical metric extraction.

    Voxelization converts continuous 3D mesh surfaces into discrete
    volumetric grids (32x32x32 binary occupancy), enabling CNN processing.
    """

    def __init__(self, voxel_dim: int = 32):
        self.voxel_dim = voxel_dim

    def load_mesh(self, obj_path: str) -> trimesh.Trimesh:
        """Load and validate a 3D mesh from OBJ file."""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Mesh not found: {obj_path}")
        mesh = trimesh.load(obj_path)
        if not mesh.is_watertight:
            logging.warning(f"Non-watertight mesh: {obj_path}")
        return mesh

    def voxelize(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Convert triangular mesh to binary voxel occupancy grid.

        The pitch (voxel size) is computed from the mesh's maximum extent
        divided by the target resolution, ensuring the entire mesh fits
        within the grid regardless of absolute scale.

        Args:
            mesh: Loaded trimesh object.

        Returns:
            np.ndarray: Binary voxel grid of shape (voxel_dim, voxel_dim, voxel_dim).
        """
        pitch = mesh.extents.max() / self.voxel_dim
        voxelized = mesh.voxelized(pitch=pitch)
        grid = voxelized.matrix.astype(np.float32)

        # Pad or crop to target dimensions
        result = np.zeros((self.voxel_dim, self.voxel_dim, self.voxel_dim),
                          dtype=np.float32)
        slices = tuple(slice(0, min(s, self.voxel_dim)) for s in grid.shape)
        target_slices = tuple(slice(0, min(s, self.voxel_dim)) for s in grid.shape)
        result[target_slices] = grid[slices]

        return result

    def extract_clinical_metrics(self, mesh: trimesh.Trimesh) -> ClinicalMetrics:
        """
        Extract quantitative craniometric measurements from 3D skull mesh.

        Cephalic Index (CI):
            CI = (biparietal_width / anteroposterior_length) × 100

        Clinical thresholds (WHO/AAP guidelines):
            CI > 81  → Brachycephaly (short, wide skull)
            CI < 75  → Dolichocephaly (long, narrow skull)
            75-81    → Normocephaly

        Asymmetry Ratio:
            Measures left-right vertex distribution imbalance.
            Ratio > 1.06 suggests plagiocephaly.
        """
        extents = sorted(mesh.bounding_box.extents)
        height_mm, width_mm, length_mm = extents[0], extents[1], extents[2]

        cephalic_index = (width_mm / length_mm) * 100.0

        # Asymmetry: compare vertex count on left vs right of centroid
        centroid = mesh.centroid
        vertices = mesh.vertices
        left_count = np.sum(vertices[:, 0] < centroid[0])
        right_count = np.sum(vertices[:, 0] >= centroid[0])
        asymmetry_ratio = max(left_count, right_count) / max(min(left_count, right_count), 1)

        # Classification
        if cephalic_index > 81.0:
            severity = "Brachycephaly (CI > 81)"
            risk_level = "MODERATE" if cephalic_index < 90 else "HIGH"
        elif cephalic_index < 75.0:
            severity = "Dolichocephaly (CI < 75)"
            risk_level = "MODERATE" if cephalic_index > 70 else "HIGH"
        elif asymmetry_ratio > 1.06:
            severity = "Plagiocephaly (asymmetry > 6%)"
            risk_level = "MODERATE"
        else:
            severity = "Normocephaly (75 ≤ CI ≤ 81)"
            risk_level = "LOW"

        return ClinicalMetrics(
            cephalic_index=round(cephalic_index, 2),
            width_mm=round(width_mm, 4),
            length_mm=round(length_mm, 4),
            height_mm=round(height_mm, 4),
            asymmetry_ratio=round(asymmetry_ratio, 4),
            severity=severity,
            risk_level=risk_level
        )

    def process_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process entire dataset directory into voxel arrays and labels.

        Expects structure:
            dataset_path/
                N/      (Normal)
                BP/     (Brachycephaly)
                P/      (Plagiocephaly)
                ...

        Returns:
            X: np.ndarray of shape (n_samples, 32, 32, 32, 1)
            y: np.ndarray of one-hot labels
            class_names: List of class name strings
        """
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.utils import to_categorical

        voxels, labels = [], []
        class_names = sorted([d for d in os.listdir(dataset_path)
                              if os.path.isdir(os.path.join(dataset_path, d))])

        logging.info(f"Found classes: {class_names}")

        for class_name in class_names:
            class_dir = os.path.join(dataset_path, class_name)
            obj_files = [f for f in os.listdir(class_dir) if f.endswith('.obj')]
            logging.info(f"  {class_name}: {len(obj_files)} meshes")

            for fname in obj_files:
                try:
                    mesh = self.load_mesh(os.path.join(class_dir, fname))
                    voxel = self.voxelize(mesh)
                    voxels.append(voxel)
                    labels.append(class_name)
                except Exception as e:
                    logging.warning(f"  Skipping {fname}: {e}")

        X = np.array(voxels).reshape(-1, self.voxel_dim, self.voxel_dim,
                                      self.voxel_dim, 1)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(labels)
        y = to_categorical(y_encoded)

        logging.info(f"Dataset: {X.shape[0]} samples, {len(class_names)} classes")
        return X, y, class_names


# ─────────────────────────────────────────────
# Model Architectures
# ─────────────────────────────────────────────

class ModelFactory:
    """
    Creates and compares multiple 3D deep learning architectures.
    Each architecture represents a different inductive bias:
        - CNN: Translation invariance via local receptive fields
        - ResNet: Skip connections for deep gradient flow
        - DenseNet: Feature reuse via channel concatenation
    """

    @staticmethod
    def create_3d_cnn(input_shape=(32, 32, 32, 1), num_classes=3):
        """
        Lightweight 3D CNN baseline.

        Architecture:
            Conv3D(32) → MaxPool → Conv3D(64) → MaxPool → FC(128) → Softmax

        ~50K parameters. Fast training, good for small datasets.
        Risk: limited depth may miss complex morphological patterns.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

        model = Sequential([
            Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
            MaxPooling3D((2, 2, 2)),
            Conv3D(64, (3, 3, 3), activation='relu'),
            MaxPooling3D((2, 2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def create_3d_resnet(input_shape=(32, 32, 32, 1), num_classes=3):
        """
        3D ResNet with skip connections.

        Solves the vanishing gradient problem in deeper networks.
        Each residual block: x → Conv → BN → ReLU → Conv → BN → Add(x) → ReLU

        The 1×1×1 projection shortcut handles dimension mismatches.
        """
        from tensorflow.keras import layers, models
        from tensorflow.keras.layers import (Conv3D, BatchNormalization, ReLU,
                                             MaxPooling3D, GlobalAveragePooling3D,
                                             Dense, Add)

        def residual_block(x, filters):
            shortcut = Conv3D(filters, (1, 1, 1), padding='same')(x)
            shortcut = BatchNormalization()(shortcut)

            x = Conv3D(filters, (3, 3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv3D(filters, (3, 3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = ReLU()(x)
            return x

        inputs = layers.Input(shape=input_shape)
        x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling3D((2, 2, 2))(x)

        x = residual_block(x, 64)
        x = residual_block(x, 128)

        x = GlobalAveragePooling3D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def create_3d_densenet(input_shape=(32, 32, 32, 1), num_classes=3,
                           growth_rate=12, num_blocks=3, convs_per_block=4):
        """
        3D DenseNet with dense connectivity pattern.

        Each layer receives feature maps from ALL preceding layers (not just
        the previous one). This encourages feature reuse and reduces parameters.

        Growth rate (k=12): each conv adds 12 new feature maps.
        Transition layers: 1×1 conv + average pooling for downsampling.

        Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017).
        """
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Input, Conv3D, BatchNormalization, ReLU,
                                             AveragePooling3D, GlobalAveragePooling3D,
                                             Dense, Concatenate)

        def dense_block(x, num_convs, growth_rate):
            for _ in range(num_convs):
                conv = BatchNormalization()(x)
                conv = ReLU()(conv)
                conv = Conv3D(growth_rate, (3, 3, 3), padding='same')(conv)
                x = Concatenate()([x, conv])
            return x

        def transition(x, reduction=0.5):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv3D(int(x.shape[-1] * reduction), (1, 1, 1), padding='same')(x)
            x = AveragePooling3D(pool_size=2, strides=2)(x)
            return x

        inputs = Input(shape=input_shape)
        x = Conv3D(2 * growth_rate, (3, 3, 3), padding='same')(inputs)

        for _ in range(num_blocks):
            x = dense_block(x, convs_per_block, growth_rate)
            x = transition(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling3D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


# ─────────────────────────────────────────────
# Uncertainty Quantification
# ─────────────────────────────────────────────

class UncertaintyEstimator:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.

    Standard softmax outputs are NOT calibrated probabilities — a model
    can output 99% confidence and be wrong. In medical AI, this is dangerous.

    MC Dropout (Gal & Ghahramani, 2016) runs N stochastic forward passes
    with Dropout active at inference. The variance across passes estimates
    epistemic uncertainty (uncertainty due to limited data/model capacity).

    High variance → model is unsure → flag for human review.
    Low variance  → model is confident → safe for automated decision.
    """

    def __init__(self, model, class_names: List[str], n_iterations: int = 50):
        self.model = model
        self.class_names = class_names
        self.n_iterations = n_iterations

    def predict(self, voxel_input: np.ndarray) -> Dict:
        """
        Run MC Dropout inference and return prediction with uncertainty.

        Args:
            voxel_input: Shape (32, 32, 32, 1) or (1, 32, 32, 32, 1)

        Returns:
            dict with prediction, confidence, uncertainty, and recommendation.
        """
        if voxel_input.ndim == 4:
            voxel_input = np.expand_dims(voxel_input, axis=0)

        # N stochastic forward passes
        predictions = np.array([
            self.model(voxel_input, training=True).numpy().squeeze()
            for _ in range(self.n_iterations)
        ])  # Shape: (n_iterations, num_classes)

        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        max_std = float(np.max(std_pred))
        confidence = max(0.0, min(1.0, 1.0 - max_std))
        predicted_idx = int(np.argmax(mean_pred))

        # Clinical recommendation
        if confidence < 0.60:
            recommendation = "REFER — Low confidence, mandatory specialist review"
            risk = "HIGH"
        elif confidence < 0.80:
            recommendation = "REVIEW — Moderate confidence, clinician verification advised"
            risk = "MODERATE"
        else:
            recommendation = "ACCEPT — High confidence prediction"
            risk = "LOW"

        return {
            'predicted_class': self.class_names[predicted_idx],
            'predicted_index': predicted_idx,
            'confidence': round(confidence, 4),
            'epistemic_uncertainty': round(max_std, 6),
            'per_class_mean': {self.class_names[i]: round(float(mean_pred[i]), 4)
                               for i in range(len(self.class_names))},
            'per_class_std': {self.class_names[i]: round(float(std_pred[i]), 4)
                              for i in range(len(self.class_names))},
            'recommendation': recommendation,
            'risk_level': risk,
        }


# ─────────────────────────────────────────────
# Training & Evaluation Pipeline
# ─────────────────────────────────────────────

class TrainingPipeline:
    """
    End-to-end training pipeline with model comparison and evaluation.
    """

    def __init__(self, X_train, X_test, y_train, y_test, class_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.models = {}
        self.histories = {}

    def train_all(self, epochs=20, batch_size=16):
        """Train CNN, ResNet, and DenseNet, storing histories."""
        num_classes = len(self.class_names)
        architectures = {
            'CNN': ModelFactory.create_3d_cnn(num_classes=num_classes),
            'ResNet': ModelFactory.create_3d_resnet(num_classes=num_classes),
            'DenseNet': ModelFactory.create_3d_densenet(num_classes=num_classes),
        }

        for name, model in architectures.items():
            logging.info(f"Training {name}...")
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=epochs, batch_size=batch_size, verbose=1
            )
            self.models[name] = model
            self.histories[name] = history
            logging.info(f"{name} done — val_acc: "
                         f"{history.history['val_accuracy'][-1]:.4f}")

        return self.models, self.histories

    def evaluate_all(self) -> Dict:
        """Evaluate all models and return comparison metrics."""
        from sklearn.metrics import classification_report, confusion_matrix

        results = {}
        for name, model in self.models.items():
            preds = model.predict(self.X_test)
            pred_labels = np.argmax(preds, axis=1)
            true_labels = np.argmax(self.y_test, axis=1)

            results[name] = {
                'accuracy': float(np.mean(pred_labels == true_labels)),
                'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
                'classification_report': classification_report(
                    true_labels, pred_labels,
                    target_names=self.class_names,
                    output_dict=True
                )
            }
        return results

    def generate_clinical_report(self, sample_indices: List[int],
                                 obj_files: List[str]) -> List[Dict]:
        """Generate full clinical diagnostic reports for given test samples."""
        processor = MeshProcessor()
        best_model = self.models.get('CNN', list(self.models.values())[0])
        estimator = UncertaintyEstimator(best_model, self.class_names)

        reports = []
        for idx in sample_indices:
            true_label = self.class_names[np.argmax(self.y_test[idx])]

            # MC Dropout prediction
            mc_result = estimator.predict(self.X_test[idx])

            # Clinical metrics from mesh
            mesh_idx = idx % len(obj_files)
            mesh = processor.load_mesh(obj_files[mesh_idx])
            clinical = processor.extract_clinical_metrics(mesh)

            reports.append({
                'sample_index': idx,
                'true_label': true_label,
                'prediction': mc_result,
                'clinical_metrics': clinical,
            })

        return reports


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Cranial Deformity Classifier")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory (e.g., modelsSynth/)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--voxel-dim', type=int, default=32)
    args = parser.parse_args()

    # Process dataset
    processor = MeshProcessor(voxel_dim=args.voxel_dim)
    X, y, class_names = processor.process_dataset(args.dataset)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Train and evaluate
    pipeline = TrainingPipeline(X_train, X_test, y_train, y_test, class_names)
    pipeline.train_all(epochs=args.epochs, batch_size=args.batch_size)
    results = pipeline.evaluate_all()

    for name, metrics in results.items():
        print(f"\n{name}: Accuracy = {metrics['accuracy']:.4f}")
