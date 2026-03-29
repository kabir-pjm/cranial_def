"""
3D Grad-CAM — Model Explainability for Cranial Classification
==============================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
adapted for 3D volumetric CNNs.

Why this matters:
    In medical AI, a prediction alone is not enough. Clinicians need to
    know WHERE the model is looking. Grad-CAM highlights which regions
    of the skull influenced the classification decision.

    This is a regulatory requirement for SaMD (Software as a Medical Device)
    under FDA/EU MDR guidelines — the model must be interpretable.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D CNNs.

    Computes the gradient of the target class score with respect to
    the feature maps of a specified convolutional layer. The gradients
    are global-average-pooled to get importance weights, then used to
    create a weighted combination of feature maps → the heatmap.

    High activation regions = "the model looked here to make its decision."
    """

    def __init__(self, model, layer_name: Optional[str] = None):
        """
        Args:
            model: Trained Keras model.
            layer_name: Name of the Conv3D layer to visualize.
                        If None, uses the last Conv3D layer.
        """
        self.model = model

        if layer_name is None:
            # Find last Conv3D layer automatically
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D):
                    layer_name = layer.name
                    break

        if layer_name is None:
            raise ValueError("No Conv3D layer found in model")

        self.layer_name = layer_name
        self.grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, voxel_input: np.ndarray,
                        target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate 3D Grad-CAM heatmap.

        Args:
            voxel_input: Shape (1, 32, 32, 32, 1)
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            heatmap: np.ndarray of shape (D, H, W), values in [0, 1].
                     High values indicate important regions.
        """
        if voxel_input.ndim == 4:
            voxel_input = np.expand_dims(voxel_input, axis=0)

        voxel_tensor = tf.cast(voxel_input, tf.float32)

        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(voxel_tensor)
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            target_score = predictions[:, target_class]

        # Gradient of target class w.r.t. conv layer output
        grads = tape.gradient(target_score, conv_output)

        # Global average pooling of gradients → importance weights
        weights = tf.reduce_mean(grads, axis=(1, 2, 3))  # Shape: (1, num_filters)

        # Weighted combination of feature maps
        conv_output = conv_output[0]  # Shape: (D, H, W, num_filters)
        heatmap = tf.reduce_sum(
            conv_output * weights[0], axis=-1
        )  # Shape: (D, H, W)

        # ReLU — only positive contributions
        heatmap = tf.nn.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    def get_attention_slices(self, voxel_input: np.ndarray,
                             target_class: Optional[int] = None,
                             num_slices: int = 5) -> list:
        """
        Get 2D slices from the 3D heatmap for visualization.

        Returns evenly spaced axial slices with both the original
        voxel data and the Grad-CAM overlay.
        """
        from scipy.ndimage import zoom

        heatmap = self.compute_heatmap(voxel_input, target_class)

        # Resize heatmap to match input dimensions
        if heatmap.shape != (32, 32, 32):
            scale = np.array([32, 32, 32]) / np.array(heatmap.shape)
            heatmap = zoom(heatmap, scale, order=1)

        voxel = voxel_input.squeeze()
        slice_indices = np.linspace(4, 27, num_slices, dtype=int)

        slices = []
        for idx in slice_indices:
            slices.append({
                'slice_index': int(idx),
                'voxel_slice': voxel[idx, :, :],
                'heatmap_slice': heatmap[idx, :, :],
                'combined': voxel[idx, :, :] * 0.5 + heatmap[idx, :, :] * 0.5
            })

        return slices

    def get_attention_summary(self, voxel_input: np.ndarray,
                              target_class: Optional[int] = None) -> dict:
        """
        Summarize where the model is focusing.

        Returns attention distribution across anatomical regions
        (anterior/posterior, left/right, superior/inferior).
        """
        heatmap = self.compute_heatmap(voxel_input, target_class)

        d, h, w = heatmap.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

        attention = {
            'anterior': float(np.mean(heatmap[:, :, :mid_w])),
            'posterior': float(np.mean(heatmap[:, :, mid_w:])),
            'left': float(np.mean(heatmap[:, :mid_h, :])),
            'right': float(np.mean(heatmap[:, mid_h:, :])),
            'superior': float(np.mean(heatmap[:mid_d, :, :])),
            'inferior': float(np.mean(heatmap[mid_d:, :, :])),
            'peak_location': tuple(int(x) for x in np.unravel_index(
                np.argmax(heatmap), heatmap.shape
            )),
            'total_activation': float(np.sum(heatmap)),
        }

        return attention
