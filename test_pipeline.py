"""
Tests for Cranial Deformity Pipeline
======================================
Covers: mesh processing, clinical metrics, model creation, uncertainty estimation.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMeshProcessor(unittest.TestCase):
    """Test 3D mesh processing and voxelization."""

    def test_voxel_output_shape(self):
        """Voxelized output should be exactly (32, 32, 32)."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor(voxel_dim=32)
        # Create a simple test sphere
        mesh = trimesh.creation.icosphere(radius=1.0)
        voxel = processor.voxelize(mesh)
        self.assertEqual(voxel.shape, (32, 32, 32))

    def test_voxel_is_binary(self):
        """Voxel values should be 0.0 or 1.0."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor()
        mesh = trimesh.creation.box()
        voxel = processor.voxelize(mesh)
        unique_vals = np.unique(voxel)
        self.assertTrue(all(v in [0.0, 1.0] for v in unique_vals))

    def test_voxel_not_empty(self):
        """Voxelized mesh should have non-zero occupancy."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor()
        mesh = trimesh.creation.icosphere(radius=1.0)
        voxel = processor.voxelize(mesh)
        self.assertGreater(np.sum(voxel), 0)


class TestClinicalMetrics(unittest.TestCase):
    """Test cephalic index and clinical classification."""

    def test_sphere_is_brachycephalic(self):
        """A sphere has CI=100 (width==length), should be brachycephaly."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor()
        mesh = trimesh.creation.icosphere(radius=1.0)
        metrics = processor.extract_clinical_metrics(mesh)
        self.assertGreater(metrics.cephalic_index, 81)
        self.assertIn("Brachycephaly", metrics.severity)

    def test_elongated_is_dolichocephalic(self):
        """An elongated box should have CI < 75."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor()
        # Long in one axis: length >> width
        mesh = trimesh.creation.box(extents=[1.0, 0.5, 3.0])
        metrics = processor.extract_clinical_metrics(mesh)
        self.assertLess(metrics.cephalic_index, 75)
        self.assertIn("Dolichocephaly", metrics.severity)

    def test_metrics_fields_present(self):
        """All clinical metric fields should be populated."""
        import trimesh
        from pipeline import MeshProcessor
        processor = MeshProcessor()
        mesh = trimesh.creation.box()
        metrics = processor.extract_clinical_metrics(mesh)
        self.assertIsNotNone(metrics.cephalic_index)
        self.assertIsNotNone(metrics.asymmetry_ratio)
        self.assertIsNotNone(metrics.risk_level)


class TestModelFactory(unittest.TestCase):
    """Test model creation and output shapes."""

    def test_cnn_output_shape(self):
        """CNN should output (batch_size, num_classes)."""
        from pipeline import ModelFactory
        model = ModelFactory.create_3d_cnn(num_classes=4)
        dummy = np.random.rand(2, 32, 32, 32, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        self.assertEqual(output.shape, (2, 4))

    def test_cnn_output_is_probability(self):
        """Softmax output should sum to ~1.0 per sample."""
        from pipeline import ModelFactory
        model = ModelFactory.create_3d_cnn(num_classes=3)
        dummy = np.random.rand(1, 32, 32, 32, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        self.assertAlmostEqual(float(np.sum(output)), 1.0, places=5)

    def test_resnet_creates(self):
        """ResNet should compile without errors."""
        from pipeline import ModelFactory
        model = ModelFactory.create_3d_resnet(num_classes=3)
        self.assertIsNotNone(model)

    def test_densenet_creates(self):
        """DenseNet should compile without errors."""
        from pipeline import ModelFactory
        model = ModelFactory.create_3d_densenet(num_classes=3)
        self.assertIsNotNone(model)


class TestUncertaintyEstimator(unittest.TestCase):
    """Test Monte Carlo Dropout uncertainty estimation."""

    def test_mc_output_fields(self):
        """MC Dropout should return all required fields."""
        from pipeline import ModelFactory, UncertaintyEstimator
        model = ModelFactory.create_3d_cnn(num_classes=3)
        estimator = UncertaintyEstimator(model, ['N', 'BP', 'P'], n_iterations=10)
        dummy = np.random.rand(32, 32, 32, 1).astype(np.float32)
        result = estimator.predict(dummy)

        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('epistemic_uncertainty', result)
        self.assertIn('recommendation', result)

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        from pipeline import ModelFactory, UncertaintyEstimator
        model = ModelFactory.create_3d_cnn(num_classes=3)
        estimator = UncertaintyEstimator(model, ['N', 'BP', 'P'], n_iterations=10)
        dummy = np.random.rand(32, 32, 32, 1).astype(np.float32)
        result = estimator.predict(dummy)

        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
