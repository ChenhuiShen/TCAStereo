import json
from pathlib import Path
import tempfile
import unittest

import torch

from core.tca.calibration import load_calibration
from core.tca.depth_anything import DepthAnythingV2Prior
from core.tca.fiu import FusionIterationUnit
from core.tca.gam import GlobalAlignmentModule
from core.tca.gem import GeometricEnhancementModule


class CalibrationTests(unittest.TestCase):
    def test_external_reverse_rt_is_converted_to_left_to_texture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            rt_path = root / "rt.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "K_left": [[100, 0, 20], [0, 100, 15], [0, 0, 1]],
                        "K_texture": [[120, 0, 25], [0, 120, 18], [0, 0, 1]],
                        "baseline": 50,
                        "length_unit": "mm",
                        "left_image_size": [40, 30],
                        "texture_image_size": [50, 36],
                    }
                ),
                encoding="utf-8",
            )
            rt_path.write_text(
                json.dumps(
                    {
                        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "t": [-100, 0, 0],
                        "translation_unit": "mm",
                        "direction": "texture_to_left",
                    }
                ),
                encoding="utf-8",
            )
            calibration = load_calibration(str(calibration_path), str(rt_path))
            self.assertAlmostEqual(calibration.baseline, 0.05)
            self.assertAlmostEqual(float(calibration.T_left_to_texture[0, 3]), 0.1)
            runtime = calibration.runtime_dict(
                left_image_size=(80, 60),
                texture_image_size=(100, 72),
                left_padding=(2, 2, 3, 3),
            )
            self.assertAlmostEqual(float(runtime["K_left"][0, 0, 0]), 200.0)
            self.assertAlmostEqual(float(runtime["K_left"][0, 0, 2]), 42.0)


class GAMTests(unittest.TestCase):
    def test_identity_cameras_recover_known_scale_and_offset(self):
        height, width = 24, 32
        fx = 40.0
        baseline = 0.1
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        true_depth = 1.0 + xx.float() * 0.01 + yy.float() * 0.02
        disparity = (fx * baseline / true_depth).unsqueeze(0).unsqueeze(0)
        mono_depth = ((true_depth - 0.4) / 1.7).unsqueeze(0).unsqueeze(0)
        intrinsics = torch.tensor(
            [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]]
        ).unsqueeze(0)
        calibration = {
            "K_left": intrinsics,
            "K_texture": intrinsics.clone(),
            "T_left_to_texture": torch.eye(4).unsqueeze(0),
            "baseline": torch.tensor([baseline]),
            "disparity_offset": torch.tensor([0.0]),
            "disparity_sign": torch.tensor([1.0]),
        }
        gam = GlobalAlignmentModule(
            ransac_iterations=64,
            ransac_threshold=1e-3,
            min_points=10,
            hole_fill_iterations=0,
        )
        result = gam(disparity, mono_depth, calibration, (height, width))
        self.assertAlmostEqual(float(result.alpha[0]), 1.7, places=3)
        self.assertAlmostEqual(float(result.beta[0]), 0.4, places=3)
        self.assertTrue(bool(result.valid_left.all()))
        error = (result.aligned_depth_left[0, 0] - true_depth).abs().max()
        self.assertLess(float(error), 1e-3)


class LearnableModuleTests(unittest.TestCase):
    def test_gem_preserves_shapes_and_has_frequency_gradients(self):
        module = GeometricEnhancementModule(channels=(8, 16), num_kernels=4)
        features = [
            torch.randn(2, 8, 12, 16, requires_grad=True),
            torch.randn(2, 16, 6, 8, requires_grad=True),
        ]
        output = module(features)
        self.assertEqual([item.shape for item in output], [item.shape for item in features])
        sum(item.square().mean() for item in output).backward()
        self.assertIsNotNone(module.blocks[0].frequency_weight_real.grad)

    def test_fiu_is_identity_outside_projection_support(self):
        module = FusionIterationUnit(hidden_channels=16)
        depth = torch.rand(1, 1, 8, 10) + 1.0
        disparity = torch.rand(1, 1, 8, 10)
        residual = torch.randn(1, 1, 8, 10, requires_grad=True)
        valid = torch.zeros(1, 1, 8, 10, dtype=torch.bool)
        valid[:, :, 2:6, 3:8] = True
        fused, attention = module(depth, disparity, residual, valid)
        self.assertTrue(torch.equal(fused[~valid], residual[~valid]))
        self.assertTrue(bool(((attention[valid] >= 0) & (attention[valid] <= 1)).all()))
        fused.mean().backward()
        self.assertIsNotNone(residual.grad)

    def test_depth_anything_adapter_supports_injected_frozen_model(self):
        class DummyDepth(torch.nn.Module):
            def forward(self, image):
                return image.mean(dim=1)

        adapter = DepthAnythingV2Prior(
            encoder="vits", input_size=28, model=DummyDepth()
        )
        image = torch.rand(2, 3, 20, 30) * 255.0
        depth = adapter(image)
        self.assertEqual(depth.shape, (2, 1, 20, 30))
        self.assertFalse(depth.requires_grad)


if __name__ == "__main__":
    unittest.main()
