import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core.tri_camera_dataset import TriCameraDataset


class TriCameraDatasetTests(unittest.TestCase):
    def test_crop_resize_and_intrinsics_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = np.zeros((8, 10, 3), dtype=np.uint8)
            texture = np.zeros((6, 12, 3), dtype=np.uint8)
            disparity = np.full((8, 10), 4.0, dtype=np.float32)
            mono = np.linspace(0, 1, 72, dtype=np.float32).reshape(6, 12)
            Image.fromarray(left).save(root / "left.png")
            Image.fromarray(left).save(root / "right.png")
            Image.fromarray(texture).save(root / "texture.png")
            np.save(root / "disparity.npy", disparity)
            np.save(root / "mono.npy", mono)
            (root / "calibration.json").write_text(
                json.dumps(
                    {
                        "K_left": [[100, 0, 5], [0, 100, 4], [0, 0, 1]],
                        "K_texture": [[120, 0, 6], [0, 120, 3], [0, 0, 1]],
                        "baseline": 0.05,
                        "length_unit": "m",
                        "left_image_size": [10, 8],
                        "texture_image_size": [12, 6],
                    }
                ),
                encoding="utf-8",
            )
            (root / "rt.json").write_text(
                json.dumps(
                    {
                        "T": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        "direction": "left_to_texture",
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "left": "left.png",
                        "right": "right.png",
                        "texture": "texture.png",
                        "disparity": "disparity.npy",
                        "mono_depth": "mono.npy",
                        "calibration": "calibration.json",
                        "rt": "rt.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = TriCameraDataset(
                str(root / "manifest.jsonl"),
                crop_size=(4, 6),
                texture_size=(3, 6),
                random_crop=False,
            )
            sample = dataset[0]
            self.assertEqual(sample["left"].shape, (3, 4, 6))
            self.assertEqual(sample["texture"].shape, (3, 3, 6))
            self.assertEqual(sample["mono_depth"].shape, (1, 3, 6))
            # Centre crop removes x=2 and y=2 from the original principal point.
            self.assertAlmostEqual(float(sample["calibration"]["K_left"][0, 2]), 3.0)
            self.assertAlmostEqual(float(sample["calibration"]["K_left"][1, 2]), 2.0)
            # Texture resize is exactly 1/2 in both dimensions.
            self.assertAlmostEqual(float(sample["calibration"]["K_texture"][0, 0]), 60.0)
            self.assertTrue(bool(sample["valid"].all()))


if __name__ == "__main__":
    unittest.main()
