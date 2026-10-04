"""Regression tests for segmentation naming and checkpoint compatibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from segment.dataset import STRUCTURES
from segment.model import UnetVMamba, main_logits
from segment.predict import (
    load_model,
    parse_thresholds,
    resolve_structure_checkpoints,
    validate_threshold,
)


class SegmentationContractTests(unittest.TestCase):
    def test_structure_mask_directories(self) -> None:
        # Tuples encode stored-image half, anatomical side and bone.
        # LI: left ilium; LS: left sacrum; RS: right sacrum; RI: right ilium.
        self.assertEqual(STRUCTURES["LI"], ("left", "left", "ilium"))
        self.assertEqual(STRUCTURES["LS"], ("left", "left", "sacrum"))
        self.assertEqual(STRUCTURES["RS"], ("right", "right", "sacrum"))
        self.assertEqual(STRUCTURES["RI"], ("right", "right", "ilium"))

    def test_model_retains_checkpoint_auxiliary_heads(self) -> None:
        model = UnetVMamba(base_channels=2, drop_path_rate=0.0).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 64, 32))
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        self.assertEqual(tuple(main_logits(output).shape), (1, 2, 64, 32))
        self.assertTrue(all(tuple(item.shape) == (1, 2, 64, 32) for item in output))

    def test_thresholds_accept_checkpoint_grid_and_structure_names(self) -> None:
        parsed = parse_thresholds("LI=0.01,LS=0.37,RI=0.55,RS=0.99")
        self.assertEqual(parsed, {"LI": 0.01, "LS": 0.37, "RI": 0.55, "RS": 0.99})
        with self.assertRaises(ValueError):
            validate_threshold(1.0, "LI")
        with self.assertRaises(ValueError):
            parse_thresholds("LI=0.2,LI=0.3,LS=0.4,RI=0.5,RS=0.6")

    def test_anatomical_checkpoint_names_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, structure in enumerate(("LI", "LS", "RI", "RS"), start=1):
                torch.manual_seed(index)
                model = UnetVMamba(base_channels=2, drop_path_rate=0.0)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "structure": structure,
                        "side": STRUCTURES[structure][0],
                        "threshold": index / 100,
                        "model_config": {
                            "in_channels": 1,
                            "num_classes": 2,
                            "base_channels": 2,
                            "drop_path_rate": 0.0,
                        },
                    },
                    root / f"unet_vmamba_best30_{structure}.pt",
                )

            paths, _ = resolve_structure_checkpoints(root)
            self.assertEqual(paths["LI"].name, "unet_vmamba_best30_LI.pt")
            self.assertEqual(paths["LS"].name, "unet_vmamba_best30_LS.pt")
            self.assertEqual(paths["RS"].name, "unet_vmamba_best30_RS.pt")
            self.assertEqual(paths["RI"].name, "unet_vmamba_best30_RI.pt")
            for structure, path in paths.items():
                _, threshold, declared, side = load_model(path, torch.device("cpu"))
                self.assertEqual(declared, structure)
                self.assertEqual(side, STRUCTURES[structure][0])
                self.assertGreater(threshold, 0.0)


    def test_filename_identifies_checkpoint_without_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unet_vmamba_best30_LI.pt"
            model = UnetVMamba(base_channels=2, drop_path_rate=0.0)
            checkpoint = {
                "model": model.state_dict(),
                "side": "left",
                "threshold": 0.95,
                "model_config": {"base_channels": 2, "drop_path_rate": 0.0},
            }
            torch.save(checkpoint, path)
            _, threshold, structure, side = load_model(path, torch.device("cpu"))
            self.assertEqual((threshold, structure, side), (0.95, "LI", "left"))
            checkpoint["structure"] = "LS"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_model(path, torch.device("cpu"))

    def test_unknown_structure_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_thresholds("UNKNOWN=0.5,LS=0.5,RI=0.5,RS=0.5")


if __name__ == "__main__":
    unittest.main()
