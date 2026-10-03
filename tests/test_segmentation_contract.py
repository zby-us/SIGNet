"""Regression tests for segmentation naming and checkpoint compatibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from segment.dataset import LEGACY_TAGS, LEGACY_TO_STRUCTURE, STRUCTURES
from segment.model import UnetVMamba, main_logits
from segment.predict import (
    load_model,
    parse_thresholds,
    resolve_structure_checkpoints,
    validate_threshold,
)


class SegmentationContractTests(unittest.TestCase):
    def test_historical_mapping_matches_paper_terminology(self) -> None:
        self.assertEqual(
            LEGACY_TO_STRUCTURE,
            {"LL": "LI", "LR": "LS", "RL": "RS", "RR": "RI"},
        )
        self.assertEqual(STRUCTURES["LI"], ("left", "left", "left"))
        self.assertEqual(STRUCTURES["LS"], ("left", "left", "right"))
        self.assertEqual(STRUCTURES["RS"], ("right", "right", "left"))
        self.assertEqual(STRUCTURES["RI"], ("right", "right", "right"))

    def test_model_retains_checkpoint_auxiliary_heads(self) -> None:
        model = UnetVMamba(base_channels=2, drop_path_rate=0.0).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 64, 32))
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        self.assertEqual(tuple(main_logits(output).shape), (1, 2, 64, 32))
        self.assertTrue(all(tuple(item.shape) == (1, 2, 64, 32) for item in output))

    def test_thresholds_accept_checkpoint_grid_and_historical_names(self) -> None:
        parsed = parse_thresholds("LL=0.01,LR=0.37,RR=0.55,RL=0.99")
        self.assertEqual(parsed, {"LI": 0.01, "LS": 0.37, "RI": 0.55, "RS": 0.99})
        with self.assertRaises(ValueError):
            validate_threshold(1.0, "LI")
        with self.assertRaises(ValueError):
            parse_thresholds("LI=0.2,LL=0.3,LS=0.4,RI=0.5,RS=0.6")

    def test_original_checkpoint_names_and_tags_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, structure in enumerate(("LI", "LS", "RI", "RS"), start=1):
                torch.manual_seed(index)
                model = UnetVMamba(base_channels=2, drop_path_rate=0.0)
                legacy = LEGACY_TAGS[structure]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "tag": legacy,
                        "side": STRUCTURES[structure][0],
                        "threshold": index / 100,
                        "model_config": {
                            "in_channels": 1,
                            "num_classes": 2,
                            "base_channels": 2,
                            "drop_path_rate": 0.0,
                        },
                    },
                    root / f"unet_vmamba_best30_{legacy}.pt",
                )

            paths, _ = resolve_structure_checkpoints(root)
            self.assertEqual(paths["LI"].name, "unet_vmamba_best30_LL.pt")
            self.assertEqual(paths["LS"].name, "unet_vmamba_best30_LR.pt")
            self.assertEqual(paths["RS"].name, "unet_vmamba_best30_RL.pt")
            self.assertEqual(paths["RI"].name, "unet_vmamba_best30_RR.pt")
            for structure, path in paths.items():
                _, threshold, declared, side = load_model(path, torch.device("cpu"))
                self.assertEqual(declared, structure)
                self.assertEqual(side, STRUCTURES[structure][0])
                self.assertGreater(threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
