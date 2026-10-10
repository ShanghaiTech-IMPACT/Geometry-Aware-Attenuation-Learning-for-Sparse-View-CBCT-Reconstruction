"""CPU data loading using the official CBCTDataset.__getitem__ unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.Dataset import CBCTDataset


class LightningCBCTDataset(CBCTDataset):
    def __init__(self, args, stage="train"):
        # The original constructor hard-codes a cwd-relative split path and
        # allocates data on args.device. Only the initialization changes here;
        # all NIfTI reading, view selection and stored vecs remain upstream.
        self.args = args
        self.stage = stage
        self.angle_sampling = args.angle_sampling
        self.device = "cpu"
        self.datadir = args.datadir
        split_path = args.split_path or (
            Path(__file__).resolve().parents[2]
            / "data" / "dataset_split" / f"{args.datatype}_split.json"
        )
        with Path(split_path).open(encoding="utf-8") as handle:
            self.dataset_split = json.load(handle)[stage]


class OfficialCBCTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = SimpleNamespace(**dict(args))
        self.args.device = "cpu"
        if self.args.batch_size != 1:
            raise ValueError("The official reconstruction model requires batch_size=1")
        if self.args.angle_sampling == "random" and self.args.num_workers:
            raise ValueError("Use --num-workers=0 for the official on-the-fly random DRR path")
        self.datasets = {}

    def setup(self, stage=None):
        stages = {
            "fit": ("train", "val", "test", "visual"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("visual",),
            None: ("train", "val", "test", "visual"),
        }[stage]
        for name in stages:
            if name not in self.datasets:
                self.datasets[name] = LightningCBCTDataset(self.args, name)

    def _loader(self, stage, shuffle):
        return DataLoader(
            self.datasets[stage],
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        # Preserve the official val/test/visual checks during fit. Each keeps
        # its own metric namespace; best checkpoints use only validation.
        if "train" in self.datasets:
            return [self._loader("val", True), self._loader("test", True),
                    self._loader("visual", False)]
        return self._loader("val", True)

    def test_dataloader(self):
        return self._loader("test", False)

    def predict_dataloader(self):
        return self._loader("visual", False)
