"""Measure complete official Lightning training steps on a CUDA GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from models.render import angle2vec
from .official import OfficialSVCTModule
from .official_cli import build_parser as training_parser, load_official_config
from .official_data import OfficialCBCTDataModule


class SyntheticCBCTDataset(Dataset):
    """Physical-attenuation fixture for reproducible capacity checks only."""
    def __init__(self, nviews, projection_size, volume_size, clamp_max):
        size = volume_size
        angles = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
        poses = np.stack([angle2vec(angle, 0, [0, 0, 0], 700, 500,
                                   size * 1.4 / projection_size,
                                   size * 1.4 / projection_size) for angle in angles])
        self.sample = {
            "images": torch.rand(nviews, 1, projection_size, projection_size) * 3.6485,
            "poses": torch.tensor(poses, dtype=torch.float32),
            "3Dvolume": torch.rand(size, size, size) * clamp_max,
            "paras": {
                "volume_phy": [float(size)] * 3,
                "volume_origin": [-size / 2] * 3,
                "volume_spacing": [1.0] * 3,
                "volume_resolution": [size] * 3,
            },
            "obj_index": "synthetic",
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def cuda_peaks():
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


class MemoryMeasurement(pl.Callback):
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.start_time = None
        self.input_shapes = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.input_shapes = {
            "images": list(batch["images"].shape),
            "gt_volume": list(batch["3Dvolume"].shape),
        }
        if trainer.global_step == self.warmup_steps and self.start_time is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_time = perf_counter()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--datadir", help="Official-layout real dataset root")
    source.add_argument("--synthetic", action="store_true", help="Capacity check; not a real-data measurement")
    parser.add_argument("--datatype", choices=("dental", "spine", "Walnuts"), default="dental")
    parser.add_argument("--split-path")
    parser.add_argument("--case", help="Fixed real case; defaults to the first training case")
    parser.add_argument("--conf", default=training_parser().get_default("conf"))
    parser.add_argument("--nviews", type=int, default=20)
    parser.add_argument("--train_scale", type=int, default=4)
    parser.add_argument("--projection-size", type=int, default=256, help="Synthetic input size only")
    parser.add_argument("--volume-size", type=int, default=256, help="Synthetic volume size only")
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true"), default="bf16-mixed")
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument("--activation-checkpointing", dest="activation_checkpointing",
                              action="store_true", help="Enable activation checkpointing (default)")
    checkpointing.add_argument("--no-activation-checkpointing", dest="activation_checkpointing",
                              action="store_false", help="Disable activation checkpointing")
    parser.set_defaults(activation_checkpointing=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/memory/result.json")
    return parser


def main(argv=None):
    options = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Memory measurement requires a CUDA GPU; no CPU fallback is used")
    if options.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    if options.warmup_steps < 1 or options.measure_steps < 1:
        raise ValueError("Use at least one warmup and one measured optimizer step")
    if options.projection_size <= 0 or options.volume_size <= 0:
        raise ValueError("Synthetic spatial sizes must be positive")
    if options.train_scale <= 0 or options.nviews <= 0:
        raise ValueError("train_scale and nviews must be positive")
    if options.volume_size % options.train_scale:
        raise ValueError("Synthetic volume-size must be divisible by train_scale")
    pl.seed_everything(options.seed, workers=True)
    settings = training_parser().parse_args([])
    for key in ("conf", "datatype", "split_path", "nviews", "train_scale", "precision",
                "activation_checkpointing", "seed"):
        setattr(settings, key, getattr(options, key))
    settings.datadir = options.datadir or ""
    settings.output_root = str(Path(options.output).parent)
    settings.name = Path(options.output).stem
    settings.epochs = options.warmup_steps + options.measure_steps
    conf = load_official_config(settings)
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(settings))
    if options.synthetic:
        dataset = SyntheticCBCTDataset(options.nviews, options.projection_size,
                                       options.volume_size, module.clamp_max)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)
        data = None
        case = "synthetic"
    else:
        data = OfficialCBCTDataModule(vars(settings))
        data.setup("fit")
        case = options.case or data.datasets["train"].dataset_split[0]
        data.datasets["train"].dataset_split = [case]
        loader = None
    measurement = MemoryMeasurement(options.warmup_steps)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=options.precision,
        max_epochs=settings.epochs, max_steps=settings.epochs,
        limit_val_batches=0, num_sanity_val_steps=0,
        gradient_clip_val=0.0, accumulate_grad_batches=1,
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
        enable_model_summary=False, callbacks=[measurement],
    )
    result = {
        "status": "running", "source": "synthetic" if options.synthetic else "official_nifti",
        "case": case, "benchmark_args": vars(options),
        "conf": conf.as_plain_ordered_dict(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": platform.python_version(), "torch": torch.__version__,
        "pytorch_lightning": pl.__version__, "cuda": torch.version.cuda,
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator_env": {key: os.environ.get(key) for key in
                          ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "measurement_scope": "Full training steps, including backward, Adam and upstream post-update eval PSNR; validation/test/visual disabled",
        "memory_scope": "PyTorch CUDA allocator; CUDA context and other non-PyTorch allocations excluded",
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        trainer.fit(module, datamodule=data, train_dataloaders=loader)
        torch.cuda.synchronize()
        result.update(status="complete", **cuda_peaks(), input_shapes=measurement.input_shapes,
                      optimizer_steps=trainer.global_step,
                      measured_seconds=perf_counter() - measurement.start_time)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error), **cuda_peaks(),
                      input_shapes=measurement.input_shapes, optimizer_steps=trainer.global_step)
        raise
    except Exception as error:
        result.update(status="failed", error=str(error), input_shapes=measurement.input_shapes,
                      optimizer_steps=trainer.global_step)
        raise
    finally:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
