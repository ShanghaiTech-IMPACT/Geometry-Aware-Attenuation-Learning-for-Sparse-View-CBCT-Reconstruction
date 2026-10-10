"""CLI retaining the upstream HOCON configuration and argument names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pyhocon import ConfigFactory
import torch

from .official_callbacks import GPUMemoryCallback
from .checkpoint import load_legacy_g_render_checkpoint
from .official import OfficialSVCTModule
from .official_data import OfficialCBCTDataModule


class ValidationCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        # Empty/skipped validation epochs must not select a best checkpoint
        # using an absent metric or a metric retained from a previous epoch.
        epoch = trainer.current_epoch
        interval = pl_module.conf.get_int("train.print.val_interval")
        if (epoch > 0 and epoch % interval == 0) or epoch == trainer.max_epochs - 1:
            super().on_validation_end(trainer, pl_module)


class HistoryCheckpoint(ModelCheckpoint):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        interval = pl_module.conf.get_int("train.print.save_interval")
        if epoch % interval == 0 or epoch == trainer.max_epochs - 1:
            super().on_train_epoch_end(trainer, pl_module)


def build_parser():
    parser = argparse.ArgumentParser(description="Official GAAL with PyTorch Lightning training")
    parser.add_argument("--conf", "-c", default=str(
        Path(__file__).resolve().parents[2] / "conf" / "train.conf"))
    parser.add_argument("--datadir", "-D", default="dataset/dental/syn_data")
    parser.add_argument("--datatype", choices=("dental", "spine", "Walnuts"), default="dental")
    parser.add_argument("--split-path", help="JSON with train/val/test/visual case lists; defaults to official splits")
    parser.add_argument("--name", "-n", default="GAAL_lightning")
    parser.add_argument("--mode", choices=("train", "validate", "test", "visual"), default="train")
    parser.add_argument("--batch_size", "-B", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=360)
    parser.add_argument("--nviews", "-V", type=int, default=20)
    parser.add_argument("--angle_sampling", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--expnorm", action="store_false", help="Disable exponential normalization (upstream flag semantics)")
    parser.add_argument("--no-expnorm", dest="expnorm", action="store_false")
    parser.add_argument("--train_scale", type=int, default=4)
    parser.add_argument("--eval_scale", type=int, default=-1)
    parser.add_argument("--fusion", default="ada")
    parser.add_argument("--gd1_lambda", type=float, default=1.0)
    parser.add_argument("--mse_lambda_2d", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true"), default="bf16-mixed")
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument("--activation-checkpointing", dest="activation_checkpointing",
                              action="store_true", help="Checkpoint encoder/decoder blocks (default)")
    checkpointing.add_argument("--no-activation-checkpointing", dest="activation_checkpointing",
                              action="store_false", help="Disable activation checkpointing")
    parser.set_defaults(activation_checkpointing=True)
    parser.add_argument("--accelerator", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/lightning")
    parser.add_argument("--checkpoint", help="Lightning checkpoint: restores full training state in train mode")
    parser.add_argument("--legacy-checkpoint", help="Official G_render checkpoint: imports model weights only")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional short training run; -1 uses --epochs")
    parser.add_argument("--limit-val-batches", type=float, default=1.0,
                        help="Set to 0 for a training-only smoke or memory run")
    return parser


def load_official_config(args):
    if args.nviews <= 0 or args.train_scale <= 0 or args.epochs <= 0:
        raise ValueError("nviews, train_scale and epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    conf = ConfigFactory.parse_file(args.conf)
    conf.put("model.SRGAN.generator.scale", args.train_scale)
    conf.put("model.fusion", args.fusion)
    conf.put("train.G_loss.gd1_lambda", args.gd1_lambda)
    conf.put("train.G_loss.mse_lambda_2d", args.mse_lambda_2d)
    if args.eval_scale == -1:
        args.eval_scale = args.train_scale
    if args.eval_scale <= 0:
        raise ValueError("eval_scale must be positive")
    return conf


def resolve_precision(args):
    if args.precision == "32-true":
        return args.precision
    if args.accelerator == "cpu" or not torch.cuda.is_available():
        print("BF16 CUDA is unavailable; using 32-true for this run.")
        return "32-true"
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision=32-true")
    return args.precision


def build_trainer(args, conf):
    output = Path(args.output_root) / args.name
    checkpoints = output / "checkpoints"
    callbacks = [
        ValidationCheckpoint(
            dirpath=checkpoints, filename="best-{epoch:04d}",
            monitor="val/psnr_3d_clamp", mode="max", save_top_k=1,
            save_on_train_epoch_end=False, auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoints, filename="epoch-{epoch:04d}", save_top_k=0,
            save_last=True, every_n_epochs=1, save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
        HistoryCheckpoint(
            dirpath=checkpoints / "history", filename="epoch-{epoch:04d}",
            save_top_k=-1, every_n_epochs=1,
            save_on_train_epoch_end=True, auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        GPUMemoryCallback(),
    ]
    return pl.Trainer(
        accelerator=args.accelerator, devices=args.devices,
        precision=resolve_precision(args), max_epochs=args.epochs, max_steps=args.max_steps,
        # Module gates val/test/visual by the original zero-based intervals.
        check_val_every_n_epoch=1, limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0, log_every_n_steps=1,
        gradient_clip_val=0.0, accumulate_grad_batches=1,
        logger=TensorBoardLogger(str(output), name="lightning_logs"), callbacks=callbacks,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.checkpoint and args.legacy_checkpoint:
        raise ValueError("Choose either --checkpoint or --legacy-checkpoint")
    if args.mode != "train" and not (args.checkpoint or args.legacy_checkpoint):
        raise ValueError("Evaluation requires --checkpoint or --legacy-checkpoint")
    pl.seed_everything(args.seed, workers=True)
    conf = load_official_config(args)
    settings = dict(vars(args))
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), settings)
    if args.legacy_checkpoint:
        load_legacy_g_render_checkpoint(module, args.legacy_checkpoint, strict=True)
    if args.checkpoint and args.mode != "train":
        module = OfficialSVCTModule.load_from_checkpoint(
            args.checkpoint, conf=conf.as_plain_ordered_dict(), args=settings, map_location="cpu")
    data = OfficialCBCTDataModule(settings)
    trainer = build_trainer(args, conf)
    if args.mode == "train":
        return trainer.fit(module, datamodule=data, ckpt_path=args.checkpoint)
    if args.mode == "validate":
        return trainer.validate(module, datamodule=data)
    if args.mode == "test":
        return trainer.test(module, datamodule=data)
    result = trainer.predict(module, datamodule=data)
    if trainer.is_global_zero:
        output = Path(args.output_root) / args.name
        output.mkdir(parents=True, exist_ok=True)
        (output / "visual_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
