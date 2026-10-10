"""CUDA memory reporting for the official Lightning trainer."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class GPUMemoryCallback(Callback):
    """Report independent training and full-reconstruction CUDA peaks.

    Validation, test, and predict all use the same ``eval`` metric namespace
    because they execute the same full-volume reconstruction path.  When
    validation runs inside ``fit``, the training peak is captured before the
    allocator counters are reset for evaluation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._train_peak: tuple[float, float] | None = None
        self._train_active = False

    @staticmethod
    def _reset() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _read(module: pl.LightningModule) -> tuple[float, float] | None:
        if not torch.cuda.is_available() or module.device.type != "cuda":
            return None
        gib = 1024.0**3
        return (
            torch.cuda.max_memory_allocated(module.device) / gib,
            torch.cuda.max_memory_reserved(module.device) / gib,
        )

    @staticmethod
    def _log(
        trainer: pl.Trainer,
        module: pl.LightningModule,
        stage: str,
        peak: tuple[float, float] | None = None,
    ) -> None:
        peak = peak if peak is not None else GPUMemoryCallback._read(module)
        if peak is None or trainer.logger is None or not trainer.is_global_zero:
            return
        trainer.logger.log_metrics(
            {
                f"memory/{stage}_peak_allocated_gib": peak[0],
                f"memory/{stage}_peak_reserved_gib": peak[1],
            },
            step=trainer.global_step,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_peak = None
        self._train_active = True
        self._reset()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._train_peak is None:
            self._train_peak = self._read(pl_module)
        self._log(trainer, pl_module, "train", self._train_peak)
        self._train_active = False

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._train_active:
            self._train_peak = self._read(pl_module)
            self._train_active = False
        self._reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            self._log(trainer, pl_module, "eval")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module, "eval")

    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset()

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module, "eval")
