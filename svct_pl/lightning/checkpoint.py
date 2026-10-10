"""Compatibility helpers for checkpoints produced by the legacy trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def _strip_prefix(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def load_legacy_g_render_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load only the legacy ``G_render`` weights into a new SVCT model.

    Optimizer and scheduler state from the hand-written trainer is deliberately
    ignored: Lightning checkpoints should be used when exact training resume is
    required. ``module`` may be either an ``OfficialSVCTModule`` or its model.
    """

    payload = (
        torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a mapping")

    state = payload.get("G_render", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint's G_render/state_dict entry must be a mapping")

    target = getattr(module, "model", module)
    # Accept weights saved from Lightning, DDP, or a directly wrapped model.
    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    return target.load_state_dict(state, strict=strict)
