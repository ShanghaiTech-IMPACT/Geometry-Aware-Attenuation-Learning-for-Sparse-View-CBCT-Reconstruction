"""Precision and optional recomputation adapters around the unmodified model."""

from __future__ import annotations

from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from models.model import model as OfficialModel
from ._checkpoint import checkpoint_context_fn


def _queryfeature_fp32(encoder, xyz):
    # grid_sample requires matching input/grid dtypes. Keep the original
    # projection equations, coordinate convention and zero padding in FP32;
    # the convolutions, fusion MLP and decoder still use Lightning's AMP.
    features = encoder.latent_list
    try:
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            encoder.latent_list = [feature.float() for feature in features]
            return encoder._queryfeature_without_amp(xyz.float())
    finally:
        encoder.latent_list = features


def _checkpointed_forward(block, *args, **kwargs):
    if block.training and torch.is_grad_enabled():
        return checkpoint(
            block._forward_without_checkpointing,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=lambda: checkpoint_context_fn(block),
            **kwargs,
        )
    return block._forward_without_checkpointing(*args, **kwargs)


def build_official_model(model_conf, *, activation_checkpointing=False):
    """Preserve the official architecture and state-dict keys.

    Adapt only this instance, leaving models/*.py and the original trainer
    untouched. Lightning moves the CPU-initialized model to its device.
    """
    network = OfficialModel(model_conf=model_conf, device="cpu")
    encoder = network.encoder
    encoder._queryfeature_without_amp = encoder.queryfeature
    encoder.queryfeature = MethodType(_queryfeature_fp32, encoder)
    if activation_checkpointing:
        blocks = [
            encoder.model.layer1,
            encoder.model.layer2,
            encoder.model.layer3,
            encoder.model.layer4,
            network.decoder.in_blk,
            *network.decoder.res_blk_list,
            network.decoder.res_blk_last,
            *network.decoder.up_blk_list,
        ]
        for block in blocks:
            block._forward_without_checkpointing = block.forward
            block.forward = MethodType(_checkpointed_forward, block)
    return network
