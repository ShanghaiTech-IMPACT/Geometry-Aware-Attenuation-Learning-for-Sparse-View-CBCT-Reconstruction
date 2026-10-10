"""Compatibility checks against the retained official trainer and model."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from models.model import model as OfficialModel
from models.render import angle2vec, predict_3d_volume
from trainer import trainer as OfficialTrainer
from svct_pl.lightning.official import OfficialSVCTModule
from svct_pl.lightning.official_cli import build_parser, build_trainer, load_official_config
from svct_pl.lightning.official_data import OfficialCBCTDataModule
from svct_pl.lightning.checkpoint import load_legacy_g_render_checkpoint
from train_lightning import configure_allocator


def tiny_settings(tmp_path, checkpointing=False):
    args = build_parser().parse_args([])
    args.nviews = 2
    args.train_scale = 2
    args.epochs = 1
    args.accelerator = "cpu"
    args.precision = "32-true"
    args.output_root = str(tmp_path / "output")
    args.datadir = str(tmp_path / "data")
    args.split_path = str(tmp_path / "split.json")
    args.activation_checkpointing = checkpointing
    conf = load_official_config(args)
    overrides = {
        "model.encoder.num_layers": 1,
        "model.encoder.inplanes": 4,
        "model.encoder.feat_num_list": [4, 8, 16, 32],
        "model.encoder.layer_num_list": [1, 1, 1, 1],
        "model.encoder.latent_size": 8,
        "model.aggregator.latent_size": 8,
        "model.SRGAN.generator.inplanes": 8,
        "model.SRGAN.generator.res_blk_num": 1,
        "model.SRGAN.generator.channel_reduce_factor": 2,
        "render.ray_batch_size": 8,
    }
    for name, value in overrides.items():
        conf.put(name, value)
    return args, conf


def synthetic_batch():
    poses = np.stack([angle2vec(a, 0, [0, 0, 0], 12, 8, 1, 1)
                      for a in [0, np.pi / 2]])
    return {
        "images": torch.rand(1, 2, 1, 12, 12),
        "poses": torch.tensor(poses, dtype=torch.float32)[None],
        "3Dvolume": torch.rand(1, 8, 8, 8) * 0.04,
        "paras": {"volume_phy": [torch.tensor([8.])] * 3,
                  "volume_origin": [torch.tensor([-4.])] * 3,
                  "volume_resolution": [torch.tensor([8])] * 3,
                  "volume_spacing": [torch.tensor([1.])] * 3},
        "obj_index": ["case"],
    }


def test_release_defaults_follow_official_configuration():
    args = build_parser().parse_args([])
    conf = load_official_config(args)
    assert args.precision == "bf16-mixed"
    assert args.activation_checkpointing is True
    assert build_parser().parse_args(["--no-activation-checkpointing"]).activation_checkpointing is False
    assert (args.nviews, args.train_scale, args.epochs) == (20, 4, 500)
    assert args.expnorm is True
    assert conf["model.encoder.normalization"] == "Batch"
    assert conf["model.SRGAN.generator.normalization"] == "Batch"
    assert conf["model.last_layer.act"] == "GELU"
    assert conf["lr_sche.step_size"] == 50
    assert conf["lr_sche.gamma"] == 0.5


def test_allocator_default_and_explicit_environment_overrides(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    configure_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    configure_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:native")
    configure_allocator()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_fp32_reconstruction_and_legacy_weights_match_original_model(tmp_path):
    args, conf = tiny_settings(tmp_path)
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args)).eval()
    reference = OfficialModel(conf["model"], device="cpu").eval()
    reference.load_state_dict(module.model.state_dict(), strict=True)
    # Strict legacy loading must also work with the adapted instance.
    load_legacy_g_render_checkpoint(module, {"G_render": reference.state_dict()}, strict=True)
    batch = synthetic_batch()
    item = module.prepare_batch(batch)
    with torch.no_grad():
        reference.encoder(item["images"], item["poses"])
        expected = predict_3d_volume(reference, item["resolution"], item["origin"],
                                     item["physical"], reference.decoder.scale, "cpu")
        actual = module.reconstruct(item)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp32_loss_and_adam_update_match_original_trainer(tmp_path):
    args, conf = tiny_settings(tmp_path)
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args)).train()
    reference_model = OfficialModel(conf["model"], device="cpu").train()
    reference_model.load_state_dict(module.model.state_dict(), strict=True)
    legacy_args = SimpleNamespace(**vars(args), is_train=True, resume=False,
                                  resume_name=None, logs_path=str(tmp_path / "logs"),
                                  visual_path=str(tmp_path / "visual"),
                                  checkpoints_path=str(tmp_path / "checkpoints"))
    reference = OfficialTrainer(reference_model, None, None, None, None,
                                legacy_args, conf, "cpu")
    batch = synthetic_batch()
    torch.manual_seed(123)
    expected = reference.train_step(batch)
    optimizer = module.configure_optimizers()["optimizer"]
    torch.manual_seed(123)
    item = module.prepare_batch(batch)
    values = module.losses(module.reconstruct(item), item)
    assert float(values["loss"].detach()) == pytest.approx(expected["G_loss"], abs=1e-8)
    assert float(values["loss_2d"].detach()) == pytest.approx(expected["mse_loss_2d"], abs=1e-8)
    values["loss"].backward()
    optimizer.step()
    for key, value in module.model.state_dict().items():
        torch.testing.assert_close(value, reference_model.state_dict()[key], rtol=0, atol=0)


def test_checkpointing_preserves_loss_gradients_and_batchnorm_buffers(tmp_path):
    args, conf = tiny_settings(tmp_path)
    plain = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args)).train()
    args.activation_checkpointing = True
    recomputed = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args)).train()
    recomputed.load_state_dict(plain.state_dict(), strict=True)
    batch = synthetic_batch()
    losses = []
    for module in [plain, recomputed]:
        torch.manual_seed(124)
        item = module.prepare_batch(batch)
        loss = module.losses(module.reconstruct(item), item)["loss"]
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, recomputed.state_dict()[key], rtol=0, atol=0)
    for (name, param), (_, other) in zip(plain.named_parameters(), recomputed.named_parameters()):
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_bf16_autocast_uses_fp32_geometry_and_finite_backward(tmp_path, checkpointing):
    args, conf = tiny_settings(tmp_path, checkpointing)
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args)).train()
    item = module.prepare_batch(synthetic_batch())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = module.reconstruct(item)
        loss = module.losses(prediction, item)["loss"]
    assert prediction.dtype == torch.float32
    assert loss.dtype == torch.float32
    loss.backward()
    gradients = [param.grad for param in module.parameters() if param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)


def write_official_case(args):
    case = Path(args.datadir) / "case"
    case.mkdir(parents=True)
    rng = np.random.default_rng(42)
    volume = rng.uniform(0, 0.04, (8, 8, 8)).astype(np.float32)
    projections = rng.uniform(0, 1, (2, 12, 12)).astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(case / "gt_volume.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(projections), str(case / "proj.nii.gz"))
    metadata = {"obj_index": "case", "angle_per_view": 180,
                "volume_phy": [8., 8., 8.], "volume_origin": [-4., -4., -4.],
                "volume_resolution": [8, 8, 8], "volume_spacing": [1., 1., 1.],
                "proj_resolution": [12, 12],
                "frames": [{"vec": angle2vec(a, 0, [0, 0, 0], 12, 8, 1, 1).tolist()}
                           for a in [0, np.pi / 2]]}
    (case / "transforms.json").write_text(json.dumps(metadata))
    Path(args.split_path).write_text(json.dumps({stage: ["case"]
                                               for stage in ["train", "val", "test", "visual"]}))


def test_lightning_cpu_fit_resume_test_and_visual(tmp_path):
    args, conf = tiny_settings(tmp_path)
    write_official_case(args)
    module = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args))
    data = OfficialCBCTDataModule(vars(args))
    trainer = build_trainer(args, conf)
    trainer.fit(module, datamodule=data)
    destination = Path(args.output_root) / args.name
    checkpoint = destination / "checkpoints" / "last.ckpt"
    assert checkpoint.is_file()
    assert list((destination / "checkpoints").glob("best-*.ckpt"))
    assert (destination / "checkpoints" / "history" / "epoch-0000.ckpt").is_file()
    assert "val/psnr_3d_clamp" in trainer.callback_metrics
    assert (destination / "visual" / "case" / "volume" / "volume_0.nii.gz").is_file()
    # Epoch 1 skips all periodic evaluation; epoch 2 is the final epoch and
    # evaluates again. This also exercises checkpoint callbacks without a
    # freshly produced validation metric on the skipped epoch.
    args.epochs = 3
    resumed = OfficialSVCTModule(conf.as_plain_ordered_dict(), vars(args))
    trainer = build_trainer(args, conf)
    trainer.fit(resumed, datamodule=OfficialCBCTDataModule(vars(args)), ckpt_path=str(checkpoint))
    assert trainer.global_step == 3
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 3
    trainer.test(resumed, datamodule=OfficialCBCTDataModule(vars(args)))
    assert (destination / "test" / "case" / "volume" / "volume_predict.nii.gz").is_file()
    predicted = trainer.predict(resumed, datamodule=OfficialCBCTDataModule(vars(args)))
    assert predicted[0]["obj_index"] == "case"
