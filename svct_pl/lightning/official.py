"""The official GAAL training calculations with Lightning-owned optimization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
from pyhocon import ConfigFactory
import torch
from torch.nn import functional as F

from models.loss import gradient1_loss
from models.render import composite, get_rays, mu2ct, predict_3d_volume
from util.util_func import data_norm, get_psnr, get_ssim_3d, tensor2nii

from .official_model import build_official_model


def _vector(value, device, dtype=torch.float32):
    if isinstance(value, (list, tuple)):
        return torch.cat([torch.as_tensor(part, device=device).reshape(-1)
                          for part in value]).to(dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)


class OfficialSVCTModule(pl.LightningModule):
    def __init__(self, conf, args):
        super().__init__()
        self.save_hyperparameters({"conf": dict(conf), "args": dict(args)})
        self.conf = ConfigFactory.from_dict(conf)
        self.args = SimpleNamespace(**args)
        self.model = build_official_model(
            self.conf["model"],
            activation_checkpointing=self.args.activation_checkpointing,
        )
        self.clamp_min = self.conf.get_float(f"data.{self.args.datatype}.clamp_min")
        self.clamp_max = self.conf.get_float(f"data.{self.args.datatype}.clamp_max")
        self.divide = 10 if self.args.datatype == "spine" else 1
        self.output_dir = Path(self.args.output_root) / self.args.name

    def prepare_batch(self, batch):
        images = batch["images"].squeeze(0).float()
        if self.args.expnorm:
            images = torch.exp(-images / self.divide)
        paras = batch["paras"]
        return {
            "images": images,
            "poses": batch["poses"].squeeze(0).float(),
            "target": batch["3Dvolume"].squeeze(0).float().clamp(self.clamp_min, self.clamp_max),
            "physical": _vector(paras["volume_phy"], images.device),
            "origin": _vector(paras["volume_origin"], images.device),
            "spacing": _vector(paras["volume_spacing"], images.device),
            "resolution": _vector(paras["volume_resolution"], images.device, torch.int64),
            "obj_index": str(batch["obj_index"][0]),
        }

    def reconstruct(self, item, scale=None):
        self.model.encoder(item["images"], item["poses"])
        prediction = predict_3d_volume(
            model=self.model,
            volume_resolution=item["resolution"],
            volume_origin=item["origin"],
            volume_phy=item["physical"],
            scale=self.model.decoder.scale if scale is None else scale,
            device=item["images"].device,
        )
        if prediction.shape != item["target"].shape:
            raise ValueError(
                f"Official reconstruction shape {tuple(prediction.shape)} differs from GT "
                f"{tuple(item['target'].shape)}. Use volumes divisible by --train_scale and "
                "--eval_scale matching the decoder scale for full-resolution reconstruction."
            )
        return prediction.float()

    def losses(self, prediction, item):
        # The upstream names say 'mse' but the loss function is L1.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = item["target"].float()
            loss_3d = F.l1_loss(prediction, target) * self.conf.get_float("train.G_loss.mse_lambda_3d")
            total = loss_3d
            values = {"loss_3d": loss_3d}
            gradient_weight = self.conf.get_float("train.G_loss.gd1_lambda")
            if gradient_weight > 0:
                gradient = gradient1_loss(target, prediction, F.l1_loss) * gradient_weight
                total = total + gradient
                values["loss_gradient"] = gradient
            projection_weight = self.conf.get_float("train.G_loss.mse_lambda_2d")
            if projection_weight > 0:
                images = item["images"]
                _, _, height, width = images.shape
                # Keep upstream CPU RNG and sampling with replacement.
                indices = torch.randint(0, self.args.nviews * height * width,
                                        (self.conf.get_int("render.ray_batch_size"),), device="cpu")
                projection_gt = images.reshape(-1, 1)[indices]
                rays = get_rays(item["poses"], height, width)
                rays = rays.reshape(-1, rays.shape[-1])[indices]
                projection = composite(
                    rays=rays,
                    volume=prediction,
                    volume_origin=item["origin"],
                    volume_phy=item["physical"],
                    render_step_size=item["spacing"].min() * self.conf.get_float("render.factor"),
                    chunksize=self.conf.get_int("render.chunksize"),
                ).reshape_as(projection_gt)
                if self.args.expnorm:
                    projection = torch.exp(-projection / self.divide)
                loss_2d = F.l1_loss(projection, projection_gt) * projection_weight
                values["loss_2d"] = loss_2d
                total = total + loss_2d
            values["loss"] = total
            return values

    def training_step(self, batch, batch_idx):
        item = self.prepare_batch(batch)
        values = self.losses(self.reconstruct(item), item)
        for name, value in values.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True,
                     prog_bar=name == "loss", batch_size=1)
        return values["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Preserve upstream's post-update eval-mode training PSNR instead of
        # changing the metric to the pre-update train-mode prediction.
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.trainer.precision_plugin.forward_context():
                item = self.prepare_batch(batch)
                prediction = self.reconstruct(item)
                psnr = get_psnr(data_norm(prediction.clamp(self.clamp_min, self.clamp_max)),
                                data_norm(item["target"]))
                self.log("train/psnr_3d_clamp", float(psnr), on_step=False,
                         on_epoch=True, batch_size=1)
        finally:
            self.model.train(previous_mode)

    def metrics(self, prediction, target):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = data_norm(prediction.float().clamp(self.clamp_min, self.clamp_max))
            target = data_norm(target.float())
            psnr = get_psnr(prediction, target)
            return {"psnr_3d_clamp": float("inf") if psnr == "INF" else float(psnr),
                    "ssim_3d_clamp": float(get_ssim_3d(prediction, target, data_range=1))}

    def _save_volume(self, item, prediction, stage):
        if not self.trainer.is_global_zero:
            return
        destination = self.output_dir / stage / item["obj_index"] / "volume"
        destination.mkdir(parents=True, exist_ok=True)
        tensor2nii(mu2ct(item["target"]), str(destination / "volume_gt.nii.gz"))
        name = f"volume_{self.current_epoch}.nii.gz" if stage == "visual" else "volume_predict.nii.gz"
        # Upstream saves raw predictions in HU and the native [Z,Y,X] array;
        # clamping/self-normalization are only for computing metrics.
        tensor2nii(mu2ct(prediction), str(destination / name))

    def _evaluate_step(self, batch, stage, save=False):
        item = self.prepare_batch(batch)
        prediction = self.reconstruct(item, self.args.eval_scale)
        values = self.metrics(prediction, item["target"])
        for name, value in values.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True, add_dataloader_idx=False)
        if save:
            self._save_volume(item, prediction, stage)
        return {"obj_index": item["obj_index"], **values}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        stage = ("val", "test", "visual")[dataloader_idx]
        # Intervals keep the upstream zero-based epoch convention. The final
        # epoch always runs all three datasets, as in trainer.start().
        interval = self.conf.get_int(f"train.print.{stage if stage != 'visual' else 'vis'}_interval")
        if self.trainer.state.fn == "fit":
            epoch = self.current_epoch
            if not ((epoch > 0 and epoch % interval == 0) or epoch == self.trainer.max_epochs - 1):
                return None
        return self._evaluate_step(batch, stage, save=stage == "visual")

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, "test", save=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # predict has no metric aggregation loop, so return scalar metrics.
        item = self.prepare_batch(batch)
        prediction = self.reconstruct(item, self.args.eval_scale)
        self._save_volume(item, prediction, "visual")
        return {"obj_index": item["obj_index"], **self.metrics(prediction, item["target"])}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.get_float("lr_sche.init_lr"))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.conf.get_int("lr_sche.step_size"),
            gamma=self.conf.get_float("lr_sche.gamma"),
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            settings = {"conf": self.hparams.conf, "args": vars(self.args),
                        "actual_precision": self.trainer.precision}
            (self.output_dir / "run_config.json").write_text(
                json.dumps(settings, indent=2), encoding="utf-8"
            )
