# v0.1.0: PyTorch Lightning training for GAAL

This release adds `train_lightning.py` to the original Geometry-Aware Attenuation Learning implementation. It retains the upstream source files and original training/evaluation entries; the Lightning adapter provides training scheduling, logging, checkpoint save/resume, default BF16 mixed precision, and optional activation checkpointing.

## Defaults and compatibility

- BF16 mixed precision on supported CUDA GPUs; FP32 fallback on CPU. Activation checkpointing and expandable CUDA segments are on by default. Explicit allocator environment settings are respected.
- Official BatchNorm/GELU architecture, physical-attenuation GT, exponential projection normalization, geometry, renderer, and three L1 losses.
- 20 views, S=4, batch size 1, Adam LR 1e-4, StepLR every 50 epochs with gamma 0.5, 500 epochs, and upstream evaluation intervals.
- Official `G_render` checkpoints can be imported as weights; Lightning checkpoints restore full Lightning training state.

Upstream source baseline: [ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction](https://github.com/ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction/tree/f18eef2f38f9ab240eb0976feffc90481f061927).

## Validation and measured memory

The publication CPU suite has 14 tests (six cover the optional LEAP adapter), including FP32 agreement with the retained official model/training update, checkpointing gradients and BatchNorm buffers, BF16 finite backward, Lightning fit/resume/test/visual flows, and allocator default/override handling. Local checks use PyTorch 2.5.1 with Lightning 2.2.5 and 2.5.6.

On an A100 80GB PCIe with PyTorch 2.4.1+cu121 and Lightning 2.5.6, the 20-view, 256² projection, 256³ reconstruction, S=4, batch-size-1 benchmark measured a sampled process peak of 31.50 GB with the release defaults. The maintainer reports 68.69 GB for the original training under the same input settings. See [A100_MEMORY.md](A100_MEMORY.md) for supporting measurements and scope.

Supplementary A100 validation completed CUDA BF16 training, full-state checkpoint recovery, one-epoch fit with evaluation/export, FP32 numerical comparisons, and 518²/24-view memory measurements. The 518²/24-view sampled process peak with release defaults was 37.5742 GiB (40.35 GB). These checks do not establish converged reconstruction quality or long-run memory stability. See [A100_VALIDATION_20261009.md](A100_VALIDATION_20261009.md) for conditions and observed upstream metric/export limitations.

## Optional LEAP DRR

`DRR_simulation_leap.py` adds offline DRR generation with the optional [LEAP library](https://github.com/LLNL/LEAP/wiki/Installing-LEAP-without-PyTorch). It reuses the official geometry/GT conversion and writes `transforms.json`, `gt_volume.nii.gz`, and `proj.nii.gz` in `syn_data_leap`. `--resolution=256` changes detector pitch together with pixel count to preserve the original 512-pixel detector FOV. CUDA uses LEAP's cone-beam projector; the native CPU modular projector requires cubic voxels. Neither training entry requires LEAP when using prepared data. The original training projector remains unchanged.

Local LEAP 1.26 CPU checks cover image orientation, attenuation linearity, preserved detector FOV, spacing restrictions, and loading generated NIfTI/geometry through the official dataset. A100 CUDA validation also passed real-case 20×256×256 DRR generation, row/column orientation, non-cubic Z spacing, and two BF16 training steps using the generated data. Numerical projections can differ from the original renderer.

## Distribution

The wheel includes the model, Lightning modules, optional LEAP adapter, official HOCON configs and dataset splits, with `gaal-lightning`, `gaal-benchmark-memory`, and `gaal-leap-drr` console commands. The source archive additionally includes the original scripts, tests, README/images, benchmark report, handoff, and CPU CI workflow.


The original paper attribution and citation remain in the README.
