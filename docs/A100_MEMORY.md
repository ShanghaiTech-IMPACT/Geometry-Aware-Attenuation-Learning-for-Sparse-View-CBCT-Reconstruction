# A100 training memory measurements

Release defaults as of 2026-10-10 are BF16, activation checkpointing on, and expandable CUDA segments on. The README reports the corresponding sampled process peak: 29.3320 GiB × 2³⁰ / 10⁹ = **31.50 GB**. The maintainer additionally reports **68.69 GB** for the original training under the same input/reconstruction settings; its raw measurement logs are not included in this workspace.

## Conditions and provenance

Measurements supplied on 2026-10-09 used NVIDIA A100 80GB PCIe, Python 3.10.18, PyTorch 2.4.1+cu121, CUDA 12.1, and PyTorch Lightning 2.5.6.

All four configurations used the same real case (`X2313838`), offline projections and physical-attenuation GT, seed 42, official model/configuration, BatchNorm/GELU, Adam with LR 1e-4, and:

- Batch size 1; 20 uniformly spaced views over 360°.
- Projection input `[1, 20, 1, 256, 256]`; GT and reconstructed output `[1, 256, 256, 256]`.
- Downsampling rate S=4; BF16 mixed precision.
- A separate process per configuration; 2 warmup optimizer steps followed by 3 measured steps.
- Full training steps: forward, three losses, backward, optimizer update, and the upstream post-update eval-mode training PSNR forward. Validation/test/visualization were disabled.

The baseline runs had both `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` unset. Supplementary runs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before Python started. Both used the native allocator; the supplied summary reports allocator snapshots confirming expandable segments (8 segments without checkpointing, 5 with checkpointing).

The supplied summary identifies baseline job 4195390 and supplementary job 4195427 (`COMPLETED`). “Original” in that summary means the same Lightning benchmark with default allocator settings, **not a comparison against the original FP32 trainer**. The summary is retained locally as `memory_results_expandable.txt`; machine-specific paths in that file are excluded from the release. Supplementary A100 verification returned the original four-run JSON records without repeating that benchmark; the complete raw handoff is archived locally outside the publication tree.

## Results

| Checkpointing | Allocator | Peak allocated (GiB) | Peak reserved (GiB) | Sampled process peak (GiB) | Three measured steps (s) |
|---|---|---:|---:|---:|---:|
| Off | Default native | 30.2874 | 37.0840 | 37.5938 | 5.2429 |
| On | Default native | 26.6480 | 34.4824 | 34.9922 | 5.6186 |
| Off | Native + expandable segments | 30.2725 | 31.4023 | 31.9121 | 5.3203 |
| On | Native + expandable segments | 26.6387 | 28.8223 | 29.3320 | 5.7064 |

GiB = 2³⁰ bytes; GB = 10⁹ bytes. For example, default/no-checkpointing reserved memory of 37.0840 GiB is **39.82 GB**. This is allocator memory, not a measured whole-process peak or a guarantee that complete training fits on a 40GB GPU.

Checkpointing reduced allocated memory by 3.6394 GiB (**12.02%**) with the default allocator, and by 3.6338 GiB (**12.00%**) with expandable segments. The release enables it by default while retaining the original BatchNorm architecture.

Expandable segments reduced reserved memory by 5.6817 GiB (**15.32%**) with checkpointing off and by 5.6601 GiB (**16.41%**) with it on; differences use the rounded table values. Allocated memory changed by only 0.0149 / 0.0093 GiB. The observed reduction is consistent with improved allocation reuse, rather than a reduction in live tensor requirements. The option and its behavior are described in the [PyTorch 2.4.1 CUDA documentation](https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/cuda.rst#optimizing-memory-usage-with-pytorch_cuda_alloc_conf).

The recorded three-step times are approximately 7.2% longer with checkpointing and 1.5% longer with expandable segments. Three steps without repeated trials are insufficient for a reliable throughput conclusion; no speed claim is made.

## Reproduction

Use a prepared official-layout dataset containing `gt_volume.nii.gz`, `proj.nii.gz` and `transforms.json` for each case. The real-data benchmark reads the input size as stored; `--projection-size` only controls synthetic fixtures. Neither the training CLI nor the benchmark automatically creates 256² data from the retained default 512² DRR script.

Replace the paths and case identifier below with your available data. Explicitly disable expandable segments for the historical baseline, and launch each command as a separate process:

```bash
GAAL_DATA_ROOT=/path/to/official/projections_256
GAAL_CASE=X2313838

env -u PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  python benchmark_memory.py --datadir "$GAAL_DATA_ROOT" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --warmup-steps 2 --measure-steps 3 \
  --no-activation-checkpointing --output outputs/memory/20v256_s4_default_off.json

env -u PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  python benchmark_memory.py --datadir "$GAAL_DATA_ROOT" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --warmup-steps 2 --measure-steps 3 \
  --activation-checkpointing --output outputs/memory/20v256_s4_default_on.json

env -u PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python benchmark_memory.py --datadir "$GAAL_DATA_ROOT" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --warmup-steps 2 --measure-steps 3 \
  --no-activation-checkpointing --output outputs/memory/20v256_s4_expandable_off.json

env -u PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python benchmark_memory.py --datadir "$GAAL_DATA_ROOT" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --warmup-steps 2 --measure-steps 3 \
  --activation-checkpointing --output outputs/memory/20v256_s4_expandable_on.json
```

The benchmark JSON records actual shapes, precision, checkpointing, software versions, allocator backend/environment, and allocated/reserved peaks. Whole-process monitoring with `nvidia-smi` is external to this script.

## Scope of the release claims

The results support reporting short-run training memory for the tested 20-view, 256²/256³, S=4 BF16 setup, the approximately 12% allocated-memory reduction from checkpointing, and lower reserved memory with the optional allocator setting.

The four-run allocator/checkpointing comparison does not isolate the BF16 contribution relative to FP32 or establish reconstruction-quality equivalence, memory use on other cases/GPUs, or long-run memory stability. Supplementary CUDA numerical, evaluation/export, and 518²/24-view checks are recorded in [A100_VALIDATION_20261009.md](A100_VALIDATION_20261009.md).

## Supplementary 24-view, 518² measurements

Job 4196790 used the same real dental case and software/GPU configuration, BF16, B=1, S=4, 256³ reconstruction, and expandable segments. Each configuration ran in its own process with two warmup and three measured optimizer steps. LEAP CUDA generated the 518² data while preserving the earlier 256² detector FOV of 112.2800064 mm.

| Checkpointing | Peak allocated (GiB) | Peak reserved (GiB) | Sampled process peak (GiB) | Process peak (GB) | Three measured steps (s) |
|---|---:|---:|---:|---:|---:|
| Off | 43.3890 | 47.1250 | 47.6348 | 51.15 | 7.0656 |
| On | 35.0084 | 37.0645 | 37.5742 | 40.35 | 7.6612 |

Both configurations completed without OOM on the A100 80GB PCIe. Process peaks include warmup and CUDA context, sampled with `nvidia-smi`; the polling sleep was 0.25 seconds plus command runtime. The existing README keeps the previously agreed 20-view/256² result of 31.50 GB.
