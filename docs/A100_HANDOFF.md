# `.7` 主机 A100 验证交接

## 已确定的发布行为

- 保留官方目录及 `train.py` / `trainer.py`；新增入口为 `train_lightning.py`。
- 默认 `bf16-mixed`、activation checkpointing 和 expandable segments 开启；通过 `--no-activation-checkpointing` 关闭重算。显式 allocator 环境变量覆盖默认设置。
- 默认官方 NIfTI 数据路径、20 views、S=4、BatchNorm、GELU、物理衰减系数 GT、指数投影归一化。
- 沿用官方 endpoint 坐标、输出 X/Z 交换、射线积分及 zero padding。
- 三项 loss 都是 L1：3D=1、gradient=1、2D=0.01；Adam LR=1e-4、StepLR 每 50 epoch ×0.5、500 epochs。
- 无梯度裁剪，梯度累积为 1；保留官方 optimizer 更新后的额外 eval-mode 训练 PSNR 前向。
- README 的 `Updated Feature` 声明 Lightning / BF16 / 可选 checkpointing，并报告已完成的 20 views、256² 投影、256³ 输出、S=4 的 A100 显存实测；不声明重建精度提升。

官方源码基线：

```text
https://github.com/ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction
f18eef2f38f9ab240eb0976feffc90481f061927
```

复制到根目录的官方源文件没有修改。新增适配层只负责 CPU data loading、Lightning 调度、FP32 precision islands、可选重算和 Lightning checkpoint。运行不依赖本地 `third_party/official_gaal`。


## 已完成的检查及待办

2026-10-09 的 A100 80GB PCIe 测试结果已交回：20 views、256² 投影、256³ 输出、S=4、BF16、checkpointing 开/关，以及默认/expandable segments 两种 native allocator，四组短训练均完成。结果和口径见 [A100_MEMORY.md](A100_MEMORY.md)。测试环境为 Python 3.10.18、PyTorch 2.4.1+cu121、CUDA 12.1、Lightning 2.5.6。

本次交回的是显存结果汇总；原始 JSON/日志未同步到当前工作目录。518²/24 views、完整训练/验证显存、GPU checkpoint 恢复与导出、官方 FP32 / Lightning FP32 / BF16 数值及重建精度对照，尚无返回证据。

CPU 上已通过：

- 小模型 FP32 重建与保留的官方模型逐元素一致。
- 相同输入、权重和随机采样下，三项 loss / 一次 Adam 更新与官方 trainer 一致。
- 开关 checkpointing 时，FP32 loss、梯度和 BatchNorm buffers 一致。
- CPU BF16 autocast 下，开关 checkpointing 均可完成前向及有限梯度反向；几何和 loss 保持 FP32。
- Lightning CPU fit、best/last checkpoint、完整恢复 optimizer/scheduler、test 和 visual 输出。

这些是小体积功能/数值检查；A100 的短训练显存测试已补充真实 256³、CUDA BF16 的运行证据，但不能替代完整数值与重建精度对照。当前本机没有运行 GPU 训练。下面的任务保留作后续验证流程，已测显存配置无需因发布重复运行。

发布目录保留官方模型、Lightning 适配及可选 LEAP DRR，包含 14 项检查（实际 LEAP 投影检查需要安装该库）。

## 环境与数据

在项目根目录运行，建议 Python 3.10。已有匹配 CUDA 的 PyTorch 可保留；新环境按 README 的安装命令安装。`requirements.txt` 对应 torch 2.1.2+cu118、torchvision 0.16.2+cu118、Lightning 2.2.5。本地 CPU 检查使用 torch 2.5.1+cu121 / Lightning 2.5.6；交回的 A100 显存结果使用 torch 2.4.1+cu121 / Lightning 2.5.6。继续验证时请记录实际版本。

```bash
python -c 'import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())'
python train_lightning.py --help
python benchmark_memory.py --help
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest -q --disable-warnings
```

单个官方病例目录必须包含：

```text
<data-root>/<case>/gt_volume.nii.gz   物理衰减系数，SimpleITK [Z,Y,X]
<data-root>/<case>/proj.nii.gz        原始线积分投影 [V,H,W]
<data-root>/<case>/transforms.json    官方字段，frames[].vec 为 12 维几何
```

使用官方 `data/dataset_split/<datatype>_split.json`，或通过 `--split-path` 提供同样含 train/val/test/visual 四个键的 JSON。`datatype` 可选 `dental`、`spine`、`Walnuts`。


保留的 `DRR_simulation.py` 默认生成 512² 投影，不能直接当作 256² 数据。256² 投影必须另行准备，并同步匹配 detector spacing 和几何；只修改 HOCON 的 `encoder.input_size` 不会生效。默认 uniform 数据路径读取已有 `proj.nii.gz`，不补生成缺失文件。在线 random 路径复用官方 `models/render.py`，在 Lightning Dataset 中运行于 CPU、需要 `--num-workers 0`，实际分辨率取自 `transforms.json`。

`model.encoder.input_size` 不会调整实际输入尺寸；实际尺寸来自 `proj.nii.gz`。真实输入尺寸会写入 benchmark JSON，不能仅根据目录名或 HOCON 字段认定是 256² / 518²。真实 GT 应为 256³，且各维可被 S=4 整除；保持完整 256³ 输出，不用降低输出尺寸来满足显存目标。

## 任务一：A100 数值与训练检查

先用官方数据的小 split 做两个 optimizer step。以下变量替换为 `.7` 上的实际路径，勿照抄占位值：

```bash
GAAL_DATA_ROOT=/path/to/official/dental/syn_data
GAAL_SPLIT=/path/to/smoke_split.json

python train_lightning.py -D "$GAAL_DATA_ROOT" --split-path "$GAAL_SPLIT" \
  --datatype dental --nviews 20 --train_scale 4 --accelerator gpu \
  --name a100_bf16_smoke --no-activation-checkpointing --max-steps 2 --limit-val-batches 0

python train_lightning.py -D "$GAAL_DATA_ROOT" --split-path "$GAAL_SPLIT" \
  --datatype dental --nviews 20 --train_scale 4 --accelerator gpu \
  --name a100_bf16_ckpt_smoke --activation-checkpointing \
  --max-steps 2 --limit-val-batches 0
```

确认默认运行输出 `Using bfloat16 Automatic Mixed Precision`，loss 和梯度有限，且写出 `last.ckpt`。恢复时使用同一配置，增大 `--max-steps`：

```bash
python train_lightning.py -D "$GAAL_DATA_ROOT" --split-path "$GAAL_SPLIT" \
  --datatype dental --nviews 20 --train_scale 4 --accelerator gpu \
  --name a100_bf16_smoke --no-activation-checkpointing --max-steps 3 --limit-val-batches 0 \
  --checkpoint outputs/lightning/a100_bf16_smoke/checkpoints/last.ckpt
```

运行 test / visual，确认尺寸、NIfTI、指标及文件路径：

```bash
python train_lightning.py --mode test -D "$GAAL_DATA_ROOT" --split-path "$GAAL_SPLIT" \
  --datatype dental --nviews 20 --train_scale 4 --accelerator gpu \
  --name a100_bf16_smoke \
  --checkpoint outputs/lightning/a100_bf16_smoke/checkpoints/last.ckpt
```

短训练不能用于声称重建精度达到论文结果。CUDA 数值对照应固定同一个病例、权重、输入视角和 ray RNG，分别运行官方 FP32、Lightning FP32、Lightning BF16。FP32 优先核对预测、三项 loss 和参数更新；BF16 报告误差及是否出现非有限值，不要求逐元素等于 FP32。沿用官方的 clamp + 独立 min-max + PSNR/SSIM 口径。若采用公开 dental 权重，对照病例须为配套 dental 数据，不用它评定 Walnut 精度。

## 任务二：训练峰值显存

`benchmark_memory.py` 每个进程只使用一张 GPU、一个固定病例和一个配置，先 warmup 2 个完整 optimizer step，再测 3 个 step。测量包含前向、三项 loss、反向、Adam states，以及官方训练 PSNR 的额外 eval-mode 前向；关闭 val/test/visual。不要用纯推理显存代替训练显存。

脚本输出 JSON，包含实际输入尺寸、配置、GPU 总显存、CUDA/PyTorch/Lightning 版本、TF32 设置、allocator backend / 环境变量和：

- `peak_allocated_bytes / gb / gib`：PyTorch 张量分配峰值。
- `peak_reserved_bytes / gb / gib`：PyTorch allocator 保留峰值。
- `status`：complete 或 oom；OOM 时仍写结果，进程返回失败。

GB=10^9 bytes，GiB=2^30 bytes，写结果时不要混用。两类数值都不包含 CUDA context / 第三方库的进程外开销；若要声明某种显存容量的 GPU 可运行，还应记录 `nvidia-smi` 的进程/设备占用、A100 型号和容量，确认 GPU 没有其他任务占用。

### 真实数据主测试

下面保留四组复现命令，每组启动新进程。20 views/256² 和 24 views/518² 均已完成；518² 两组补测见 [验证记录](A100_VALIDATION_20261009.md)。需要两份实际分辨率不同且物理几何正确的数据；`--projection-size` 只控制 synthetic，不能调整真实数据。allocator 对照的完整命令见 [A100_MEMORY.md](A100_MEMORY.md)。

```bash
GAAL_DATA_ROOT_256=/path/to/official/dental/projections_256
GAAL_DATA_ROOT_518=/path/to/official/dental/projections_518
GAAL_CASE=X2316453

# checkpointing off 对照：20 views，256² 投影，256³ GT，S=4，BF16
python benchmark_memory.py -h
python benchmark_memory.py --datadir "$GAAL_DATA_ROOT_256" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --no-activation-checkpointing --output outputs/memory/real_256_20v_off.json

# 同样输入，checkpointing on
python benchmark_memory.py --datadir "$GAAL_DATA_ROOT_256" --case "$GAAL_CASE" \
  --nviews 20 --train_scale 4 --activation-checkpointing \
  --output outputs/memory/real_256_20v_on.json

# 历史输入规格参考：24 views，518² 投影，256³ GT，S=4
python benchmark_memory.py --datadir "$GAAL_DATA_ROOT_518" --case "$GAAL_CASE" \
  --nviews 24 --train_scale 4 --no-activation-checkpointing --output outputs/memory/real_518_24v_off.json

python benchmark_memory.py --datadir "$GAAL_DATA_ROOT_518" --case "$GAAL_CASE" \
  --nviews 24 --train_scale 4 --activation-checkpointing \
  --output outputs/memory/real_518_24v_on.json
```

若使用自定义 split，追加 `--split-path "$GAAL_SPLIT"`；上述示例病例也要替换成实际可用病例。checkpointing off 若 OOM，保留 JSON，如实记录，使用另一独立进程测试 on，不降低 views / S / 输出尺寸来替换默认结果。

### 数据尚未就绪时的 synthetic 预检查

以下命令生成物理 attenuation GT、原始 projection 输入和配套圆轨道 vec，执行同一模型和三项 loss。只用于容量/实现检查；`source=synthetic` 会明确写进 JSON，不能冒充真实数据测量：

```bash
python benchmark_memory.py --synthetic --projection-size 256 --volume-size 256 \
  --nviews 20 --train_scale 4 --no-activation-checkpointing --output outputs/memory/synthetic_256_20v_off.json

python benchmark_memory.py --synthetic --projection-size 256 --volume-size 256 \
  --nviews 20 --train_scale 4 --activation-checkpointing \
  --output outputs/memory/synthetic_256_20v_on.json

python benchmark_memory.py --synthetic --projection-size 518 --volume-size 256 \
  --nviews 24 --train_scale 4 --no-activation-checkpointing --output outputs/memory/synthetic_518_24v_off.json
```

若主机由 Slurm 管理，先申请 A100 资源并在分配的任务中运行这些命令。需要终止训练/benchmark 时用 `scancel <job_id>` 释放资源；不使用 `scontrol suspend`。

## 返回材料与结果记录

请保留并补交 benchmark JSON、训练日志、`run_config.json`、实际数据尺寸和 A100 型号/容量。已收到结果如下，显存单位为 GiB；所有已测行均是同一真实病例的短训练：

| 投影 / views | GT / 输出 | S / precision | checkpointing | allocator | allocated GiB | reserved GiB | 采样进程峰值 GiB | 结果 |
|---|---|---|---|---|---:|---:|---:|---|
| 256² / 20 | 256³ / 256³ | 4 / bf16-mixed | off | default native | 30.2874 | 37.0840 | 37.5938 | 完成 |
| 256² / 20 | 256³ / 256³ | 4 / bf16-mixed | on | default native | 26.6480 | 34.4824 | 34.9922 | 完成 |
| 256² / 20 | 256³ / 256³ | 4 / bf16-mixed | off | expandable | 30.2725 | 31.4023 | 31.9121 | 完成 |
| 256² / 20 | 256³ / 256³ | 4 / bf16-mixed | on | expandable | 26.6387 | 28.8223 | 29.3320 | 完成 |
| 518² / 24 | 256³ / 256³ | 4 / bf16-mixed | off | expandable | 43.3890 | 47.1250 | 47.6348 | 完成 |
| 518² / 24 | 256³ / 256³ | 4 / bf16-mixed | on | expandable | 35.0084 | 37.0645 | 37.5742 | 完成 |

2026-10-10 的发布默认改为 checkpointing + expandable segments 开启。README 仅报告该默认组合的采样进程峰值 31.50 GB，以及维护者提供的相同输入条件下原训练 68.69 GB；原训练原始测量记录未同步到本地。继续返回的数据用于补充相应条件的结论。

## 可选 LEAP DRR 的 GPU 验证

先按 [LEAP 官方安装说明](https://github.com/LLNL/LEAP/wiki/Installing-LEAP-without-PyTorch) 安装带 CUDA 支持的库。准备仅包含一个真实 dental 病例的 `raw_volume` 目录，在已分配的 A100 上运行：

```bash
python DRR_simulation_leap.py --start=0 --end=360 --num=20 --sad=500 --sid=700 --datapath=/path/to/dental_smoke --resolution=256 --device=cuda:0
```

输出在 `syn_data_leap/<case>/`，包含与官方相同的三份文件。确认投影尺寸为 `[20,256,256]`、数据有限且方向正确，再把该目录传给上述 Lightning smoke 命令的 `-D=` 参数。GPU 路径使用 LEAP cone-beam，角度偏移 +90° 并翻转投影行方向；检查 `transforms.json` 中的 vec 与投影保持对应，尤其是源 CT 的 Z spacing 与 X/Y 不同时。

本机 LEAP 1.26 的小体积 CPU 检查已通过；CPU modular 投影只接受等距体素，不会自动重采样 GT。A100 补测已通过真实 20×256×256 DRR、与官方射线积分的方向核对、[0.7,0.7,1.2] mm 非等距体素检查，以及使用生成数据的两步 BF16 训练。LEAP 离线投影与原手写积分存在数值差异，训练投影损失及在线随机视角 DRR 仍使用原实现。

## 补测完成记录

Job 4196790 的 16 个 CUDA 子进程均返回 0，CPU 测试 14/14 通过；GPU 训练、完整 checkpoint 恢复、fit 内 val/test/visual、独立导出及数值对照已经核验。返回的 118 份原始资料逐文件 SHA256 全部一致，完整回交材料已在本地归档，发布目录保留[精简验证记录](A100_VALIDATION_20261009.md)。随机初始化的常量预测会触发官方 min-max 指标 NaN，导出 NIfTI 的物理头信息也沿用官方默认值；本次记录这两处限制，未修改官方代码。尚未进行收敛训练或公开权重的重建质量对照。
