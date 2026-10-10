"""Optional offline LEAP projections using the official geometry and GT conversion."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from DRR_simulation import GeometryProduction, angle2vec


LEAP_INSTALL_URL = "https://github.com/LLNL/LEAP/wiki/Installing-LEAP-without-PyTorch"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="Start angle in degrees")
    parser.add_argument("--end", type=int, default=360, help="End angle in degrees (excluded)")
    parser.add_argument("--num", type=int, default=20, help="Number of projection angles")
    parser.add_argument("--sad", type=float, default=1000, help="Source-to-axis distance in mm")
    parser.add_argument("--sid", type=float, default=1500, help="Source-to-detector distance in mm")
    parser.add_argument("--datapath", type=Path, default=Path("./dataset/head"),
                        help="Dataset root containing raw_volume/*.nii.gz")
    parser.add_argument("--output", type=Path,
                        help="Output directory (default: <datapath>/syn_data_leap)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Square projection resolution; preserves the original 512-pixel detector FOV")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu (requires cubic voxels), or cuda:<index>")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output cases")
    return parser


def create_projector(device):
    try:
        from leapctype import tomographicModels
    except ImportError as error:
        raise RuntimeError("Install the optional LEAP library first: " + LEAP_INSTALL_URL) from error
    projector = tomographicModels()
    if projector.libprojectors is None:
        raise RuntimeError("LEAP could not load its compiled library. See " + LEAP_INSTALL_URL)
    if device == "auto":
        device = "cuda:0" if projector.number_of_gpus() > 0 else "cpu"
    if device == "cpu":
        valid = projector.set_gpu(-1)
        # A CPU-only LEAP build may return False from the GPU setter.
        valid = valid or projector.get_gpu() < 0
    elif device.startswith("cuda:") and device[5:].isdigit():
        index = int(device[5:])
        if index >= projector.number_of_gpus():
            raise ValueError("LEAP cannot access the requested GPU: " + device)
        valid = projector.set_gpus([index])
    else:
        raise ValueError("--device must be auto, cpu, or cuda:<index>")
    if not valid:
        raise RuntimeError("LEAP could not select device " + device)
    return projector, device


def resize_detector(parameters, resolution):
    """Change pixel count and pitch together, preserving the official detector FOV."""
    if resolution <= 0:
        raise ValueError("--resolution must be positive")
    parameters = copy.deepcopy(parameters)
    width, height = parameters["proj_resolution"]
    scale_u, scale_v = width / resolution, height / resolution
    parameters["proj_resolution"] = [resolution, resolution]
    parameters["proj_spacing"] = [parameters["proj_spacing"][0] * scale_u,
                                   parameters["proj_spacing"][1] * scale_v]
    for frame in parameters["frames"]:
        vector = np.asarray(frame["vec"], dtype=np.float64)
        vector[6:9] *= scale_u
        vector[9:12] *= scale_v
        frame["vec"] = vector.tolist()
    return parameters


def project_volume(volume, parameters, projector, device):
    """Return raw attenuation line integrals in [view, row, column] order."""
    size = np.asarray(parameters["volume_resolution"], dtype=int)
    spacing = np.asarray(parameters["volume_spacing"], dtype=float)
    physical = np.asarray(parameters["volume_phy"], dtype=float)
    origin = np.asarray(parameters["volume_origin"], dtype=float)
    if np.any(size <= 0) or np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError("Volume dimensions and spacing must be positive and finite")
    if not np.isclose(spacing[0], spacing[1], rtol=1e-6, atol=1e-8):
        raise ValueError("LEAP requires equal X/Y voxel spacing; resample the raw volume first")
    if device == "cpu" and not np.isclose(spacing[0], spacing[2], rtol=1e-6, atol=1e-8):
        raise ValueError("LEAP's CPU modular projector requires cubic voxels; use --device=cuda:0")
    if volume.shape != tuple(size[::-1]):
        raise ValueError("Volume shape must match the metadata in Z/Y/X order")
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("The GT volume must contain finite, nonnegative attenuation coefficients")
    if not np.isfinite(origin).all() or not np.allclose(physical, size * spacing):
        raise ValueError("Volume extent must equal volume_resolution * volume_spacing")

    vectors = np.ascontiguousarray([frame["vec"] for frame in parameters["frames"]],
                                   dtype=np.float32)
    count = parameters["N_views"]
    if vectors.shape != (count, 12) or not np.isfinite(vectors).all():
        raise ValueError("Expected one finite 12-element vec for each projection")
    width, height = parameters["proj_resolution"]
    col_steps, row_steps = vectors[:, 6:9], vectors[:, 9:12]
    pixel_widths = np.linalg.norm(col_steps, axis=1)
    pixel_heights = np.linalg.norm(row_steps, axis=1)
    if (np.any(pixel_widths <= 0) or np.any(pixel_heights <= 0)
            or not np.allclose(pixel_widths, pixel_widths[0])
            or not np.allclose(pixel_heights, pixel_heights[0])):
        raise ValueError("LEAP requires a constant positive detector pixel pitch")
    rows = np.ascontiguousarray(row_steps / pixel_heights[:, None])
    cols = np.ascontiguousarray(col_steps / pixel_widths[:, None])
    if device == "cpu":
        # Native CPU modular projection accepts the official vec directly.
        # Its Joseph kernel assumes cubic voxels, checked above.
        valid = projector.set_modularbeam(
            count, height, width, float(pixel_heights[0]), float(pixel_widths[0]),
            np.ascontiguousarray(vectors[:, :3]), np.ascontiguousarray(vectors[:, 3:6]),
            rows, cols,
        )
    else:
        # The official offline simulation is a circular cone-beam scan.
        # LEAP's angle convention differs by 90 degrees and its image rows
        # point in the opposite direction; reverse the output rows below.
        angles = parameters["start"] + np.arange(count) * parameters["angle_per_view"]
        expected = np.asarray([angle2vec(np.deg2rad(angle), 0, [0, 0, 0],
                                        parameters["sid"], parameters["sad"],
                                        float(pixel_widths[0]), float(pixel_heights[0]))
                               for angle in angles])
        if not np.allclose(vectors, expected, rtol=1e-5, atol=1e-4):
            raise ValueError("The GPU cone-beam projector requires the official circular scan geometry")
        valid = projector.set_conebeam(
            count, height, width, float(pixel_heights[0]), float(pixel_widths[0]),
            (height - 1) / 2, (width - 1) / 2,
            np.ascontiguousarray(angles + 90, dtype=np.float32),
            float(parameters["sad"]), float(parameters["sid"]),
        )
    if not valid:
        raise ValueError("LEAP rejected the projection geometry")
    center = origin + physical / 2
    valid = projector.set_volume(int(size[0]), int(size[1]), int(size[2]),
                                 float(spacing[0]), float(spacing[2]),
                                 float(center[0]), float(center[1]), float(center[2]))
    if not valid or not projector.set_volumeDimensionOrder(1):
        raise ValueError("LEAP rejected the volume geometry")
    if not np.allclose([projector.get_voxelWidth(), projector.get_voxelHeight()],
                       [spacing[0], spacing[2]], rtol=1e-6, atol=1e-8):
        raise ValueError("LEAP changed voxel spacing; this geometry requires cubic voxels")
    # Disable LEAP's automatic circular FOV mask so all official GT voxels contribute.
    if not projector.set_diameterFOV(1e10):
        raise RuntimeError("LEAP could not disable the circular volume mask")

    volume = np.ascontiguousarray(volume, dtype=np.float32)
    projections = np.full((count, height, width), np.nan, dtype=np.float32)
    if device == "cpu":
        projector.project_cpu(projections, volume)
    else:
        # CPU arrays let LEAP manage transfers and chunking internally on the selected GPU.
        projector.project(projections, volume)
    if not np.isfinite(projections).all():
        raise RuntimeError("LEAP projection failed or produced non-finite values")
    if device != "cpu":
        projections = np.ascontiguousarray(projections[:, ::-1, :])
    return projections


def generate(args):
    if args.num <= 0 or args.end <= args.start or args.resolution <= 0:
        raise ValueError("Require --num>0, --end>--start, and --resolution>0")
    if not (0 < args.sad < args.sid < float("inf")):
        raise ValueError("Require finite distances with 0 < SAD < SID")
    raw = args.datapath / "raw_volume"
    if not raw.is_dir():
        raise ValueError("Raw volume directory does not exist: " + str(raw))
    files = sorted(raw.iterdir())
    if not files or any(not path.is_file() or not path.name.endswith(".nii.gz") for path in files):
        raise ValueError("raw_volume must contain only .nii.gz volumes, as in the official workflow")
    output = args.output or args.datapath / "syn_data_leap"
    if output.resolve() == raw.resolve():
        raise ValueError("The output directory must differ from raw_volume")
    cases = [output / path.name[:-7] for path in files]
    existing = [str(case) for case in cases if case.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output cases already exist; use --overwrite to replace them: " + existing[0])
    projector, device = create_projector(args.device)
    print("LEAP DRR device:", device)
    # Keep the official HU-to-mu conversion, recentering, angle2vec, and output names.
    GeometryProduction(args, str(raw), str(output))
    for case in cases:
        with (case / "transforms.json").open() as handle:
            parameters = resize_detector(json.load(handle), args.resolution)
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "gt_volume.nii.gz")))
        projections = project_volume(volume, parameters, projector, device)
        sitk.WriteImage(sitk.GetImageFromArray(projections), str(case / "proj.nii.gz"))
        with (case / "transforms.json").open("w") as handle:
            json.dump(parameters, handle, indent=4)
        print("Finish LEAP projection generation for", case.name)
    return output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        generate(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, "LEAP DRR error: " + str(error) + "\n")
