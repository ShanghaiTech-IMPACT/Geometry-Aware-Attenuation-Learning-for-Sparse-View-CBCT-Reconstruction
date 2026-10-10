"""Geometry, attenuation, and official-dataset compatibility of optional LEAP DRRs."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import SimpleITK as sitk

from models.render import angle2vec, ct2mu
from svct_pl.leap_drr import build_parser, create_projector, generate, project_volume, resize_detector
from svct_pl.lightning.official_data import LightningCBCTDataset


def parameters(size=(12, 10, 8), spacing=(1., 1., 1.), resolution=33):
    physical = np.asarray(size) * spacing
    return {
        "N_views": 2, "volume_resolution": list(size), "volume_spacing": list(spacing),
        "volume_phy": physical.tolist(), "volume_origin": (-physical / 2).tolist(),
        "proj_resolution": [resolution, resolution], "proj_spacing": [1., 1.],
        "frames": [{"file": str(index).zfill(4),
                    "vec": angle2vec(angle, 0, [0, 0, 0], 100, 50, 1, 1).tolist()}
                   for index, angle in enumerate([0, np.pi / 2])],
    }


@pytest.fixture
def cpu_projector():
    module = pytest.importorskip("leapctype", reason="LEAP is an optional compiled dependency")
    probe = module.tomographicModels()
    if probe.libprojectors is None:
        pytest.skip("The optional LEAP compiled library is unavailable")
    return create_projector("cpu")[0]


@pytest.mark.parametrize("resolution", [512, 256])
def test_resolution_preserves_detector_extent_and_vec_directions(resolution):
    original = parameters(resolution=512)
    modified = resize_detector(original, resolution)
    assert original["proj_resolution"] == [512, 512]
    assert modified["proj_resolution"] == [resolution, resolution]
    np.testing.assert_allclose(np.array(modified["proj_spacing"]) * resolution,
                               np.array(original["proj_spacing"]) * 512)
    before = np.array([frame["vec"] for frame in original["frames"]])
    after = np.array([frame["vec"] for frame in modified["frames"]])
    np.testing.assert_array_equal(after[:, :6], before[:, :6])
    np.testing.assert_allclose(after[:, 6:] * resolution, before[:, 6:] * 512)


def test_unequal_xy_spacing_rejected_before_projection():
    metadata = parameters(spacing=(1., 2., 1.))
    projector = Mock()
    with pytest.raises(ValueError, match="equal X/Y voxel spacing"):
        project_volume(np.zeros((8, 10, 12), dtype=np.float32), metadata, projector, "cpu")
    projector.set_modularbeam.assert_not_called()


def test_cpu_non_cubic_voxels_rejected_before_geometry_can_be_changed():
    metadata = parameters(spacing=(0.7, 0.7, 1.2))
    projector = Mock()
    with pytest.raises(ValueError, match="CPU modular projector requires cubic voxels"):
        project_volume(np.zeros((8, 10, 12), dtype=np.float32), metadata, projector, "cpu")
    projector.set_modularbeam.assert_not_called()


def test_cpu_projection_preserves_row_column_orientation_and_attenuation_units(cpu_projector):
    metadata = parameters()
    # Positive Y/Z shifts must appear to the right / above at primary angle zero.
    volume = np.zeros((8, 10, 12), dtype=np.float32)
    volume[5:7, 6:8, 3:9] = 0.022
    projection = project_volume(volume, metadata, cpu_projector, "cpu")
    assert projection.shape == (2, 33, 33)
    assert projection.dtype == np.float32 and np.isfinite(projection).all()
    row, col = np.unravel_index(projection[0].argmax(), projection[0].shape)
    assert row < 16 and col > 16
    scaled = project_volume(volume * 2, metadata, cpu_projector, "cpu")
    np.testing.assert_allclose(scaled, projection * 2, rtol=2e-6, atol=1e-7)


def test_generated_nifti_and_geometry_load_through_official_dataset(tmp_path, cpu_projector):
    root = tmp_path / "dental"
    raw = root / "raw_volume"
    raw.mkdir(parents=True)
    hu = np.full((32, 32, 32), -1000., dtype=np.float32)
    hu[8:24, 8:24, 8:24] = 0
    hu[12:20, 12:20, 12:20] = 3095
    hu[0, 0, 0] = -1200
    image = sitk.GetImageFromArray(hu)
    image.SetSpacing((0.7, 0.7, 0.7))
    image.SetOrigin((17., 21., 42.))
    sitk.WriteImage(image, str(raw / "case.nii.gz"))
    args = build_parser().parse_args([
        "--datapath=" + str(root), "--num=4", "--sad=50", "--sid=100",
        "--resolution=64", "--device=cpu",
    ])
    output = generate(args)
    assert output == root / "syn_data_leap"
    assert not (root / "syn_data").exists()
    case = output / "case"
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "gt_volume.nii.gz")))
    expected_gt = ct2mu(hu)
    expected_gt = np.clip(expected_gt, 0, expected_gt.max())
    np.testing.assert_array_equal(gt, expected_gt)
    projections = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "proj.nii.gz")))
    assert projections.shape == (4, 64, 64)
    assert np.isfinite(projections).all() and projections.max() > 0
    metadata = json.loads((case / "transforms.json").read_text())
    assert metadata["proj_resolution"] == [64, 64]
    np.testing.assert_allclose(metadata["volume_origin"], -np.array(metadata["volume_phy"]) / 2)
    np.testing.assert_allclose(metadata["proj_spacing"], [0.7 * 2 * 8, 0.7 * 2 * 8])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["case"]}))
    data_args = SimpleNamespace(angle_sampling="uniform", datadir=str(output),
                                split_path=str(split), start=0, end=360, nviews=2)
    dataset = LightningCBCTDataset(data_args, "train")
    item = dataset[0]
    np.testing.assert_array_equal(item["images"].numpy()[:, 0], projections[[0, 2]])
    np.testing.assert_array_equal(item["3Dvolume"].numpy(), gt)
    expected = np.array([metadata["frames"][index]["vec"] for index in [0, 2]], dtype=np.float32)
    np.testing.assert_array_equal(item["poses"].numpy(), expected)
    # Existing data needs an explicit replacement request.
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate(args)
