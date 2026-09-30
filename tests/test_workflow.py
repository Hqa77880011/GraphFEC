import numpy as np
import pytest
import torch
from PIL import Image

from graphfec.channels import sample_mask
from graphfec.config import Config
from graphfec.data import ManifestDataset, prepare, stratified_split
from graphfec.metrics import stratified_interval
from graphfec.models import build_model
from graphfec.training import task_loss


def test_holdout_is_disjoint_stratified_and_repeatable():
    labels = np.repeat(np.arange(5), 20)
    train, val = stratified_split(labels, 0.1, 17)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(100))
    np.testing.assert_array_equal(np.bincount(labels[val]), np.full(5, 2))
    assert (train, val) == stratified_split(labels, 0.1, 17)


@pytest.mark.parametrize("dataset", ["tinyimagenet", "imagenet100"])
def test_image_manifests_preserve_class_labels_and_subset_ids(tmp_path, dataset):
    classes = ["n0001", "n0002"]
    for label, name in enumerate(classes):
        folder = tmp_path / "train" / name / "images"
        folder.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (8, 8), color=(label * 100, i * 20, 0)).save(folder / f"{i}.JPEG")
    if dataset == "tinyimagenet":
        folder = tmp_path / "val" / "images"
        folder.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (8, 8)).save(folder / f"{i}.JPEG")
        (tmp_path / "val" / "val_annotations.txt").write_text(
            "0.JPEG\tn0002\t0\t0\t8\t8\n1.JPEG\tn0001\t0\t0\t8\t8\n"
        )
    else:
        for name in classes:
            folder = tmp_path / "val" / name
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(folder / "test.JPEG")
    cfg = Config(dataset=dataset, data_root=str(tmp_path), num_classes=2, image_size=8)
    prepare(cfg)
    train, val, test = [ManifestDataset(cfg, split) for split in ("train", "val", "test")]
    assert {row[0] for row in train.rows}.isdisjoint(row[0] for row in val.rows)
    assert [test[i][1] for i in range(2)] == ([1, 0] if dataset == "tinyimagenet" else [0, 1])
    assert torch.isfinite(test[0][0]).all()
    subset = ManifestDataset(cfg, "test", limit=1)
    _, label, original_id = subset[0]
    assert label == test[original_id][1]
    cfg.split_seed += 1
    with pytest.raises(ValueError, match="manifest disagree"):
        ManifestDataset(cfg, "train")


def test_loss_models_stationarity_burst_duration_and_endpoints():
    mask = sample_mask(20000, 100, 0.2, "gilbert", np.random.default_rng(5)).numpy()
    assert abs((~mask).mean() - 0.2) < 0.005
    bad = ~mask[:, :-1]
    p_bg = (bad & mask[:, 1:]).sum() / bad.sum()
    assert abs(p_bg - 0.25) < 0.005
    for channel in ("gilbert", "bernoulli"):
        assert sample_mask(10, 20, 0, channel, np.random.default_rng(1)).all()
    assert not sample_mask(10, 20, 1, "bernoulli", np.random.default_rng(1)).any()


def test_paired_bootstrap_keeps_pairing_and_known_difference():
    labels = np.repeat([0, 1], 10)
    assert stratified_interval(np.zeros(20), labels, 100) == [0.0, 0.0]
    assert stratified_interval(np.ones(20), labels, 100) == pytest.approx([1.0, 1.0])
    values = np.tile([0.0, 1.0], 10)
    low, high = stratified_interval(values, labels, 200)
    assert 0 <= low < values.mean() < high <= 1


@pytest.mark.parametrize(
    "backbone,size,stage,shape",
    [
        ("resnet18", 32, 1, (64, 32, 32)),
        ("resnet18", 32, 2, (128, 16, 16)),
        ("resnet18", 32, 3, (256, 8, 8)),
        ("mobilenet_v3_large", 64, 2, (40, 8, 8)),
        ("convnext_tiny", 224, 2, (192, 28, 28)),
    ],
)
def test_backbone_split_shape_and_edge_freezing(backbone, size, stage, shape):
    torch.set_num_threads(2)
    cfg = Config(backbone=backbone, image_size=size, split_stage=stage, num_classes=7)
    model = build_model(cfg).eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        z = model.edge(x)
        assert tuple(z.shape[1:]) == shape
    model.requires_grad_(False)
    model.enable_final_stage()
    assert not any(p.requires_grad for p in model.edge.parameters())
    assert any(p.requires_grad for p in model.cloud.parameters())


def test_task_only_ablation_removes_auxiliary_losses_exactly():
    cfg = Config()
    z = torch.randn(2, 4, 2, 2)
    logits = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    actual = task_loss(z + 1, z, logits, logits.detach(), labels, cfg, task_only=True)
    torch.testing.assert_close(actual, ce)
    full = task_loss(z + 1, z, logits, logits.detach(), labels, cfg)
    assert full > actual
