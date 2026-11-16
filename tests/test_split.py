"""Split determinism, stratification, and disjointness."""
import numpy as np

from src.data.split import split_hash, split_report, stratified_split


def test_split_is_deterministic(dataset, cfg):
    a = stratified_split(dataset, cfg.split, cfg.seed)
    b = stratified_split(dataset, cfg.split, cfg.seed)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)
    assert split_hash(a) == split_hash(b)


def test_split_differs_with_different_seed(dataset, cfg):
    a = stratified_split(dataset, cfg.split, 42)
    b = stratified_split(dataset, cfg.split, 7)
    assert not np.array_equal(a.train, b.train)


def test_split_is_disjoint_and_exhaustive(dataset, cfg):
    s = stratified_split(dataset, cfg.split, cfg.seed)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(all_idx) == len(dataset)
    assert len(np.unique(all_idx)) == len(dataset)


def test_split_preserves_stratification(dataset, cfg):
    s = stratified_split(dataset, cfg.split, cfg.seed)
    y = dataset.target.to_numpy()
    overall = y.mean()
    for part in (s.train, s.val, s.test):
        assert abs(y[part].mean() - overall) < 0.05


def test_split_report_shape(dataset, cfg):
    s = stratified_split(dataset, cfg.split, cfg.seed)
    rep = split_report(s, dataset, cfg.seed)
    assert rep["n_train"] + rep["n_val"] + rep["n_test"] == len(dataset)
    assert rep["seed"] == cfg.seed
    assert len(rep["hash"]) == 16
