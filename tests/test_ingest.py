"""Ingest normalization for both sources: local CSV and the OpenML fallback."""
import pandas as pd
import pytest

from src.data.ingest import load_local_claims, load_openml_claims


def _write(tmp_path, df, name="claims.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_local_loader_maps_exposure_and_drops_zero(tmp_path, cfg):
    df = pd.DataFrame({
        "policy_id": ["a", "b", "c"],
        "subscription_length": [5.0, 0.0, 8.0],
        "claim_status": [0, 1, 1],
        "customer_age": [40, 50, 60],
    })
    ds = load_local_claims(_write(tmp_path, df), cfg.data)
    assert len(ds) == 2                       # zero-exposure row dropped
    assert ds.dropped_zero_exposure == 1
    assert (ds.exposure > 0).all()
    assert set(ds.target.unique()) <= {0, 1}
    # approved adjustment 2: duration stays an input feature
    assert "subscription_length" in ds.features.columns
    assert "exposure" in ds.features.columns
    assert "claim_status" not in ds.features.columns


def test_local_loader_missing_file_mentions_fix(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="make setup"):
        load_local_claims(tmp_path / "absent.csv", cfg.data)


def test_local_loader_missing_target_is_actionable(tmp_path, cfg):
    df = pd.DataFrame({
        "policy_id": ["a", "b"],
        "subscription_length": [5.0, 6.0],
        "customer_age": [40, 50],
    })
    with pytest.raises(ValueError, match="claim_status"):
        load_local_claims(_write(tmp_path, df), cfg.data)


def test_local_loader_rejects_non_binary_target(tmp_path, cfg):
    df = pd.DataFrame({
        "policy_id": ["a", "b", "c"],
        "subscription_length": [5.0, 6.0, 7.0],
        "claim_status": [0, 2, 1],
        "customer_age": [40, 50, 60],
    })
    with pytest.raises(ValueError, match="binary"):
        load_local_claims(_write(tmp_path, df), cfg.data)


def test_openml_fallback_normalizes_and_joins(monkeypatch, cfg):
    freq = pd.DataFrame({
        "IDpol": [1, 2, 3],
        "Exposure": [1.0, 0.0, 2.0],
        "ClaimNb": [2, 1, 0],
        "Area": ["A", "B", "A"],
        "VehPower": [10, 12, 11],
    })
    sev = pd.DataFrame({
        "IDpol": [1, 1, 2],
        "ClaimAmount": [1000.0, 500.0, 250.0],
    })

    def fake_fetch(name):
        return freq.copy() if "freq" in str(name) else sev.copy()

    monkeypatch.setattr("src.data.ingest._fetch_openml", fake_fetch)
    ds = load_openml_claims(cfg.data)

    assert len(ds) == 2                       # zero-exposure policy 2 dropped
    assert (ds.exposure > 0).all()
    assert "Exposure" in ds.features.columns  # stays a feature (approved adjustment 2)
    amounts = dict(zip(ds.features["policy_id"], ds.claim_amount))
    targets = dict(zip(ds.features["policy_id"], ds.target))
    assert amounts["1"] == pytest.approx(1500.0)   # aggregated from freMTPL2sev
    assert amounts["3"] == pytest.approx(0.0)
    assert targets["1"] == 1                       # ClaimNb > 0
    assert targets["3"] == 0
    assert set(targets.values()) <= {0, 1}


def test_load_dataset_prefers_local_file(tmp_path, cfg, monkeypatch):
    from src.data.ingest import load_dataset
    df = pd.DataFrame({
        "policy_id": ["a", "b"],
        "subscription_length": [5.0, 6.0],
        "claim_status": [0, 1],
        "customer_age": [40, 50],
    })
    p = _write(tmp_path, df)

    class FakeCfg:
        data = dict(cfg.data)
        data["raw_path"] = str(p)

    def fail_fetch(name):
        raise AssertionError("OpenML fetch must not run when the local file exists")

    monkeypatch.setattr("src.data.ingest._fetch_openml", fail_fetch)
    ds = load_dataset(FakeCfg())
    assert len(ds) == 2
    assert ds.source == str(p)
