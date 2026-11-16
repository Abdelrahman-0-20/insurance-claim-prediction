"""Approved adjustment 1: frequency classifiers use exposure weights with natural
class priors -- no scale_pos_weight, no class_weight, no SMOTE anywhere."""
import io
import tokenize
from pathlib import Path

import numpy as np

from src.models.tune import build_estimator, fit_estimator

SRC = Path(__file__).resolve().parent.parent / "src"


def _py_files():
    return sorted(SRC.rglob("*.py"))


def _code_only(text: str) -> str:
    """Strip comments and string literals so docstrings may discuss the rules."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def test_no_cost_sensitive_weighting_in_source():
    for f in _py_files():
        code = _code_only(f.read_text())
        assert "scale_pos_weight" not in code, f"{f} must not use scale_pos_weight"
        assert "class_weight" not in code, f"{f} must not use class_weight"
        assert "SMOTE" not in code, f"{f} must not use SMOTE"


def test_frequency_config_has_no_class_weighting(cfg):
    for name, spec in cfg.models.items():
        if spec["task"] == "frequency":
            params = spec.get("params", {})
            assert "scale_pos_weight" not in params, name
            assert "class_weight" not in params, name


def test_sample_weight_is_applied():
    """fit_estimator must pass sample_weight through to the estimator."""
    rng = np.random.RandomState(0)
    X = rng.rand(300, 4)
    y = (X[:, 0] + rng.rand(300) > 0.7).astype(int)
    w = rng.uniform(0.5, 5.0, 300)

    est_a = build_estimator("glm", "frequency", {"C": 1.0, "max_iter": 500}, 42)
    fit_estimator(est_a, X, y, w, [], "glm")

    est_b = build_estimator("glm", "frequency", {"C": 1.0, "max_iter": 500}, 42)
    est_b.fit(X, y, sample_weight=w)   # reference: explicit sample_weight

    np.testing.assert_allclose(est_a.coef_, est_b.coef_, rtol=1e-6)

    # and weights must actually matter: without them the fit differs
    est_c = build_estimator("glm", "frequency", {"C": 1.0, "max_iter": 500}, 42)
    est_c.fit(X, y)
    assert not np.allclose(est_a.coef_, est_c.coef_)
