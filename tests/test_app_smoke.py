"""Headless execution smoke test for the Streamlit app (AppTest runs the script)."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_renders_without_exception():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    # three focused tabs
    assert len(at.tabs) == 3, [t.label for t in at.tabs]
    labels = [t.label for t in at.tabs]
    assert labels == [
        "Single Policy Simulator",
        "Batch Scoring",
        "Model Diagnostics",
    ]


def test_app_form_and_kpi_flow():
    """Score a policy through the form and expect KPI metric cards."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if at.exception:
        pytest.fail(f"app raised: {at.exception}")
    # model artifacts must exist for the interactive flow
    if not (APP.parent.parent / "artifacts" / "models" / "champion.joblib").exists():
        pytest.skip("artifacts missing - run 'make train' first")
    submit = at.button[0]
    submit.click().run()
    assert not at.exception, f"app raised: {at.exception}"
    metric_labels = [m.label for m in at.metric]
    assert "Claim Probability" in metric_labels
    assert "Risk Tier" in metric_labels
    assert "Expected Pure Premium" in metric_labels
