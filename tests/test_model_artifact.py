from __future__ import annotations

import joblib

from src.config import MODEL_PATH, WEATHER_INFORMED_FEATURES


def test_model_artifact_loads_with_the_pinned_runtime() -> None:
    bundle = joblib.load(MODEL_PATH)

    assert bundle["selected_model"] == "hgb_weather_informed"
    assert bundle["feature_columns"] == WEATHER_INFORMED_FEATURES
    assert set(bundle["conformal_radii_mwh"]) == {"p80", "p95"}
    assert bundle["runtime"]["scikit_learn"] == "1.7.2"
