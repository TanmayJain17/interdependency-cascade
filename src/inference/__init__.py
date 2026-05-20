"""Inference utilities for the cascade GNN."""

from src.inference.predict_cascade import (
    predict_cascade,
    predict_cascade_for_scenario,
    load_depths_for_scenario,
)

__all__ = [
    "predict_cascade",
    "predict_cascade_for_scenario",
    "load_depths_for_scenario",
]