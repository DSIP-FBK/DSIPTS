# Vendored from https://github.com/mistralai/timestral (Apache-2.0, see LICENSE in this folder),
# commit a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff (2026-10-15). Only change: imports made relative.
# Model weights are NOT included: they are downloaded from the HF Hub (Modified MIT license).
"""Standalone inference for Timestral time-series forecasting models."""

from .config import TimestralConfig
from .forecaster import Forecast, TimestralForecaster
from .loading import load_model
from .model import TimestralModel

__all__ = ["Forecast", "TimestralConfig", "TimestralForecaster", "TimestralModel", "load_model"]
__version__ = "1.0.0"
