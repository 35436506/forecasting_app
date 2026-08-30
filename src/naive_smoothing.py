"""NF1, NF2 và họ Exponential Smoothing: SES, Holt, Holt-Winters.

Khớp Lecture Notes Chương 2 (Naive) và Chương 3 (Exponential Smoothing).
Mỗi hàm trả về DataFrame có cột fitted (trong mẫu) và, khi có future_steps,
thêm bảng dự báo ra tương lai.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SmoothingResult:
    method_name: str
    fitted: np.ndarray
    params: dict[str, float]
    forecast: np.ndarray | None = None


def naive_nf1(y: np.ndarray) -> SmoothingResult:
    """NF1: F(t+1) = Y(t) — mốc chuẩn tối thiểu, không có tham số."""
    fitted = np.array([np.nan] + list(y[:-1]))
    return SmoothingResult(method_name="NF1 (Naive đơn giản)", fitted=fitted, params={})


def naive_nf2(y: np.ndarray, season_length: int = 12) -> SmoothingResult:
    """NF2: dự báo bằng giá trị cùng kỳ năm trước, điều chỉnh theo xu hướng gần đây.
    Công thức đơn giản hoá dùng trong khóa: F(t) = Y(t-s) + [Y(t-1) - Y(t-1-s)].
    """
    n = len(y)
    fitted = np.full(n, np.nan)
    for t in range(season_length, n):
        seasonal_base = y[t - season_length]
        recent_trend = y[t - 1] - y[t - 1 - season_length] if t - 1 - season_length >= 0 else 0.0
        fitted[t] = seasonal_base + recent_trend
    return SmoothingResult(method_name="NF2 (Naive mùa vụ)", fitted=fitted, params={"season_length": season_length})


def fit_ses(y: np.ndarray, future_steps: int = 0) -> SmoothingResult:
    """Simple Exponential Smoothing — dùng statsmodels, alpha tối ưu tự động."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    if len(y) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để fit SES.")

    model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name="SES",
        fitted=np.asarray(model.fittedvalues),
        params={"alpha": float(model.params["smoothing_level"])},
        forecast=np.asarray(forecast) if forecast is not None else None,
    )


def fit_holt(y: np.ndarray, damped: bool = True, future_steps: int = 0) -> SmoothingResult:
    """Holt (Double Exponential Smoothing) — có xu hướng, không mùa vụ."""
    from statsmodels.tsa.holtwinters import Holt

    if len(y) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để fit Holt.")

    model = Holt(y, damped_trend=damped, initialization_method="estimated").fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name="Holt" + (" (damped)" if damped else ""),
        fitted=np.asarray(model.fittedvalues),
        params={
            "alpha": float(model.params["smoothing_level"]),
            "beta": float(model.params["smoothing_trend"]),
        },
        forecast=np.asarray(forecast) if forecast is not None else None,
    )


def fit_holt_winters(
    y: np.ndarray,
    season_length: int = 12,
    trend: str = "add",
    seasonal: str = "mul",
    future_steps: int = 0,
) -> SmoothingResult:
    """Holt-Winters — xu hướng + mùa vụ. trend/seasonal nhận 'add' hoặc 'mul'."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    min_required = season_length * 2
    if len(y) < min_required:
        raise ValueError(
            f"Holt-Winters cần tối thiểu {min_required} quan sát (2 chu kỳ mùa vụ {season_length} kỳ), "
            f"chuỗi hiện chỉ có {len(y)}."
        )

    model = ExponentialSmoothing(
        y, seasonal_periods=season_length, trend=trend, seasonal=seasonal,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name=f"Holt-Winters ({trend}+{seasonal})",
        fitted=np.asarray(model.fittedvalues),
        params={
            "alpha": float(model.params["smoothing_level"]),
            "beta": float(model.params["smoothing_trend"]),
            "gamma": float(model.params["smoothing_seasonal"]),
        },
        forecast=np.asarray(forecast) if forecast is not None else None,
    )
