"""ARIMA/SARIMA: sai phân, ACF/PACF, grid search AIC, Ljung-Box, dự báo.

Khớp Lecture Notes Chương 4 — bao gồm cải tiến "bản đồ nhiệt AIC" (thay vì
chỉ liệt kê vài tổ hợp) và kiểm định Ljung-Box chính thức cho phần dư, lấy
cảm hứng từ tài liệu tham khảo "Mô hình ARIMA trong bài toán dự báo chuỗi
thời gian" (AI VIETNAM, 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ArimaFitResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    aic: float
    fitted: np.ndarray
    forecast_mean: np.ndarray | None = None
    forecast_ci_lower: np.ndarray | None = None
    forecast_ci_upper: np.ndarray | None = None
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


def adf_test(y: np.ndarray) -> dict[str, float | bool]:
    """Kiểm định ADF — H0: chuỗi có nghiệm đơn vị (không dừng)."""
    from statsmodels.tsa.stattools import adfuller

    y_clean = np.asarray(y, dtype=float)
    y_clean = y_clean[~np.isnan(y_clean)]
    if len(y_clean) < 8:
        raise ValueError("Cần tối thiểu 8 quan sát để chạy kiểm định ADF một cách có ý nghĩa.")

    result = adfuller(y_clean, autolag="AIC")
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags_used": int(result[2]),
        "n_obs": int(result[3]),
        "critical_1pct": float(result[4]["1%"]),
        "critical_5pct": float(result[4]["5%"]),
        "critical_10pct": float(result[4]["10%"]),
        "is_stationary_5pct": bool(result[1] < 0.05),
    }


def apply_differencing(y: np.ndarray, order: int) -> np.ndarray:
    """Sai phân bậc order (0, 1, hoặc 2)."""
    if order not in (0, 1, 2):
        raise ValueError("Chỉ hỗ trợ sai phân bậc 0, 1 hoặc 2.")
    result = np.asarray(y, dtype=float)
    for _ in range(order):
        result = np.diff(result)
    return result


def fit_sarima(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    future_steps: int = 0,
    ci_alpha: float = 0.2,
) -> ArimaFitResult:
    """Fit một mô hình SARIMA(X)(order)(seasonal_order) cụ thể."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_arr = np.asarray(y, dtype=float)
    p, d, q = order
    min_obs = max(p, q) + d + 5
    if len(y_arr) < min_obs:
        raise ValueError(f"Chuỗi quá ngắn cho order {order} — cần tối thiểu khoảng {min_obs} quan sát.")

    try:
        model = SARIMAX(
            y_arr, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
    except Exception as exc:
        raise ValueError(
            f"Không fit được SARIMA{order}{seasonal_order} — thử giảm bậc p,d,q hoặc kiểm tra dữ liệu."
        ) from exc

    forecast_mean = forecast_lo = forecast_hi = None
    if future_steps > 0:
        fc = model.get_forecast(steps=future_steps)
        forecast_mean = np.asarray(fc.predicted_mean)
        ci = fc.conf_int(alpha=ci_alpha)
        ci_arr = np.asarray(ci)
        forecast_lo, forecast_hi = ci_arr[:, 0], ci_arr[:, 1]

    return ArimaFitResult(
        order=order, seasonal_order=seasonal_order, aic=float(model.aic),
        fitted=np.asarray(model.fittedvalues), residuals=np.asarray(model.resid),
        forecast_mean=forecast_mean, forecast_ci_lower=forecast_lo, forecast_ci_upper=forecast_hi,
    )


def aic_grid_search(
    y: np.ndarray,
    p_max: int = 3,
    q_max: int = 3,
    d: int = 1,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[np.ndarray, tuple[int, int]]:
    """Quét toàn bộ lưới (p,q) trong [0,p_max] x [0,q_max], trả về ma trận AIC
    và toạ độ (p,q) có AIC nhỏ nhất. Dùng để vẽ bản đồ nhiệt.

    LƯU Ý QUAN TRỌNG: kết quả phụ thuộc trực tiếp vào p_max/q_max — phạm vi
    quét càng hẹp, càng dễ bỏ lỡ mô hình tốt hơn (xem Lecture Notes Chương 4).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    aic_grid = np.full((p_max + 1, q_max + 1), np.nan)
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                model = SARIMAX(
                    y, order=(p, d, q), seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                aic_grid[p, q] = model.aic
            except Exception:
                continue

    if np.all(np.isnan(aic_grid)):
        raise ValueError("Không có tổ hợp (p,q) nào hội tụ được — thử phạm vi khác hoặc kiểm tra dữ liệu.")

    best_idx = np.unravel_index(np.nanargmin(aic_grid), aic_grid.shape)
    return aic_grid, (int(best_idx[0]), int(best_idx[1]))


def ljung_box_test(residuals: np.ndarray, lags: list[int] | None = None, skip_initial: int = 0) -> pd.DataFrame:
    """Kiểm định Ljung-Box cho phần dư — H0: phần dư không tự tương quan (nhiễu trắng).
    p-value > 0.05 -> không đủ bằng chứng bác bỏ H0 -> phần dư chấp nhận được.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    resid_clean = np.asarray(residuals, dtype=float)
    resid_clean = resid_clean[skip_initial:]
    resid_clean = resid_clean[~np.isnan(resid_clean)]
    if len(resid_clean) < 8:
        raise ValueError("Phần dư quá ngắn để chạy kiểm định Ljung-Box có ý nghĩa.")

    if lags is None:
        max_lag = min(12, len(resid_clean) // 2)
        lags = [lag for lag in (6, 12) if lag <= max_lag] or [max_lag]

    return acorr_ljungbox(resid_clean, lags=lags, return_df=True)
