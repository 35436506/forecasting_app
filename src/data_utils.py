"""Nạp dữ liệu: bộ mẫu ERP_4SKU của khóa học, hoặc CSV người dùng tự tải lên."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

SAMPLE_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "ERP_4SKU_sample.csv"

SAMPLE_SKU_INFO = {
    "NoiComDien": {
        "label": "Nồi cơm điện 1.8L",
        "shape": "Mùa vụ – Xu hướng",
        "note": "Case chính của khóa học — mùa vụ NHÂN, đỉnh Tết/Q4.",
    },
    "BepTuDoi": {
        "label": "Bếp từ đôi",
        "shape": "Ổn định",
        "note": "CV rất thấp, gần như nhiễu thuần túy — SES với alpha≈0 là tối ưu.",
    },
    "PhuTungX12": {
        "label": "Phụ tùng thay thế X12",
        "shape": "Rời rạc",
        "note": "61% số tháng bằng 0 — cần Croston/SBA/TSB, không dùng Exponential Smoothing thường.",
    },
    "AoKhoacMuaDong": {
        "label": "Áo khoác mùa đông (mới)",
        "shape": "Vòng đời ngắn",
        "note": "Ra mắt giữa kỳ — đường cong tăng trưởng-chững lại, hợp với Quadratic Trend.",
    },
}


def load_sample_dataset() -> pd.DataFrame:
    """Nạp bộ dữ liệu mẫu ERP_4SKU (36 tháng, 4 SKU) đi kèm khóa học."""
    try:
        df = pd.read_csv(SAMPLE_DATASET_PATH, parse_dates=["Thang"])
    except Exception as exc:
        raise ValueError("Không thể đọc bộ dữ liệu mẫu trong thư mục data/.") from exc
    if df.empty:
        raise ValueError("Bộ dữ liệu mẫu trống.")
    return df


def load_csv_data(file_obj: BinaryIO) -> pd.DataFrame:
    """Nạp file CSV do người dùng tải lên."""
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError("Không thể đọc file CSV — kiểm tra lại định dạng file.") from exc
    if df.empty:
        raise ValueError("File CSV không có dữ liệu.")
    return df


def prepare_time_series(df: pd.DataFrame, time_column: str, value_column: str) -> pd.DataFrame:
    """Chuẩn hoá cột thời gian sang datetime, sắp xếp tăng dần, ép cột giá trị
    sang số — CHUẨN chung cho mọi trang trong app."""
    if time_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột thời gian '{time_column}'.")
    if value_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột giá trị '{value_column}'.")
    if time_column == value_column:
        raise ValueError("Cột thời gian và cột giá trị phải khác nhau.")

    result = df[[time_column, value_column]].copy()
    result[time_column] = pd.to_datetime(
        result[time_column].astype("string").str.strip(), errors="coerce", format="mixed",
    )
    if result[time_column].isna().all():
        raise ValueError("Không thể chuyển cột thời gian sang định dạng ngày tháng.")

    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.dropna(subset=[time_column]).sort_values(by=time_column).reset_index(drop=True)
    return result


def month_numbers_from_dates(dates: pd.Series) -> "pd.Series[int]":
    """Trích tháng (1-12) từ cột datetime — dùng cho biến giả mùa vụ."""
    return pd.to_datetime(dates).dt.month


def export_comparison_to_excel(sku_label: str, methods: list[str], mapes: list[float],
                                raw_df: "pd.DataFrame | None" = None,
                                time_col: str | None = None, value_col: str | None = None) -> bytes:
    """Xuat bang xep hang phuong phap (va du lieu goc neu co) ra file Excel,
    tra ve bytes de dua vao st.download_button."""
    import io as _io

    buf = _io.BytesIO()
    sorted_pairs = sorted(zip(methods, mapes), key=lambda x: x[1])
    ranking_df = pd.DataFrame({
        "Phương pháp": [p[0] for p in sorted_pairs],
        "MAPE (%)": [round(p[1], 2) for p in sorted_pairs],
        "Xếp hạng": range(1, len(sorted_pairs) + 1),
    })

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ranking_df.to_excel(writer, sheet_name="Bang xep hang", index=False)
        if raw_df is not None and time_col is not None and value_col is not None:
            raw_df[[time_col, value_col]].to_excel(writer, sheet_name="Du lieu goc", index=False)

    return buf.getvalue()
