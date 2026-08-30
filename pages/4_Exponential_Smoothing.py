import streamlit as st
import numpy as np

from src.ui_helpers import select_series, show_series_preview
from src.naive_smoothing import fit_ses, fit_holt, fit_holt_winters, compare_all_holt_winters
from src.metrics import calculate_metrics
from src.plotting import line_chart, forecast_chart_with_ci
from src.app_state import register_method_mape

st.title("📉 Exponential Smoothing: SES · Holt · Holt-Winters")
st.caption("Chương 3 — càng nhiều tham số KHÔNG đồng nghĩa càng tốt; phải khớp đúng cấu trúc dữ liệu.")

result = select_series("expsmooth")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values
n = len(y)

method = st.selectbox("Chọn phương pháp", ["SES", "Holt", "Holt-Winters"])
future_steps = st.slider("Số kỳ dự báo tương lai", 0, 12, 6)

fitted = None
forecast = None
params = {}

try:
    if method == "SES":
        r = fit_ses(y, future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

    elif method == "Holt":
        damped = st.checkbox("Dùng damped trend (khuyến nghị)", value=True)
        r = fit_holt(y, damped=damped, future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

    else:  # Holt-Winters
        col1, col2, col3 = st.columns(3)
        with col1:
            season_length = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12)
        with col2:
            trend_type = st.selectbox("Xu hướng", ["add", "mul"], format_func=lambda x: "Cộng" if x == "add" else "Nhân")
        with col3:
            seasonal_type = st.selectbox("Mùa vụ", ["add", "mul"], format_func=lambda x: "Cộng" if x == "add" else "Nhân")
        r = fit_holt_winters(y, season_length=season_length, trend=trend_type, seasonal=seasonal_type,
                              future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

except ValueError as error:
    st.error(str(error))
    st.stop()

valid_mask = ~np.isnan(fitted)
m = calculate_metrics(y[valid_mask], fitted[valid_mask])
register_method_mape(method, m.mape, value_col)

col1, col2, col3 = st.columns(3)
col1.metric("MAPE", f"{m.mape:.2f}%")
col2.metric("MSE", f"{m.mse:,.0f}")
col3.metric("Tham số", ", ".join(f"{k}={v:.4f}" for k, v in params.items()) or "—")

if forecast is not None and future_steps > 0:
    future_index = np.arange(n, n + future_steps)
    fig = forecast_chart_with_ci(
        df[time_col], y, future_index, forecast, title=f"{method} — Dự báo {future_steps} kỳ tới",
        xaxis_title="Thời gian", yaxis_title=value_col,
    )
else:
    fig = line_chart(df[time_col], {value_col: y, method: fitted}, title=f"{method} — Khớp trong mẫu",
                      xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

if any(v < 0.05 for v in params.values() if isinstance(v, float)):
    st.info(
        "💡 Tham số làm mịn gần 0 KHÔNG phải lỗi — nghĩa là mô hình đang 'tin vào mức nền' hơn là "
        "chạy theo quan sát mới, thường xảy ra khi SKU gần như thuần nhiễu ngẫu nhiên (xem Lecture "
        "Notes Chương 3)."
    )

st.divider()
st.subheader("🧭 So sánh cả 4 tổ hợp Holt-Winters cùng lúc")
st.caption(
    "Khớp trực tiếp 4 góc của Ma trận Pegels (Chương 1) — thay vì chọn từng tổ hợp một, xem ngay "
    "tổ hợp nào có MSE thấp nhất cho đúng SKU đang chọn."
)

if st.button("🔍 Chạy cả 4 tổ hợp", type="primary"):
    with st.spinner("Đang fit 4 mô hình Holt-Winters..."):
        try:
            combo_season_length = season_length if method == "Holt-Winters" else 12
            st.session_state["hw4_results"] = compare_all_holt_winters(y, season_length=combo_season_length)
            st.session_state["hw4_season_length"] = combo_season_length
        except Exception as error:
            st.error(f"Không chạy được: {error}")

if "hw4_results" in st.session_state:
    combo_results = st.session_state["hw4_results"]
    ok_results = [r for r in combo_results if r["status"] == "ok"]

    if not ok_results:
        st.warning("Không có tổ hợp nào chạy thành công trên chuỗi này.")
    else:
        best = min(ok_results, key=lambda r: r["mse"])

        table_rows = []
        for r in combo_results:
            if r["status"] == "ok":
                table_rows.append({
                    "Xu hướng + Mùa vụ": r["label"], "MSE": f"{r['mse']:,.1f}",
                    "": "🏆 Tốt nhất" if r is best else "",
                })
            else:
                table_rows.append({"Xu hướng + Mùa vụ": r["label"], "MSE": "—",
                                    "": f"⚠ {r.get('reason', 'lỗi')}"})
        st.dataframe(table_rows, width="stretch", hide_index=True)

        best_r = best["result"]
        valid_best = ~np.isnan(best_r.fitted)
        m_best = calculate_metrics(y[valid_best], best_r.fitted[valid_best])
        register_method_mape(f"Holt-Winters ({best['label']})", m_best.mape, value_col)

        st.success(f"🏆 Tổ hợp tốt nhất: **{best['label']}** — MSE={best['mse']:,.1f}, MAPE={m_best.mape:.2f}%")

        fig_combo = line_chart(
            df[time_col], {value_col: y, f"Holt-Winters ({best['label']})": best_r.fitted},
            title=f"Tổ hợp tốt nhất: {best['label']}", xaxis_title="Thời gian", yaxis_title=value_col,
        )
        st.plotly_chart(fig_combo, width="stretch")

        st.caption(
            "💡 Nếu tổ hợp Nhân bị bỏ qua (⚠), nghĩa là chuỗi có giá trị ≤ 0 — mùa vụ Nhân không xác "
            "định được vì công thức nhân với mức nền."
        )
