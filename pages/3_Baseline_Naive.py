import streamlit as st
import numpy as np

from src.ui_helpers import select_series, show_series_preview
from src.naive_smoothing import naive_nf1, naive_nf2
from src.metrics import calculate_metrics
from src.plotting import line_chart
from src.app_state import register_method_mape

st.title("📏 Baseline: NF1, NF2 & 7 chỉ số sai số")
st.caption("Chương 2 — mọi phương pháp phức tạp hơn PHẢI đánh bại hai mốc chuẩn này mới đáng dùng.")

result = select_series("baseline")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values

season_length = st.slider("Chu kỳ mùa vụ dùng cho NF2 (số kỳ)", min_value=2, max_value=12, value=12)

nf1_result = naive_nf1(y)
try:
    nf2_result = naive_nf2(y, season_length=season_length)
except Exception:
    nf2_result = None

valid_mask = ~np.isnan(nf1_result.fitted)
m_nf1 = calculate_metrics(y[valid_mask], nf1_result.fitted[valid_mask])
register_method_mape("NF1", m_nf1.mape, value_col)

series_to_plot = {value_col: y, "NF1": nf1_result.fitted}
if nf2_result is not None:
    valid_mask2 = ~np.isnan(nf2_result.fitted)
    if valid_mask2.sum() > 0:
        m_nf2 = calculate_metrics(y[valid_mask2], nf2_result.fitted[valid_mask2])
        register_method_mape("NF2", m_nf2.mape, value_col)
        series_to_plot["NF2"] = nf2_result.fitted
    else:
        m_nf2 = None
else:
    m_nf2 = None

fig = line_chart(df[time_col], series_to_plot, title="NF1 vs NF2", xaxis_title="Thời gian",
                  yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Bảng 7 chỉ số sai số")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**NF1**")
    st.table(m_nf1.as_dict())
with col2:
    st.markdown("**NF2**")
    if m_nf2 is not None:
        st.table(m_nf2.as_dict())
    else:
        st.caption("Chuỗi chưa đủ dài hơn chu kỳ mùa vụ để tính NF2.")

with st.expander("Ý nghĩa từng chỉ số (Lecture Notes Chương 2)"):
    st.markdown(
        """
        - **ME (Forecast Bias)**: trung bình sai số CÓ DẤU — phát hiện thiên lệch hệ thống,
          nhưng dễ "nói dối" vì dương/âm triệt tiêu nhau.
        - **MAE**: trung bình trị tuyệt đối sai số — cùng đơn vị dữ liệu gốc.
        - **MSE / RMSE**: phạt nặng sai số lớn — MSE là đầu vào trực tiếp cho Safety Stock.
        - **MPE / MAPE**: sai số phần trăm — MAPE là chỉ số phổ biến nhất, nhưng "vỡ" khi có
          kỳ giá trị 0.
        - **WMAPE**: tổng sai số tuyệt đối chia tổng doanh số — ưu tiên khi các kỳ có quy mô
          doanh số chênh lệch lớn (xem Chương 2, mục WMAPE).
        """
    )
