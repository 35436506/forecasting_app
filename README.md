# Forecasting Lab — GDMN

Ứng dụng Streamlit đồng hành cùng khóa học **Forecasting Cho Chuỗi Cung Ứng**
(GDMN) — thực hành trực quan toàn bộ 7 phương pháp dự báo và các công cụ vận
hành đã học, dùng đúng bộ dữ liệu mẫu **ERP_4SKU** của khóa học.

Khác với bản "ARIMA Demo Lab" trước đây (chỉ có một trang, chỉ có ARIMA), app
này mở rộng thành 10 trang bao phủ toàn bộ giáo trình — Naive, Exponential
Smoothing, ARIMA/SARIMA, Regression, Safety Stock/Croston, Train/Test
Validation, và xử lý dữ liệu bẩn — đồng thời cải tiến phần ARIMA với bản đồ
nhiệt AIC và kiểm định Ljung-Box (tham khảo tài liệu *"Mô hình ARIMA trong
bài toán dự báo chuỗi thời gian"*, AI VIETNAM 2026).

## Cách chạy

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc project

```text
.
├── app.py                          # Điều hướng đa trang
├── pages/
│   ├── 1_Tong_Quan.py              # Ma trận Pegels, quy trình 6 bước
│   ├── 2_Phan_Loai_Du_Lieu.py      # CV, %kỳ=0, SLOPE
│   ├── 3_Baseline_Naive.py         # NF1, NF2, 7 chỉ số sai số (kể cả WMAPE)
│   ├── 4_Exponential_Smoothing.py  # SES, Holt, Holt-Winters
│   ├── 5_ARIMA_Lab.py              # ADF, ACF/PACF, heatmap AIC, Ljung-Box
│   ├── 6_Regression.py             # Trend/Quadratic/Dummy/Promo
│   ├── 7_Van_Hanh.py               # Safety Stock, ROP, Tracking Signal, Croston/SBA/TSB
│   ├── 8_Kiem_Tra_Mo_Hinh.py       # Train/Test Split, Walk-Forward Validation
│   ├── 9_Xu_Ly_Du_Lieu.py          # Outlier IQR/Z-score, nội suy dữ liệu thiếu
│   └── 10_So_Sanh_Tong_Hop.py      # Bảng xếp hạng MAPE mọi phương pháp
├── src/
│   ├── app_state.py                # Session state + registry so sánh phương pháp
│   ├── data_utils.py                # Nạp dữ liệu mẫu / CSV riêng
│   ├── ui_helpers.py                # Component chọn dữ liệu dùng chung
│   ├── metrics.py                   # ME, MAE, MSE, RMSE, MPE, MAPE, WMAPE
│   ├── pegels.py                    # Phân loại hình dạng nhu cầu
│   ├── naive_smoothing.py           # NF1, NF2, SES, Holt, Holt-Winters
│   ├── arima_modeling.py            # ADF, SARIMA, grid search AIC, Ljung-Box
│   ├── regression_modeling.py       # Linear/Quadratic/Seasonal Regression
│   ├── croston.py                   # Croston, SBA, TSB
│   ├── inventory.py                 # Safety Stock, ROP, chi phí giữ hàng
│   ├── validation.py                # Train/Test, Walk-Forward, Adjusted R²
│   ├── outliers.py                  # IQR, Z-score, nội suy
│   ├── diagnostics.py               # ACF/PACF
│   └── plotting.py                  # Các hàm vẽ Plotly dùng chung
├── data/
│   └── ERP_4SKU_sample.csv          # 4 SKU, 36 tháng — cùng bộ dữ liệu Lecture Notes dùng
├── assets/
│   └── pegels_3x3_grid.png
└── requirements.txt
```

## Nguyên tắc thiết kế

- **Mọi công thức khớp 1-1 với Lecture Notes và Python Instructions** của
  khóa học — dùng app để thực hành trực quan, rồi đối chiếu lại với Excel/
  Python đã làm trên lớp. Toàn bộ số liệu trong `src/` đã được kiểm chứng
  khớp với các con số MAPE/AIC/R² đã công bố trong Lecture Notes.
- **Tách biệt logic (`src/`) và giao diện (`pages/`)** — mỗi hàm trong `src/`
  có thể test độc lập bằng `python3 -c "from src.xxx import ...`, không cần
  chạy Streamlit.
- **Luôn chia dữ liệu THEO THỨ TỰ THỜI GIAN** — không có nơi nào trong app
  dùng `train_test_split()` ngẫu nhiên cho chuỗi thời gian (xem trang
  *Train/Test · Walk-Forward*).
- **Bảng So Sánh Tổng Hợp** tự động ghi nhận MAPE của mọi phương pháp đã
  chạy trong phiên làm việc — không cần copy tay từng con số.

## Dữ liệu đầu vào tự tải lên

CSV cần tối thiểu:

- một cột thời gian convert được sang `datetime`
- một cột giá trị convert được sang số

App sẽ tự động chuẩn hoá (convert datetime, sort tăng dần, ép kiểu số) ở mọi
trang thông qua `src/data_utils.py`.
