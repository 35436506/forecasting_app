import streamlit as st
import pandas as pd
import numpy as np
import io
import calendar
import warnings
warnings.filterwarnings('ignore')

# ── Lazy-load heavy libraries once per app lifecycle ─────────────────────────
@st.cache_resource(show_spinner=False)
def _load_heavy_libs():
    import statsmodels.api as _sm
    from statsmodels.tsa.seasonal import seasonal_decompose as _sd
    from statsmodels.tsa.api import ExponentialSmoothing as _ES
    from statsmodels.tsa.stattools import adfuller as _adf
    from sklearn.metrics import mean_squared_error as _mse
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    return _sm, _sd, _ES, _adf, _mse, _plt

sm, seasonal_decompose, ExponentialSmoothing, adfuller, mean_squared_error, plt = _load_heavy_libs()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecasting Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette (matches Regression Analyst dark theme) ────────────────────
DARK   = '#0d1117'
PANEL  = '#161b22'
GRID   = '#30363d'
WHITE  = '#e6edf3'
GRAY   = '#8b949e'
BLUE   = '#58a6ff'
GREEN  = '#3fb950'
RED    = '#f85149'
YELLOW = '#d29922'
PURPLE = '#bc8cff'
PINK   = '#f778ba'

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }
h1,h2,h3 { font-family: 'Space Mono', monospace; color: #e6edf3; }

[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
.stDataFrame td, .stDataFrame th { color: #e6edf3 !important; background: #161b22 !important; }
.stSelectbox div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
.stMultiSelect div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="option"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="popover"] { background: #161b22 !important; }
.stTextInput input, .stTextArea textarea { color: #e6edf3 !important; background: #161b22 !important; }
div[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

.hero-title {
    font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#58a6ff,#bc8cff,#f778ba);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2;
}
.hero-sub { color:#8b949e; font-size:1rem; margin-bottom:1.5rem; }

.section-hdr {
    font-family:'Space Mono',monospace; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:2px; color:#58a6ff; margin-bottom:0.8rem;
    border-bottom:1px solid #21262d; padding-bottom:0.5rem;
}
.card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.card-accent { border-left:4px solid #58a6ff; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem;
         font-weight:600; font-family:'Space Mono',monospace; margin-right:4px; }
.badge-blue   { background:#1f3a5f; color:#58a6ff; }
.badge-green  { background:#1a3a2a; color:#3fb950; }
.badge-yellow { background:#3a2d10; color:#d29922; }
.badge-red    { background:#3d1f1f; color:#f85149; }
.badge-purple { background:#2d1f5f; color:#bc8cff; }

.warn-box { background:#2a1f0a; border:1px solid #d29922; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#d29922; font-size:0.88rem; }
.ok-box   { background:#0a2a14; border:1px solid #3fb950; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#3fb950; font-size:0.88rem; }
.err-box  { background:#2a0a0a; border:1px solid #f85149; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#f85149; font-size:0.88rem; }
.info-box { background:#0a1a2a; border:1px solid #58a6ff; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#a8d8ff; font-size:0.88rem; }

.interpret-box {
    background:#1c2333; border:1px solid #58a6ff; border-radius:10px;
    padding:1.1rem 1.3rem; margin:0.8rem 0; color:#e6edf3;
    font-size:0.88rem; line-height:1.8;
}

.stButton>button {
    background:linear-gradient(90deg,#1f3a5f,#2d4a7a); color:#58a6ff; border:1px solid #58a6ff;
    border-radius:8px; font-family:'Space Mono',monospace; font-weight:700;
    padding:0.5rem 1.2rem; transition:all 0.2s;
}
.stButton>button:hover { background:linear-gradient(90deg,#2d4a7a,#3a5a9a); opacity:0.9; }

div[data-testid="stExpander"] { background:#161b22; border:1px solid #30363d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — data loading
# ══════════════════════════════════════════════════════════════════════════════

def detect_header_row(raw_df, max_scan=10):
    n_cols = raw_df.shape[1]
    scan = min(max_scan, raw_df.shape[0] - 1)
    for i in range(scan):
        row = raw_df.iloc[i]
        text_count = sum(
            1 for v in row
            if v is not None and not (isinstance(v, float) and np.isnan(v))
            and not isinstance(v, (int, float, np.integer, np.floating))
        )
        text_ratio = text_count / n_cols if n_cols else 0
        if i + 1 < raw_df.shape[0]:
            next_row = raw_df.iloc[i + 1]
            num_count = sum(1 for v in next_row if isinstance(v, (int, float, np.integer, np.floating))
                            and not (isinstance(v, float) and np.isnan(v)))
            num_ratio = num_count / n_cols if n_cols else 0
        else:
            num_ratio = 0
        if text_ratio >= 0.5 and num_ratio >= 0.4:
            return i
    return 0


@st.cache_data(show_spinner="Đang đọc file...")
def _load_data_core(file_bytes: bytes, file_name: str, chosen_sheet=None):
    name = file_name.lower()
    header_row = 0
    if name.endswith(".csv"):
        buf = io.BytesIO(file_bytes)
        try:
            raw = pd.read_csv(buf, header=None, nrows=15)
        except Exception:
            buf = io.BytesIO(file_bytes)
            raw = pd.read_csv(buf, header=None, nrows=15, encoding='latin1')
        header_row = detect_header_row(raw)
        buf = io.BytesIO(file_bytes)
        try:
            df = pd.read_csv(buf, header=header_row)
        except Exception:
            buf = io.BytesIO(file_bytes)
            df = pd.read_csv(buf, header=header_row, encoding='latin1')
    elif name.endswith((".xlsx", ".xls")):
        buf = io.BytesIO(file_bytes)
        xf = pd.ExcelFile(buf)
        sheet = chosen_sheet or xf.sheet_names[0]
        buf = io.BytesIO(file_bytes)
        raw = pd.read_excel(buf, sheet_name=sheet, header=None, nrows=15)
        header_row = detect_header_row(raw)
        buf = io.BytesIO(file_bytes)
        df = pd.read_excel(buf, sheet_name=sheet, header=header_row)
    else:
        return None, 0

    df = df.dropna(how='all').reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted
    return df, header_row


def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    chosen_sheet = None
    sheet_names = []
    if name.endswith((".xlsx", ".xls")):
        xf = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = xf.sheet_names
        if len(sheet_names) > 1:
            chosen_sheet = st.sidebar.selectbox("📑 Sheet", sheet_names)
        else:
            chosen_sheet = sheet_names[0]
    result = _load_data_core(file_bytes, uploaded_file.name, chosen_sheet)
    if result[0] is None:
        st.error("Không hỗ trợ định dạng file này.")
        return None
    df, _ = result
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — date parsing & series construction
# ══════════════════════════════════════════════════════════════════════════════

MONTH_MAP = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTH_MAP.update({m.upper(): i for i, m in enumerate(calendar.month_name) if m})

PERIOD_LABELS = {12: 'Tháng', 4: 'Quý', 52: 'Tuần', 7: 'Ngày trong tuần', 1: 'Kỳ'}


def try_parse_date_column(col):
    """Robustly parse a column of date-like values into a DatetimeIndex-compatible Series."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col)
    try:
        out = pd.to_datetime(col, errors='raise')
        return out
    except Exception:
        pass

    def parse_one(v):
        s = str(v).strip().upper().replace('-', ' ').replace('/', ' ').replace('.', ' ')
        parts = [p for p in s.split() if p]
        if len(parts) == 2:
            a, b = parts
            if a.isdigit() and b in MONTH_MAP:
                return pd.Timestamp(year=int(a), month=MONTH_MAP[b], day=1)
            if b.isdigit() and a in MONTH_MAP:
                return pd.Timestamp(year=int(b), month=MONTH_MAP[a], day=1)
            if a.isdigit() and b.isdigit() and len(a) == 4:
                try:
                    return pd.Timestamp(year=int(a), month=int(b), day=1)
                except Exception:
                    return pd.NaT
        return pd.NaT

    out = col.apply(parse_one)
    if out.notna().sum() / max(len(out), 1) > 0.8:
        return out
    return pd.to_datetime(col, errors='coerce')


def infer_period_and_freq(idx):
    """Infer seasonal period (e.g. 12 for monthly) and a pandas frequency string."""
    if len(idx) < 3:
        return 12, 'MS'
    diffs = (idx[1:] - idx[:-1]).days
    median_days = float(np.median(diffs))
    if 27 <= median_days <= 31:
        return 12, 'MS'
    if 85 <= median_days <= 95:
        return 4, 'QS'
    if 360 <= median_days <= 370:
        return 1, 'AS'
    if 6 <= median_days <= 8:
        return 52, 'W'
    if median_days == 1:
        return 7, 'D'
    return 12, 'MS'


def build_series(df, date_col, value_col):
    dates = try_parse_date_column(df[date_col])
    s = pd.Series(pd.to_numeric(df[value_col], errors='coerce').values, index=dates, name=value_col)
    s = s[~s.index.isna()]
    s = s.dropna()
    s = s[~s.index.duplicated(keep='last')]
    s = s.sort_index()
    return s


def auto_detect_columns(df):
    """Return (best_date_col, candidate_value_cols)."""
    best_date_col = None
    best_score = -1
    for col in df.columns:
        parsed = try_parse_date_column(df[col])
        score = parsed.notna().sum() / max(len(df), 1)
        if score > best_score:
            best_score = score
            best_date_col = col
    numeric_cols = []
    for col in df.columns:
        if col == best_date_col:
            continue
        conv = pd.to_numeric(df[col], errors='coerce')
        if conv.notna().sum() / max(len(df), 1) > 0.7:
            numeric_cols.append(col)
    return best_date_col, numeric_cols


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — plotting (dark theme)
# ══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=GRAY, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=9, labelpad=6)
    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)


def new_fig(nrows=1, ncols=1, figsize=(10, 4), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(DARK)
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            ax.set_facecolor(PANEL)
    else:
        axes.set_facecolor(PANEL)
    return fig, axes


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — decomposition
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_decompositions(series, period):
    out = {}
    series_mean = float(series.mean())
    for mode in ['additive', 'multiplicative']:
        if mode == 'multiplicative' and (series <= 0).any():
            out[mode] = {'status': 'skip', 'reason': 'Chuỗi có giá trị ≤ 0, không áp dụng được mô hình nhân.'}
            continue
        try:
            res = seasonal_decompose(series, model=mode, period=period, extrapolate_trend='freq')
            resid = res.resid.dropna()
            resid_mse_raw = float((resid ** 2).mean())
            resid_std = float(resid.std())
            # Additive residuals are absolute (centered at 0); multiplicative residuals are
            # ratios (centered at 1). Raw MSE is therefore NOT comparable across the two modes.
            # We normalize both to a "% of series level" scale for a fair comparison:
            #   additive:        resid_std / mean(series)
            #   multiplicative:  resid_std  (already a ratio, ~fraction of level)
            norm_resid_pct = (resid_std / series_mean * 100) if mode == 'additive' else (resid_std * 100)
            out[mode] = {
                'status': 'ok',
                'trend': res.trend, 'seasonal': res.seasonal, 'resid': res.resid,
                'resid_mse': resid_mse_raw,
                'resid_std': resid_std,
                'resid_pct': norm_resid_pct,
            }
        except Exception as e:
            out[mode] = {'status': 'error', 'reason': str(e)}
    return out


def seasonal_pivot(series, period):
    tmp = series.reset_index()
    tmp.columns = ['date', 'value']
    tmp['year'] = tmp['date'].dt.year
    if period == 12:
        tmp['p'] = tmp['date'].dt.month
        xlabel = 'Tháng'
    elif period == 4:
        tmp['p'] = tmp['date'].dt.quarter
        xlabel = 'Quý'
    else:
        tmp['p'] = (np.arange(len(tmp)) % period) + 1
        xlabel = 'Kỳ'
    pivot = tmp.pivot_table(index='p', columns='year', values='value', aggfunc='mean')
    return pivot, xlabel


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Exponential Smoothing (4 Holt-Winters combinations)
# ══════════════════════════════════════════════════════════════════════════════

HW_COMBOS = [
    ('Model 1', 'add', 'add'),
    ('Model 2', 'add', 'mul'),
    ('Model 3', 'mul', 'add'),
    ('Model 4', 'mul', 'mul'),
]


def run_hw_models(series, period):
    results = []
    positive = (series > 0).all()
    for name, trend, seasonal in HW_COMBOS:
        if (trend == 'mul' or seasonal == 'mul') and not positive:
            results.append({'name': name, 'trend': trend, 'seasonal': seasonal, 'status': 'skip',
                             'reason': 'cần dữ liệu > 0 cho mô hình nhân'})
            continue
        try:
            fit = ExponentialSmoothing(series, seasonal_periods=period, trend=trend, seasonal=seasonal).fit()
            mse_val = float(mean_squared_error(series, fit.fittedvalues))
            results.append({
                'name': name, 'trend': trend, 'seasonal': seasonal, 'status': 'ok', 'mse': mse_val,
                'alpha': float(fit.params['smoothing_level']),
                'beta': float(fit.params['smoothing_trend']) if fit.params['smoothing_trend'] is not None else np.nan,
                'gamma': float(fit.params['smoothing_seasonal']) if fit.params['smoothing_seasonal'] is not None else np.nan,
                'fit': fit,
            })
        except Exception as e:
            results.append({'name': name, 'trend': trend, 'seasonal': seasonal, 'status': 'error', 'reason': str(e)})
    return results


def label_combo(trend, seasonal):
    t = {'add': 'cộng', 'mul': 'nhân'}.get(trend, trend)
    s = {'add': 'cộng', 'mul': 'nhân'}.get(seasonal, seasonal)
    return f"Xu hướng {t} / Mùa vụ {s}"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — SARIMA model search
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SARIMA_CANDIDATES = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 2), (2, 1, 0), (1, 1, 2), (2, 1, 1), (0, 1, 0)]


def adf_pvalue(series):
    try:
        _, pvalue, *_ = adfuller(series.dropna())
        return float(pvalue)
    except Exception:
        return np.nan


def determine_d_D(series, period):
    p0 = adf_pvalue(series)
    if p0 <= 0.05:
        return 0, 0, {'original': p0}
    seas_diff = series.diff(period).dropna()
    p1 = adf_pvalue(seas_diff)
    if p1 <= 0.05:
        return 0, 1, {'original': p0, 'seasonal_diff': p1}
    both = seas_diff.diff(1).dropna()
    p2 = adf_pvalue(both)
    return 1, 1, {'original': p0, 'seasonal_diff': p1, 'seasonal_first_diff': p2}


@st.cache_data(show_spinner=False)
def sarima_grid_search(series, period, d, D, candidates):
    results = []
    for (p, _d, q) in candidates:
        order = (p, d, q)
        seasonal_order = (p, D, q, period)
        try:
            mod = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                             enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            start = period + d + 1
            fitted = res.fittedvalues.iloc[start:]
            actual = series.iloc[start:]
            mse_val = float(mean_squared_error(actual, fitted)) if len(fitted) > 0 else np.nan
            results.append({'order': order, 'seasonal_order': seasonal_order, 'aic': float(res.aic),
                             'mse': mse_val, 'status': 'ok'})
        except Exception as e:
            results.append({'order': order, 'seasonal_order': seasonal_order, 'aic': np.nan, 'mse': np.nan,
                             'status': 'error', 'reason': str(e)})
    return results


def fit_sarima(series, order, seasonal_order):
    mod = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                     enforce_stationarity=False, enforce_invertibility=False)
    return mod.fit(disp=False)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — multivariable regression forecasting (optional)
# ══════════════════════════════════════════════════════════════════════════════

def build_dummies(index, period):
    if period == 12:
        p = index.month
    elif period == 4:
        p = index.quarter
    else:
        p = (np.arange(len(index)) % period) + 1
    dummies = pd.DataFrame(index=index)
    for k in range(1, period):
        dummies[f'D{k}'] = (p == k).astype(int)
    return dummies


def make_future_index(last_date, periods, freq):
    try:
        return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    except Exception:
        return pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT TO EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def export_results_excel(series, hw_results, best_hw, sarima_results, best_sarima_res, best_sarima_order,
                          hw_forecast_vals, sarima_forecast_vals, horizon, freq,
                          backtest_table=None, reg_forecast=None):
    buf = io.BytesIO()
    future_index = make_future_index(series.index[-1], horizon, freq)
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Forecast sheet
        fc_df = pd.DataFrame(index=future_index)
        if hw_forecast_vals is not None:
            fc_df['Holt-Winters'] = hw_forecast_vals.values
        if sarima_forecast_vals is not None:
            fc_df['SARIMA'] = sarima_forecast_vals.values
        if reg_forecast is not None:
            fc_df['Regression'] = reg_forecast.values
        fc_df.index.name = 'Date'
        fc_df.to_excel(writer, sheet_name='Forecast')

        # HW comparison
        hw_rows = []
        for r in hw_results:
            row = {'Model': r['name'], 'Trend': r['trend'], 'Seasonal': r['seasonal'], 'Status': r['status']}
            if r['status'] == 'ok':
                row.update({'alpha': round(r['alpha'], 4), 'beta': round(r['beta'], 4) if not np.isnan(r['beta']) else None,
                             'gamma': round(r['gamma'], 4) if not np.isnan(r['gamma']) else None, 'MSE': round(r['mse'], 4)})
            else:
                row['Note'] = r.get('reason', '')
            hw_rows.append(row)
        pd.DataFrame(hw_rows).to_excel(writer, sheet_name='HW Model details', index=False)

        # SARIMA comparison
        sarima_rows = []
        for r in sarima_results:
            sarima_rows.append({
                'order (p,d,q)': str(r['order']), 'seasonal_order (P,D,Q,s)': str(r['seasonal_order']),
                'AIC': round(r['aic'], 2) if not np.isnan(r['aic']) else None,
                'MSE': round(r['mse'], 4) if not np.isnan(r['mse']) else None, 'Status': r['status'],
            })
        pd.DataFrame(sarima_rows).to_excel(writer, sheet_name='SARIMA model details', index=False)

        # Backtest
        if backtest_table is not None:
            backtest_table.to_excel(writer, sheet_name='Backtest comparison')

        # Original data
        series.to_frame(name=series.name or 'value').to_excel(writer, sheet_name='Data')

    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI — HEADER & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">📈 Forecasting Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Tải lên một file Excel/CSV chứa chuỗi thời gian — app sẽ tự phân rã, '
             'tìm mô hình San bằng số mũ (Holt-Winters) và SARIMA tốt nhất, kiểm định bằng backtest, '
             'và đưa ra dự báo kèm phân tích.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-hdr">① TẢI DỮ LIỆU</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Chọn file Excel hoặc CSV", type=["xlsx", "xls", "csv"])

    # ── Sample data downloads ───────────────────────────────────────────────
    with st.expander("📂 Tải file dữ liệu mẫu"):
        st.markdown("**📦 Dữ liệu mẫu theo phương pháp**")

        SAMPLE_FILES = [
            # (label, path, filename, mime)
            ("🏠 Doanh số đồ gia dụng (Additive)",
             "/mnt/user-data/uploads/01_Doanh_so_do_gia_dung_Additive.xlsx",
             "Doanh_so_do_gia_dung_Additive.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("⚡ Sản lượng điện (Multiplicative)",
             "/mnt/user-data/uploads/02_San_luong_dien_Multiplicative.xlsx",
             "San_luong_dien_Multiplicative.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("🏢 Giá thuê văn phòng (Regression)",
             "/mnt/user-data/uploads/03_Gia_thue_van_phong_Regression.xlsx",
             "Gia_thue_van_phong_Regression.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("✈️ Khách du lịch – có COVID (Dữ liệu xáo trộn)",
             "/mnt/user-data/uploads/04_Khach_du_lich_Messy_COVID.xlsx",
             "Khach_du_lich_Messy_COVID.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]
        for label, fpath, fname, mime in SAMPLE_FILES:
            try:
                with open(fpath, "rb") as f:
                    st.download_button(label=label, data=f.read(), file_name=fname,
                                       mime=mime, use_container_width=True, key=f"dl_{fname}")
            except Exception:
                st.caption(f"⚠ Không tìm thấy: {fname}")

        st.markdown("**🌍 Dữ liệu quốc tế (CSV)**")
        CSV_FILES = [
            ("🛫 Air Passengers – Hành khách hàng không (Box-Jenkins kinh điển)",
             "/mnt/user-data/uploads/AirPassengers.csv",
             "AirPassengers.csv"),
            ("🌡️ Daily Min Temperatures – Nhiệt độ tối thiểu hàng ngày (Melbourne)",
             "/mnt/user-data/uploads/daily-min-temperatures.csv",
             "daily-min-temperatures.csv"),
        ]
        for label, fpath, fname in CSV_FILES:
            try:
                with open(fpath, "rb") as f:
                    st.download_button(label=label, data=f.read(), file_name=fname,
                                       mime="text/csv", use_container_width=True, key=f"dl_{fname}")
            except Exception:
                st.caption(f"⚠ Không tìm thấy: {fname}")

        st.markdown("**📊 Dữ liệu hồi quy đa biến**")
        try:
            with open("/mnt/user-data/uploads/Time_Series_Regression_Dataset.xlsx", "rb") as f:
                st.download_button(
                    label="📈 Time Series Regression Dataset (nhiều biến X)",
                    data=f.read(),
                    file_name="Time_Series_Regression_Dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_regression",
                )
        except Exception:
            st.caption("⚠ Không tìm thấy file hồi quy đa biến.")

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
    👋 Hãy tải lên một file dữ liệu chuỗi thời gian (ví dụ: doanh số theo tháng, chỉ số FTSE, sản lượng khai thác...).<br><br>
    App hoạt động với file có ít nhất <b>một cột ngày/tháng</b> và <b>một cột giá trị số</b>.
    Các cột ngày kiểu <code>2000 JAN</code>, <code>2000-01</code>, hoặc dạng ngày chuẩn đều được hỗ trợ.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = load_data(uploaded_file)
if df is None or df.empty:
    st.error("Không đọc được dữ liệu từ file.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="section-hdr">② CHỌN CỘT</div>', unsafe_allow_html=True)
    auto_date_col, auto_value_cols = auto_detect_columns(df)
    all_cols = df.columns.tolist()
    date_col = st.selectbox("Cột thời gian (ngày/tháng)", all_cols,
                             index=all_cols.index(auto_date_col) if auto_date_col in all_cols else 0)
    numeric_candidates = [c for c in all_cols if c != date_col]
    default_value = auto_value_cols[0] if auto_value_cols else numeric_candidates[0]
    value_col = st.selectbox("Cột cần dự báo (Y)", numeric_candidates,
                              index=numeric_candidates.index(default_value) if default_value in numeric_candidates else 0)

series_raw = build_series(df, date_col, value_col)
if len(series_raw) < 24:
    st.markdown('<div class="warn-box">⚠️ Chuỗi dữ liệu quá ngắn (cần ít nhất 24 quan sát, ~2 chu kỳ) để phân tích mùa vụ một cách tin cậy.</div>', unsafe_allow_html=True)
    st.stop()

auto_period, auto_freq = infer_period_and_freq(series_raw.index)

with st.sidebar:
    st.markdown('<div class="section-hdr">③ THIẾT LẬP DỰ BÁO</div>', unsafe_allow_html=True)
    period = st.number_input("Chu kỳ mùa vụ (s)", min_value=1, max_value=365, value=int(auto_period),
                              help="Số kỳ trong một chu kỳ mùa vụ — 12 cho dữ liệu tháng, 4 cho dữ liệu quý.")
    horizon = st.number_input("Số kỳ dự báo (horizon)", min_value=1, max_value=60, value=int(period))
    test_size = st.number_input("Số kỳ giữ lại để Backtest (Test set)", min_value=2, max_value=60, value=int(period))
    freq = st.selectbox("Tần suất (freq)", ['MS', 'M', 'QS', 'Q', 'W', 'D', 'AS'],
                         index=['MS', 'M', 'QS', 'Q', 'W', 'D', 'AS'].index(auto_freq) if auto_freq in ['MS', 'M', 'QS', 'Q', 'W', 'D', 'AS'] else 0,
                         help="Tần suất của chuỗi thời gian — ảnh hưởng đến cách tạo trục thời gian trong dự báo.")
    st.markdown("""
    <div style="background:#111827;border:1px solid #30363d;border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:#8b949e;margin-top:0.3rem;">
    <b style="color:#58a6ff;">Giải thích viết tắt Freq:</b><br>
    <b style="color:#e6edf3;">MS</b> — Month Start: đầu mỗi tháng (phổ biến nhất)<br>
    <b style="color:#e6edf3;">M</b> &nbsp;— Month End: cuối mỗi tháng<br>
    <b style="color:#e6edf3;">QS</b> — Quarter Start: đầu mỗi quý (Q1=T1, Q2=T4...)<br>
    <b style="color:#e6edf3;">Q</b> &nbsp;— Quarter End: cuối mỗi quý<br>
    <b style="color:#e6edf3;">W</b> &nbsp;— Weekly: mỗi tuần (chủ nhật)<br>
    <b style="color:#e6edf3;">D</b> &nbsp;— Daily: mỗi ngày<br>
    <b style="color:#e6edf3;">AS</b> — Annual Start: đầu mỗi năm
    </div>
    """, unsafe_allow_html=True)

# try to attach a regular frequency to the index (helps statsmodels avoid warnings)
series = series_raw.copy()
try:
    series = series.asfreq(freq)
    if series.isna().any():
        series = series.interpolate()
except Exception:
    pass

PLABEL = PERIOD_LABELS.get(period, 'Kỳ')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">① TỔNG QUAN DỮ LIỆU</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số quan sát", len(series))
c2.metric("Từ", series.index.min().strftime('%b %Y'))
c3.metric("Đến", series.index.max().strftime('%b %Y'))
c4.metric("Chu kỳ mùa vụ", f"{period} ({PLABEL})")

fig, ax = new_fig(figsize=(11, 3.6))
ax.plot(series.index, series.values, color=BLUE, linewidth=1.4)
style_ax(ax, f'Time plot — {value_col}', 'Thời gian', value_col)
show_fig(fig)

first_q = series.iloc[:max(period, 1)].mean()
last_q = series.iloc[-max(period, 1):].mean()
trend_pct = (last_q - first_q) / abs(first_q) * 100 if first_q != 0 else np.nan
trend_dir = "tăng" if trend_pct > 1 else ("giảm" if trend_pct < -1 else "ổn định")
st.markdown(f"""
<div class="interpret-box">
📊 <b>Nhận xét nhanh:</b> So sánh trung bình {period} kỳ đầu và {period} kỳ cuối, chuỗi <b>{trend_dir}</b>
khoảng <b>{trend_pct:+.1f}%</b>. Giá trị trung bình toàn chuỗi ≈ <b>{series.mean():,.2f}</b>,
độ lệch chuẩn ≈ <b>{series.std():,.2f}</b>.
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 Xem dữ liệu thô"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">② PHÂN RÃ CHUỖI THỜI GIAN (DECOMPOSITION)</div>', unsafe_allow_html=True)

decomp = run_decompositions(series, period)
valid_modes = [m for m in ['additive', 'multiplicative'] if decomp[m]['status'] == 'ok']

if not valid_modes:
    st.markdown('<div class="err-box">Không thể phân rã chuỗi này (chuỗi quá ngắn hoặc lỗi dữ liệu).</div>', unsafe_allow_html=True)
else:
    cols = st.columns(len(valid_modes))
    for col, mode in zip(cols, valid_modes):
        d = decomp[mode]
        with col:
            fig, axes = new_fig(3, 1, figsize=(6, 6), sharex=True)
            axes[0].plot(d['trend'].index, d['trend'].values, color=BLUE)
            style_ax(axes[0], f"{'Cộng (Additive)' if mode=='additive' else 'Nhân (Multiplicative)'} — Trend", '', '')
            axes[1].plot(d['seasonal'].index, d['seasonal'].values, color=GREEN)
            style_ax(axes[1], 'Seasonal', '', '')
            axes[2].plot(d['resid'].index, d['resid'].values, color=RED, linewidth=0.8)
            style_ax(axes[2], 'Residual', 'Thời gian', '')
            fig.tight_layout()
            show_fig(fig)
            st.caption(f"Phần dư ≈ **{d['resid_pct']:.2f}%** so với mức trung bình của chuỗi (Residual MSE thô = {d['resid_mse']:,.4g}, không so sánh trực tiếp giữa 2 mô hình)")
    if decomp['multiplicative']['status'] != 'ok':
        st.markdown(f'<div class="warn-box">ℹ️ Mô hình nhân không khả dụng: {decomp["multiplicative"]["reason"]}</div>', unsafe_allow_html=True)

    if len(valid_modes) == 2:
        rec = 'multiplicative' if decomp['multiplicative']['resid_pct'] < decomp['additive']['resid_pct'] else 'additive'
        rec_vn = 'nhân (multiplicative)' if rec == 'multiplicative' else 'cộng (additive)'
        st.markdown(f"""
        <div class="interpret-box">
        🧩 <b>Phân tích:</b> Phần dư mô hình cộng ≈ <b>{decomp['additive']['resid_pct']:.2f}%</b> so với mức trung bình chuỗi,
        mô hình nhân ≈ <b>{decomp['multiplicative']['resid_pct']:.2f}%</b>.
        (Lưu ý: Residual MSE thô của 2 mô hình nằm trên thang đo khác nhau — phần dư cộng là giá trị tuyệt đối quanh 0,
        còn phần dư nhân là tỷ số quanh 1 — nên không thể so sánh trực tiếp Residual MSE thô; ở đây đã chuẩn hoá về
        cùng đơn vị % để so sánh công bằng.)
        Mô hình <b>{rec_vn}</b> cho phần dư (đã chuẩn hoá) nhỏ hơn → đây là dạng phân rã <b>phù hợp hơn</b> với chuỗi {value_col}.
        Nếu biên độ dao động mùa vụ <b>tăng theo mức độ của xu hướng</b>, mô hình nhân thường phù hợp hơn;

        nếu biên độ mùa vụ ổn định, mô hình cộng phù hợp hơn — điều này sẽ được dùng để gợi ý lựa chọn
        mô hình San bằng số mũ ở phần tiếp theo.
        </div>
        """, unsafe_allow_html=True)
    else:
        rec = 'additive'

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SEASONAL PLOT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">③ BIỂU ĐỒ MÙA VỤ THEO NĂM</div>', unsafe_allow_html=True)

pivot, xlabel = seasonal_pivot(series, period)
fig, ax = new_fig(figsize=(11, 4))
palette = [BLUE, GREEN, YELLOW, PURPLE, PINK, RED, GRAY, '#79c0ff', '#56d364', '#ffab70']
for i, year in enumerate(pivot.columns):
    ax.plot(pivot.index, pivot[year].values, label=str(year), color=palette[i % len(palette)], linewidth=1.3)
style_ax(ax, f'Seasonality of {value_col}', xlabel, value_col)
ax.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID, ncol=min(len(pivot.columns), 8))
show_fig(fig)

# find peak / trough period
period_means = pivot.mean(axis=1)
peak_p = int(period_means.idxmax())
trough_p = int(period_means.idxmin())
st.markdown(f"""
<div class="interpret-box">
📅 <b>Phân tích mùa vụ:</b> Trung bình theo {xlabel.lower()}, giá trị <b>cao nhất</b> thường rơi vào
{xlabel.lower()} <b>{peak_p}</b> (≈ {period_means.max():,.2f}), và <b>thấp nhất</b> vào {xlabel.lower()}
<b>{trough_p}</b> (≈ {period_means.min():,.2f}). Nếu hình dạng các đường qua các năm tương đối giống nhau,
mùa vụ là <b>ổn định</b> — phù hợp để mô hình hoá bằng Holt-Winters.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPONENTIAL SMOOTHING (4 HOLT-WINTERS MODELS)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">④ SAN BẰNG SỐ MŨ — HOLT-WINTERS (4 MÔ HÌNH)</div>', unsafe_allow_html=True)

with st.spinner("Đang chạy 4 mô hình Holt-Winters..."):
    hw_results = run_hw_models(series, period)

hw_table_rows = []
for r in hw_results:
    if r['status'] == 'ok':
        hw_table_rows.append({
            'Model': r['name'], 'Loại': label_combo(r['trend'], r['seasonal']),
            'α (level)': round(r['alpha'], 4), 'β (trend)': round(r['beta'], 4) if not np.isnan(r['beta']) else '—',
            'γ (seasonal)': round(r['gamma'], 4) if not np.isnan(r['gamma']) else '—',
            'MSE': round(r['mse'], 4),
        })
    else:
        hw_table_rows.append({'Model': r['name'], 'Loại': label_combo(r['trend'], r['seasonal']),
                               'α (level)': '—', 'β (trend)': '—', 'γ (seasonal)': '—',
                               'MSE': f"⚠ {r.get('reason','lỗi')}"})

hw_df = pd.DataFrame(hw_table_rows)
ok_results = [r for r in hw_results if r['status'] == 'ok']

if not ok_results:
    st.markdown('<div class="err-box">Không có mô hình Holt-Winters nào chạy thành công.</div>', unsafe_allow_html=True)
    st.stop()

best_hw = min(ok_results, key=lambda r: r['mse'])
best_idx = [r['name'] for r in hw_results].index(best_hw['name'])


def highlight_best(row):
    return ['background-color:#0a2a14;color:#3fb950;font-weight:700' if row['Model'] == best_hw['name'] else '' for _ in row]


st.dataframe(hw_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

st.markdown(f"""
<div class="ok-box">
🏆 <b>Mô hình tốt nhất:</b> {best_hw['name']} — {label_combo(best_hw['trend'], best_hw['seasonal'])},
MSE = <b>{best_hw['mse']:,.4f}</b> (thấp nhất trong 4 mô hình).
</div>
""", unsafe_allow_html=True)

# Forecast plot
hw_forecast_vals = best_hw['fit'].forecast(int(horizon))
fig, ax = new_fig(figsize=(11, 4))
ax.plot(series.index, series.values, color=WHITE, linewidth=1.2, label='Dữ liệu thực tế')
ax.plot(series.index, best_hw['fit'].fittedvalues, color=BLUE, linewidth=1, alpha=0.85, label='Giá trị khớp (fitted)')
ax.plot(hw_forecast_vals.index, hw_forecast_vals.values, color=RED, linewidth=1.6, linestyle='--', label=f'Dự báo {horizon} kỳ')
style_ax(ax, f'Dự báo {value_col} — {best_hw["name"]} ({label_combo(best_hw["trend"], best_hw["seasonal"])})', 'Thời gian', value_col)
ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID)
show_fig(fig)

with st.expander(f"📋 Bảng giá trị dự báo ({horizon} kỳ tới) — Holt-Winters"):
    hw_fc_table = pd.DataFrame({'Date': hw_forecast_vals.index.strftime('%b %Y'), 'Forecast': hw_forecast_vals.values.round(2)})
    st.dataframe(hw_fc_table, use_container_width=True, hide_index=True)

trend_word = {'add': 'cộng', 'mul': 'nhân'}[best_hw['trend']]
seas_word = {'add': 'cộng', 'mul': 'nhân'}[best_hw['seasonal']]
st.markdown(f"""
<div class="interpret-box">
💬 <b>Diễn giải:</b> Mô hình {best_hw['name']} (xu hướng {trend_word}, mùa vụ {seas_word}) được chọn nhờ MSE thấp nhất
(<b>{best_hw['mse']:,.4f}</b>). Hệ số α = {best_hw['alpha']:.3f} cho biết mức độ chuỗi
{'phản ứng nhanh' if best_hw['alpha']>0.5 else 'phản ứng chậm và mượt'} với thay đổi gần đây.
{f"Hệ số β = {best_hw['beta']:.3f} kiểm soát tốc độ cập nhật xu hướng." if not np.isnan(best_hw['beta']) else ''}
{f"Hệ số γ = {best_hw['gamma']:.3f} kiểm soát tốc độ cập nhật mùa vụ." if not np.isnan(best_hw['gamma']) else ''}
Đường nét đứt đỏ là dự báo {horizon} kỳ tiếp theo, được ngoại suy theo xu hướng và mùa vụ đã học từ dữ liệu lịch sử.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SARIMA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">⑤ ARIMA / SARIMA</div>', unsafe_allow_html=True)

with st.spinner("Đang kiểm định tính dừng (ADF test)..."):
    d, D, adf_info = determine_d_D(series, period)

adf_rows = []
if 'original' in adf_info:
    adf_rows.append({'Chuỗi': 'Dữ liệu gốc', 'ADF p-value': round(adf_info['original'], 4),
                      'Dừng?': '✅ Có' if adf_info['original'] <= 0.05 else '❌ Không'})
if 'seasonal_diff' in adf_info:
    adf_rows.append({'Chuỗi': f'Sai phân mùa vụ (lag {period})', 'ADF p-value': round(adf_info['seasonal_diff'], 4),
                      'Dừng?': '✅ Có' if adf_info['seasonal_diff'] <= 0.05 else '❌ Không'})
if 'seasonal_first_diff' in adf_info:
    adf_rows.append({'Chuỗi': 'Sai phân mùa vụ + bậc 1', 'ADF p-value': round(adf_info['seasonal_first_diff'], 4),
                      'Dừng?': '✅ Có' if adf_info['seasonal_first_diff'] <= 0.05 else '❌ Không'})

st.dataframe(pd.DataFrame(adf_rows), use_container_width=True, hide_index=True)
st.markdown(f"""
<div class="info-box">
🔬 Dựa trên kiểm định ADF (Augmented Dickey-Fuller), bậc sai phân được chọn: <b>d = {d}</b>, <b>D = {D}</b>
(chu kỳ mùa vụ s = {period}). Các mô hình SARIMA({{p}},{d},{{q}})({{P}},{D},{{Q}})<sub>{period}</sub> dưới đây
được thử với p, q ∈ {{0,1,2}} (và P, Q được đặt cùng giá trị với p, q theo quy ước phổ biến).
</div>
""", unsafe_allow_html=True)

# differencing plot
fig, axes = new_fig(1, 2 if D > 0 else 1, figsize=(11, 3))
if D > 0:
    seas_diff = series.diff(period).dropna()
    axes[0].plot(seas_diff.index, seas_diff.values, color=GREEN, linewidth=1)
    style_ax(axes[0], f'Sai phân mùa vụ (lag {period})', '', '')
    if d > 0:
        both = seas_diff.diff(1).dropna()
        axes[1].plot(both.index, both.values, color=PURPLE, linewidth=1)
        style_ax(axes[1], 'Sai phân mùa vụ + bậc 1', '', '')
    else:
        axes[1].axis('off')
else:
    axes.plot(series.index, series.values, color=BLUE, linewidth=1)
    style_ax(axes, 'Dữ liệu gốc (đã dừng)', '', '')
fig.tight_layout()
show_fig(fig)

# ── ADF Test explanation box ───────────────────────────────────────────────
st.markdown(f"""
<div class="interpret-box">
🔬 <b>Kiểm định ADF (Augmented Dickey-Fuller) là gì?</b><br><br>
ADF kiểm tra xem chuỗi thời gian có <b>tính dừng</b> (stationary) hay không.
Chuỗi dừng là chuỗi có <b>trung bình, phương sai và tự tương quan không thay đổi theo thời gian</b> —
đây là điều kiện bắt buộc để ARIMA/SARIMA hoạt động đúng.<br><br>
<b>Đọc kết quả:</b><br>
• <b>p-value ≤ 0.05</b> → ✅ <b>Chuỗi Dừng</b> — không cần sai phân thêm<br>
• <b>p-value &gt; 0.05</b> → ❌ <b>Chuỗi Không Dừng</b> — cần lấy sai phân (differencing)<br><br>
<b>Sai phân là gì?</b><br>
• <b>Sai phân mùa vụ (lag {period})</b>: lấy <code>y(t) − y(t−{period})</code> để loại bỏ mùa vụ lặp lại
  → tham số <b>D=1</b> trong SARIMA<br>
• <b>Sai phân bậc 1</b>: lấy <code>y(t) − y(t−1)</code> để loại bỏ xu hướng
  → tham số <b>d=1</b> trong SARIMA<br><br>
<b>Kết quả hiện tại:</b> d = <b>{d}</b>, D = <b>{D}</b>
{"— Dữ liệu gốc đã dừng, không cần sai phân." if d == 0 and D == 0 else
 f"— Sau khi sai phân mùa vụ (lag {period}), chuỗi đã dừng." if d == 0 and D == 1 else
 "— Cần cả sai phân mùa vụ lẫn sai phân bậc 1 để đạt tính dừng."}
</div>
""", unsafe_allow_html=True)

# ── ACF & PACF plots ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr" style="margin-top:1rem;">ACF & PACF — TỰ TƯƠNG QUAN</div>', unsafe_allow_html=True)

# Choose the differenced series for ACF/PACF (same as used for SARIMA)
if D > 0:
    _acf_series = series.diff(period).dropna()
    if d > 0:
        _acf_series = _acf_series.diff(1).dropna()
    _acf_label = f"Chuỗi sau sai phân (d={d}, D={D})"
else:
    _acf_series = series.copy()
    _acf_label = "Dữ liệu gốc"

_nlags = min(40, len(_acf_series) // 2 - 1)

try:
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig_acf, axes_acf = new_fig(1, 2, figsize=(11, 3.5))

    # ACF
    plot_acf(_acf_series, lags=_nlags, ax=axes_acf[0], color=BLUE, alpha=0.05)
    axes_acf[0].lines[0].set_color(BLUE)
    for line in axes_acf[0].lines[1:]:
        line.set_color(BLUE)
    axes_acf[0].collections[0].set_facecolor(BLUE)
    axes_acf[0].collections[0].set_alpha(0.2)
    style_ax(axes_acf[0], f'ACF — {_acf_label}', 'Lag', 'Tương quan')

    # PACF
    plot_pacf(_acf_series, lags=_nlags, ax=axes_acf[1], color=PURPLE, alpha=0.05, method='ywm')
    axes_acf[1].lines[0].set_color(PURPLE)
    for line in axes_acf[1].lines[1:]:
        line.set_color(PURPLE)
    axes_acf[1].collections[0].set_facecolor(PURPLE)
    axes_acf[1].collections[0].set_alpha(0.2)
    style_ax(axes_acf[1], f'PACF — {_acf_label}', 'Lag', 'Tương quan riêng phần')

    fig_acf.tight_layout()
    show_fig(fig_acf)

    # Auto-read insights from ACF/PACF
    acf_vals = acf(_acf_series, nlags=_nlags, fft=True)
    pacf_vals = pacf(_acf_series, nlags=_nlags, method='ywm')
    ci_bound = 1.96 / np.sqrt(len(_acf_series))

    sig_acf_lags = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci_bound]
    sig_pacf_lags = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci_bound]

    # Detect q hint (ACF cut-off) and p hint (PACF cut-off)
    acf_cutoff = next((i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) <= ci_bound), None)
    pacf_cutoff = next((i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) <= ci_bound), None)

    seasonal_sig = [l for l in sig_acf_lags if l % period == 0 and l > 0][:3]

    q_hint = min(acf_cutoff - 1, 3) if acf_cutoff and acf_cutoff > 1 else 1
    p_hint = min(pacf_cutoff - 1, 3) if pacf_cutoff and pacf_cutoff > 1 else 1

    st.markdown(f"""
    <div class="interpret-box">
    📊 <b>ACF và PACF là gì?</b><br><br>
    <b>ACF (Autocorrelation Function)</b> — Tự tương quan: đo xem giá trị tại thời điểm <i>t</i>
    tương quan bao nhiêu với giá trị tại thời điểm <i>t−k</i> (k = lag). Nếu ACF giảm chậm → chuỗi
    có xu hướng hoặc chưa dừng. Nếu có spike tại các bội số của {period} → mùa vụ rõ rệt.<br><br>
    <b>PACF (Partial Autocorrelation Function)</b> — Tự tương quan riêng phần: đo tương quan giữa
    <i>t</i> và <i>t−k</i> sau khi đã loại bỏ ảnh hưởng của các lag trung gian. PACF giúp xác định
    bậc AR (<b>p</b>) — số lần "nhìn về quá khứ" của mô hình.<br><br>
    <b>Vùng tin cậy 95%</b> (đường kẻ ngang): spike vượt ra ngoài vùng này mới có ý nghĩa thống kê.<br><br>
    <b>🔍 Insights từ dữ liệu hiện tại ({_acf_label}):</b><br>
    {"• ACF có " + str(len(sig_acf_lags)) + f" lag vượt ngưỡng (±{ci_bound:.2f}): lag " + str(sig_acf_lags[:8])[1:-1] + ("..." if len(sig_acf_lags) > 8 else "") + "<br>" if sig_acf_lags else "• ACF không có lag nào vượt ngưỡng ý nghĩa → chuỗi gần như nhiễu trắng (white noise).<br>"}
    {"• PACF có " + str(len(sig_pacf_lags)) + f" lag vượt ngưỡng: lag " + str(sig_pacf_lags[:8])[1:-1] + ("..." if len(sig_pacf_lags) > 8 else "") + "<br>" if sig_pacf_lags else "• PACF không có lag nào vượt ngưỡng.<br>"}
    {"• <b>Mùa vụ rõ qua ACF:</b> spike tại các bội số của " + str(period) + " (lag " + ", ".join(str(l) for l in seasonal_sig) + ") → xác nhận cần thành phần seasonal trong mô hình.<br>" if seasonal_sig else ""}
    • <b>Gợi ý tham số:</b> PACF cắt tại lag {pacf_cutoff or "?"} → <b>p ≈ {p_hint}</b>;
      ACF cắt tại lag {acf_cutoff or "?"} → <b>q ≈ {q_hint}</b>.
      (App đã tự động thử nhiều tổ hợp p, q ở phần SARIMA bên dưới.)
    </div>
    """, unsafe_allow_html=True)

except Exception as _acf_err:
    st.markdown(f'<div class="warn-box">⚠ Không thể vẽ ACF/PACF: {_acf_err}</div>', unsafe_allow_html=True)

with st.spinner("Đang tìm mô hình SARIMA tốt nhất (có thể mất khoảng 30-60 giây)..."):
    sarima_results = sarima_grid_search(series, period, d, D, DEFAULT_SARIMA_CANDIDATES)

sarima_ok = [r for r in sarima_results if r['status'] == 'ok' and not np.isnan(r['aic'])]

if not sarima_ok:
    st.markdown('<div class="err-box">Không có mô hình SARIMA nào hội tụ thành công với chuỗi này.</div>', unsafe_allow_html=True)
    best_sarima_res = None
    sarima_forecast_vals = None
else:
    sarima_table = pd.DataFrame([{
        'order (p,d,q)': str(r['order']), 'seasonal_order (P,D,Q,s)': str(r['seasonal_order']),
        'AIC': round(r['aic'], 2), 'MSE': round(r['mse'], 4) if not np.isnan(r['mse']) else None,
    } for r in sarima_results if r['status'] == 'ok'])

    best_sarima = min(sarima_ok, key=lambda r: r['aic'])

    def highlight_best_sarima(row):
        is_best = row['order (p,d,q)'] == str(best_sarima['order']) and row['seasonal_order (P,D,Q,s)'] == str(best_sarima['seasonal_order'])
        return ['background-color:#0a2a14;color:#3fb950;font-weight:700' if is_best else '' for _ in row]

    st.dataframe(sarima_table.style.apply(highlight_best_sarima, axis=1), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="ok-box">
    🏆 <b>Mô hình tốt nhất:</b> SARIMA{best_sarima['order']}{best_sarima['seasonal_order']} —
    AIC = <b>{best_sarima['aic']:,.2f}</b>, MSE (fitted) = <b>{best_sarima['mse']:,.4f}</b>.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Đang fit mô hình SARIMA tốt nhất và dự báo..."):
        best_sarima_res = fit_sarima(series, best_sarima['order'], best_sarima['seasonal_order'])
        sarima_fc_obj = best_sarima_res.get_forecast(steps=int(horizon))
        sarima_forecast_vals = sarima_fc_obj.predicted_mean
        sarima_ci = sarima_fc_obj.conf_int()

    fig, ax = new_fig(figsize=(11, 4))
    ax.plot(series.index, series.values, color=WHITE, linewidth=1.2, label='Dữ liệu thực tế')
    ax.plot(sarima_forecast_vals.index, sarima_forecast_vals.values, color=RED, linewidth=1.6, linestyle='--', label=f'Dự báo {horizon} kỳ')
    ax.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color=BLUE, alpha=.18, label='Khoảng tin cậy 95%')
    style_ax(ax, f'Dự báo {value_col} — SARIMA{best_sarima["order"]}{best_sarima["seasonal_order"]}', 'Thời gian', value_col)
    ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID)
    show_fig(fig)

    with st.expander(f"📋 Bảng giá trị dự báo ({horizon} kỳ tới) — SARIMA"):
        sarima_fc_table = pd.DataFrame({'Date': sarima_forecast_vals.index.strftime('%b %Y'),
                                          'Forecast': sarima_forecast_vals.values.round(2),
                                          'Lower CI': sarima_ci.iloc[:, 0].values.round(2),
                                          'Upper CI': sarima_ci.iloc[:, 1].values.round(2)})
        st.dataframe(sarima_fc_table, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="interpret-box">
    💬 <b>Diễn giải:</b> SARIMA{best_sarima['order']}{best_sarima['seasonal_order']} có AIC thấp nhất trong các mô hình thử
    nghiệm, cho thấy đây là sự đánh đổi tốt nhất giữa độ khớp và độ phức tạp (số tham số). Dải tô màu xanh là
    khoảng tin cậy 95% — khoảng này thường <b>mở rộng dần</b> theo thời gian dự báo vì độ bất định tăng lên
    khi dự báo xa hơn vào tương lai.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BACKTEST: HOLT-WINTERS vs SARIMA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">⑥ BACKTEST: SO SÁNH HOLT-WINTERS vs SARIMA</div>', unsafe_allow_html=True)

if len(series) <= test_size + period:
    st.markdown('<div class="warn-box">⚠️ Chuỗi quá ngắn so với kích thước Test set đã chọn — bỏ qua backtest.</div>', unsafe_allow_html=True)
    backtest_table = None
    hw_bt_ok = False
    sarima_bt_ok = False
    hw_test_mse = np.nan
    sarima_test_mse = np.nan
else:
    train = series.iloc[:-int(test_size)]
    test = series.iloc[-int(test_size):]

    with st.spinner("Đang chạy backtest..."):
        # HW on train
        try:
            hw_train_fit = ExponentialSmoothing(train, seasonal_periods=period, trend=best_hw['trend'], seasonal=best_hw['seasonal']).fit()
            hw_test_fc = hw_train_fit.forecast(int(test_size))
            hw_test_mse = float(mean_squared_error(test, hw_test_fc))
            hw_bt_ok = True
        except Exception as e:
            hw_bt_ok = False
            hw_test_fc = None
            hw_test_mse = np.nan

        # SARIMA on train
        sarima_bt_ok = False
        if best_sarima_res is not None:
            try:
                sarima_train_res = fit_sarima(train, best_sarima['order'], best_sarima['seasonal_order'])
                sarima_test_fc = sarima_train_res.get_forecast(steps=int(test_size)).predicted_mean
                sarima_test_mse = float(mean_squared_error(test, sarima_test_fc))
                sarima_bt_ok = True
            except Exception:
                sarima_test_fc = None
                sarima_test_mse = np.nan

    fig, ax = new_fig(figsize=(11, 4))
    ax.plot(train.index, train.values, color=GRAY, linewidth=1, label='Train')
    ax.plot(test.index, test.values, color=WHITE, linewidth=1.6, label='Test (thực tế)')
    if hw_bt_ok:
        ax.plot(hw_test_fc.index, hw_test_fc.values, color=BLUE, linewidth=1.6, linestyle='--', label='Holt-Winters dự báo')
    if sarima_bt_ok:
        ax.plot(sarima_test_fc.index, sarima_test_fc.values, color=RED, linewidth=1.6, linestyle='--', label='SARIMA dự báo')
    style_ax(ax, f'Backtest {int(test_size)} kỳ cuối', 'Thời gian', value_col)
    ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID)
    show_fig(fig)

    bt_rows = []
    if hw_bt_ok:
        bt_rows.append({'Phương pháp': f"Holt-Winters ({best_hw['name']})", 'MSE trên Test set': round(hw_test_mse, 4)})
    if sarima_bt_ok:
        bt_rows.append({'Phương pháp': f"SARIMA{best_sarima['order']}{best_sarima['seasonal_order']}", 'MSE trên Test set': round(sarima_test_mse, 4)})

    backtest_table = pd.DataFrame(bt_rows)
    if len(bt_rows) > 0:
        st.dataframe(backtest_table, use_container_width=True, hide_index=True)

    if hw_bt_ok and sarima_bt_ok:
        if sarima_test_mse < hw_test_mse:
            winner, w_mse, loser, l_mse = 'SARIMA', sarima_test_mse, 'Holt-Winters', hw_test_mse
        else:
            winner, w_mse, loser, l_mse = 'Holt-Winters', hw_test_mse, 'SARIMA', sarima_test_mse
        st.markdown(f"""
        <div class="interpret-box">
        🥇 <b>Kết luận backtest:</b> Trên {int(test_size)} kỳ gần nhất (giữ lại làm test set), phương pháp
        <b>{winner}</b> cho MSE thấp hơn (<b>{w_mse:,.4f}</b> so với <b>{l_mse:,.4f}</b> của {loser}).
        {'ARIMA/SARIMA thường vượt trội hơn cho dự báo ngắn hạn vì tập trung vào diễn biến gần đây, trong khi Holt-Winters thích hợp hơn khi cần nắm bắt xu hướng và mùa vụ dài hạn ổn định.' if winner=='SARIMA' else 'Holt-Winters thường vượt trội khi mùa vụ và xu hướng dài hạn ổn định, ít bị nhiễu bởi biến động ngắn hạn gần đây.'}
        ⇒ Khuyến nghị sử dụng <b>{winner}</b> để dự báo {horizon} kỳ tiếp theo cho chuỗi {value_col}.
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OPTIONAL MULTIVARIABLE REGRESSION FORECAST
# ══════════════════════════════════════════════════════════════════════════════
other_numeric = [c for c in df.columns if c not in (date_col, value_col) and pd.to_numeric(df[c], errors='coerce').notna().sum() / max(len(df), 1) > 0.7]

reg_forecast = None
reg_mse = None
if other_numeric:
    st.markdown('<div class="section-hdr">⑦ DỰ BÁO HỒI QUY ĐA BIẾN (TÙY CHỌN)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    File này có các cột số khác: {', '.join(f'<code>{c}</code>' for c in other_numeric)}.
    Bạn có thể dùng chúng làm biến giải thích để dự báo <b>{value_col}</b> bằng hồi quy OLS
    (kèm biến giả theo {PLABEL.lower()} và biến thời gian), tương tự cách dự báo FTSE từ K54D, EAFV, K226, JQ2J
    trong báo cáo mẫu.
    </div>
    """, unsafe_allow_html=True)
    enable_reg = st.checkbox("Bật dự báo hồi quy đa biến", value=False)
    if enable_reg:
        x_cols = st.multiselect("Chọn biến giải thích (X)", other_numeric, default=other_numeric[:min(3, len(other_numeric))])
        use_dummies = st.checkbox(f"Thêm biến giả theo {PLABEL.lower()} (D1..D{period-1})", value=(period > 1))
        use_time = st.checkbox("Thêm biến xu hướng thời gian (t)", value=True)

        if x_cols:
            with st.spinner("Đang chạy hồi quy và dự báo các biến giải thích..."):
                aligned = build_series(df, date_col, value_col).to_frame()
                for xc in x_cols:
                    xs = build_series(df, date_col, xc)
                    aligned = aligned.join(xs.to_frame(), how='inner')
                aligned = aligned.dropna()
                try:
                    aligned = aligned.asfreq(freq)
                    aligned = aligned.interpolate()
                except Exception:
                    pass

                n = len(aligned)
                X = aligned[x_cols].copy()
                if use_dummies and period > 1:
                    X = pd.concat([X, build_dummies(aligned.index, period)], axis=1)
                if use_time:
                    X['time'] = np.arange(1, n + 1)
                X_const = sm.add_constant(X)
                y = aligned[value_col]

                model = sm.OLS(y, X_const).fit()
                fitted = model.predict(X_const)
                reg_mse = float(mean_squared_error(y, fitted))

                # forecast each X with its best HW model, then plug into regression
                x_forecasts = {}
                for xc in x_cols:
                    xs = aligned[xc]
                    xpositive = (xs > 0).all()
                    try:
                        xfit = ExponentialSmoothing(xs, seasonal_periods=period,
                                                       trend='mul' if xpositive else 'add',
                                                       seasonal='mul' if xpositive else 'add').fit()
                    except Exception:
                        xfit = ExponentialSmoothing(xs, seasonal_periods=period, trend='add', seasonal='add').fit()
                    x_forecasts[xc] = xfit.forecast(int(horizon))

                future_index = make_future_index(aligned.index[-1], int(horizon), freq)
                Xf = pd.DataFrame(index=future_index)
                for xc in x_cols:
                    Xf[xc] = x_forecasts[xc].values
                if use_dummies and period > 1:
                    Xf = pd.concat([Xf, build_dummies(future_index, period)], axis=1)
                if use_time:
                    Xf['time'] = np.arange(n + 1, n + 1 + int(horizon))
                Xf_const = sm.add_constant(Xf, has_constant='add')
                Xf_const = Xf_const.reindex(columns=X_const.columns, fill_value=0)

                reg_forecast = model.predict(Xf_const)

            st.markdown(f"**R² = {model.rsquared:.4f} · Adj. R² = {model.rsquared_adj:.4f} · MSE = {reg_mse:,.4f}**")
            with st.expander("📄 Bảng hệ số hồi quy (OLS)"):
                coef_df = pd.DataFrame({'Biến': model.params.index, 'Hệ số': model.params.values.round(4),
                                          'p-value': model.pvalues.values.round(4)})
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

            fig, ax = new_fig(figsize=(11, 4))
            ax.plot(aligned.index, y.values, color=WHITE, linewidth=1.2, label='Dữ liệu thực tế')
            ax.plot(aligned.index, fitted.values, color=BLUE, linewidth=1, alpha=.85, label='Giá trị khớp')
            ax.plot(reg_forecast.index, reg_forecast.values, color=RED, linewidth=1.6, linestyle='--', label=f'Dự báo {horizon} kỳ')
            z = 1.282  # 80% CI
            ax.fill_between(reg_forecast.index, reg_forecast.values - z * np.sqrt(reg_mse),
                             reg_forecast.values + z * np.sqrt(reg_mse), color=BLUE, alpha=.15, label='Khoảng tin cậy 80%')
            style_ax(ax, f'Dự báo {value_col} bằng hồi quy đa biến', 'Thời gian', value_col)
            ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID)
            show_fig(fig)

            sig_vars = [v for v in x_cols if model.pvalues.get(v, 1) < 0.05]
            st.markdown(f"""
            <div class="interpret-box">
            💬 <b>Diễn giải:</b> Mô hình hồi quy giải thích được <b>{model.rsquared*100:.1f}%</b> biến động của
            {value_col} (R² = {model.rsquared:.3f}). {f"Các biến có ý nghĩa thống kê (p &lt; 0.05): {', '.join(sig_vars)}." if sig_vars else "Không có biến giải thích nào có ý nghĩa thống kê ở mức 5% — kết quả dự báo cần được xem xét cẩn trọng."}
            Dự báo {value_col} cho {horizon} kỳ tới được tính bằng cách: (1) dự báo từng biến giải thích bằng
            Holt-Winters, (2) đưa các giá trị dự báo này vào phương trình hồi quy đã ước lượng.
            Khoảng tin cậy 80% (vùng tô màu) dựa trên MSE = {reg_mse:,.4f} của phần khớp trong mẫu.
            </div>
            """, unsafe_allow_html=True)

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FINAL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">⑧ 🎯 KHUYẾN NGHỊ CUỐI CÙNG</div>', unsafe_allow_html=True)

candidates = []
if hw_bt_ok and not np.isnan(hw_test_mse):
    candidates.append({
        'name': f"Holt-Winters ({best_hw['name']} — {label_combo(best_hw['trend'], best_hw['seasonal'])})",
        'key': 'hw', 'mse': hw_test_mse, 'forecast': hw_forecast_vals,
        'reason': 'phù hợp khi xu hướng và mùa vụ ổn định, ít bị xáo trộn bởi biến động ngắn hạn gần đây.',
    })
if sarima_bt_ok and sarima_ok and not np.isnan(sarima_test_mse):
    candidates.append({
        'name': f"SARIMA{best_sarima['order']}{best_sarima['seasonal_order']}",
        'key': 'sarima', 'mse': sarima_test_mse, 'forecast': sarima_forecast_vals,
        'reason': 'phù hợp cho dự báo ngắn hạn vì tập trung khai thác diễn biến và tương quan gần nhất của chuỗi.',
    })
if reg_forecast is not None and reg_mse is not None:
    candidates.append({
        'name': 'Hồi quy đa biến (OLS)',
        'key': 'reg', 'mse': reg_mse, 'forecast': reg_forecast,
        'reason': 'phù hợp khi có các biến giải thích liên quan chặt với biến mục tiêu và quan hệ này ổn định theo thời gian.',
    })

if not candidates:
    st.markdown('<div class="warn-box">⚠️ Không đủ dữ liệu backtest để đưa ra khuyến nghị (chuỗi quá ngắn hoặc các mô hình đều lỗi). '
                'Bạn có thể tham khảo trực tiếp kết quả Holt-Winters hoặc SARIMA ở các phần trên.</div>', unsafe_allow_html=True)
else:
    winner = min(candidates, key=lambda c: c['mse'])
    others = [c for c in candidates if c is not winner]

    cmp_rows = [{'Phương pháp': c['name'], 'MSE (backtest / khớp mẫu)': round(c['mse'], 4),
                 '': '🏆 Tốt nhất' if c is winner else ''} for c in sorted(candidates, key=lambda c: c['mse'])]
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    others_txt = ''
    if others:
        comp_list = ', '.join(f"{c['name']} (MSE={c['mse']:,.4f})" for c in others)
        others_txt = f" So với {comp_list}, đây là lựa chọn có sai số thấp nhất trên dữ liệu kiểm định."

    st.markdown(f"""
    <div class="ok-box" style="font-size:0.95rem;">
    🏆 <b>Khuyến nghị: sử dụng {winner['name']} để dự báo {value_col} cho {horizon} {PLABEL.lower()} tới.</b><br><br>
    MSE trên dữ liệu kiểm định (backtest) = <b>{winner['mse']:,.4f}</b> — thấp nhất trong số các phương pháp đã thử.{others_txt}<br>
    Đây là phương pháp {winner['reason']}
    </div>
    """, unsafe_allow_html=True)

    final_forecast = winner['forecast']
    fig, ax = new_fig(figsize=(11, 4))
    ax.plot(series.index, series.values, color=WHITE, linewidth=1.3, label='Dữ liệu thực tế')
    ax.plot(final_forecast.index, final_forecast.values, color=GREEN, linewidth=2, linestyle='--',
            label=f"Dự báo khuyến nghị — {winner['name']}")
    style_ax(ax, f'Dự báo cuối cùng được khuyến nghị cho {value_col}', 'Thời gian', value_col)
    ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=GRID)
    show_fig(fig)

    final_table = pd.DataFrame({
        'Date': final_forecast.index.strftime('%b %Y'),
        f'Forecast ({winner["name"]})': final_forecast.values.round(2),
    })
    st.dataframe(final_table, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="interpret-box">
    💬 <b>Lưu ý khi sử dụng kết quả:</b> Khuyến nghị này dựa trên việc so sánh sai số dự báo (MSE) trên dữ liệu
    kiểm định gần nhất — đây là cách đánh giá khách quan nhưng vẫn có giới hạn vì chỉ phản ánh hiệu suất trong
    quá khứ gần. Nếu có sự kiện bất thường sắp xảy ra (thay đổi chính sách, cú sốc kinh tế, v.v.), các mô hình
    ngoại suy như trên đều khó nắm bắt — nên kết hợp thêm đánh giá định tính (judgmental forecasting) khi cần.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">⑨ XUẤT KẾT QUẢ EXCEL</div>', unsafe_allow_html=True)

excel_bytes = export_results_excel(
    series, hw_results, best_hw, sarima_results if sarima_ok else [], best_sarima_res,
    best_sarima['order'] if sarima_ok else None, hw_forecast_vals,
    sarima_forecast_vals if sarima_ok else None, int(horizon), freq,
    backtest_table=backtest_table, reg_forecast=reg_forecast,
)

ecol, icol = st.columns([1, 2])
with ecol:
    st.download_button(
        label="⬇️ Tải xuống Excel (kết quả đầy đủ)",
        data=excel_bytes,
        file_name=f"Forecast_{value_col[:20].strip().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with icol:
    st.markdown("""
    <div class="info-box">
    File Excel bao gồm:<br>
    • <b>Sheet Forecast</b>: dự báo Holt-Winters / SARIMA / Hồi quy (nếu có) cho các kỳ tới<br>
    • <b>Sheet HW Model details</b>: bảng so sánh 4 mô hình Holt-Winters<br>
    • <b>Sheet SARIMA model details</b>: bảng so sánh các mô hình SARIMA đã thử<br>
    • <b>Sheet Backtest comparison</b>: MSE trên test set của từng phương pháp<br>
    • <b>Sheet Data</b>: dữ liệu gốc đã dùng để phân tích
    </div>
    """, unsafe_allow_html=True)
