import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap

# =========================
# 0. パス設定（Streamlit Cloud対応）
# =========================
BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "cultivate_dataset_0122.csv")

MODEL_G_PATH = os.path.join(BASE_DIR, "gndvi_rf_model.joblib")
MODEL_D_RAW_PATH = os.path.join(BASE_DIR, "drymatter_rf_model.joblib")
MODEL_D_LOG_PATH = os.path.join(BASE_DIR, "drymatter_log1p_rf_model.joblib")

COLS_PATH = os.path.join(BASE_DIR, "model_feature_columns.txt")
RECO_PATH = os.path.join(BASE_DIR, "recommendation_grid_results.csv")

st.set_page_config(
    page_title="中耕除草効果予測アプリ（ML_AL7）",
    layout="wide"
)

# =========================
# 1. ユーティリティ
# =========================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def load_feature_cols(path: str):
    with open(path, encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols


def make_slider_range(series: pd.Series, fallback_min: float, fallback_max: float, fallback_mid: float):
    if series is None or series.dropna().empty:
        return fallback_min, fallback_max, fallback_mid
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return fallback_min, fallback_max, fallback_mid
    mn, mx = float(s.min()), float(s.max())
    mid = float(s.median())
    if mn == mx:
        mn -= 1e-6
        mx += 1e-6
    return mn, mx, mid


def build_row_dum(
    controls: dict,
    feature_cols_expanded: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
):
    """
    controls（数値+カテゴリ）から1行DataFrameを作り、
    学習時と同じダミー化(drop_first=True)→reindexで列揃えして返す
    """
    row = {}
    for c in numeric_cols:
        row[c] = controls.get(c, np.nan)
    for c in cat_cols:
        row[c] = controls.get(c, "Unknown")

    row_df = pd.DataFrame([row])

    for c in numeric_cols:
        row_df[c] = pd.to_numeric(row_df[c], errors="coerce")

    for c in cat_cols:
        row_df[c] = row_df[c].astype(str)

    row_dum = pd.get_dummies(row_df, columns=cat_cols, drop_first=True)
    row_dum = row_dum.reindex(columns=feature_cols_expanded, fill_value=0)
    return row_dum


def predict_drymatter(model_raw, model_log, row_dum: pd.DataFrame, use_log1p: bool):
    if not use_log1p:
        return float(model_raw.predict(row_dum)[0])
    pred_log = float(model_log.predict(row_dum)[0])
    return float(np.expm1(pred_log))


def format_controls_for_display(controls: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "中耕前雑草率 (%)": controls["weed_before_gndvi"],
        "中耕前雑草乾物重 (g/m²)": controls["weed_before_dm"],
        "ダイズ被覆率 (%)": controls["soy_cover_mean"],
        "ダイズ被覆率SD": controls["soy_cover_sd"],
        "条間 (m)": controls["row_space"],
        "作業速度 (m/s)": controls["work_speed"],
        "作業能率 (h/ha)": controls["work_rate"],
        "緑肥": controls["green_manure"],
        "中耕機種類": controls["culti_type"],
        "培土板": controls["mold_boad"],
        "1日前降雨量 (mm)": controls["rain_1d"],
        "3日前まで降雨量 (mm)": controls["rain_3d"],
        "7日前まで降雨量 (mm)": controls["rain_7d"],
        "1日の日照時間 (h)": controls["sun_time_1d"],
        "3日の日照時間 (h)": controls["sun_time_3d"],
        "7日の日照時間 (h)": controls["sun_time_7d"],
        "1日の積算日射量 (MJ/m²)": controls["sun_energy_1d"],
        "3日の積算日射量 (MJ/m²)": controls["sun_energy_3d"],
        "7日の積算日射量 (MJ/m²)": controls["sun_energy_7d"],
        "平均気温 (℃)": controls["temp_avg"],
        "平均湿度 (%)": controls["hum_avg"],
        "平均風速 (m/s)": controls["air_speed_avg"],
        "中耕までの積算気温 (℃・日)": controls["until_culti_temp"],
        "播種後積算気温 (℃・日)": controls["after_sowing_temp"],
    }])


def encode_dataframe_for_model(
    df_input: pd.DataFrame,
    feature_cols_expanded: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """
    学習データ相当のDataFrameを、推論時と同じダミー化ルールで
    model_feature_columns.txtに合わせて列揃えする
    """
    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=feature_cols_expanded)

    df_tmp = df_input.copy()

    for c in numeric_cols:
        if c not in df_tmp.columns:
            df_tmp[c] = np.nan
        df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce")

    for c in cat_cols:
        if c not in df_tmp.columns:
            df_tmp[c] = "Unknown"
        df_tmp[c] = df_tmp[c].astype(str)

    df_tmp = df_tmp[numeric_cols + cat_cols].copy()
    df_dum = pd.get_dummies(df_tmp, columns=cat_cols, drop_first=True)
    df_dum = df_dum.reindex(columns=feature_cols_expanded, fill_value=0)
    return df_dum


def clean_feature_name(name: str) -> str:
    return (
        name.replace("weed_before_gndvi", "Weed coverage before cultivation")
            .replace("weed_before_dm", "Weed dry weight before cultivation")
            .replace("soy_cover_mean", "Soybean coverage")
            .replace("soy_cover_sd", "Soybean coverage SD")
            .replace("until_culti_temp", "Accumulated temp. until cultivation")
            .replace("after_sowing_temp", "Accumulated temp. after sowing")
            .replace("rain_1d", "Rainfall 1 day")
            .replace("rain_3d", "Rainfall 3 days")
            .replace("rain_7d", "Rainfall 7 days")
            .replace("sun_time_1d", "Sunshine duration 1 day")
            .replace("sun_time_3d", "Sunshine duration 3 days")
            .replace("sun_time_7d", "Sunshine duration 7 days")
            .replace("sun_energy_1d", "Solar radiation 1 day")
            .replace("sun_energy_3d", "Solar radiation 3 days")
            .replace("sun_energy_7d", "Solar radiation 7 days")
            .replace("temp_avg", "Mean temperature")
            .replace("hum_avg", "Mean humidity")
            .replace("air_speed_avg", "Mean wind speed")
            .replace("work_speed", "Work speed")
            .replace("work_rate", "Work rate")
            .replace("row_space", "Row spacing")
            .replace("green_manure_", "Green manure: ")
            .replace("culti_type_", "Cultivator type: ")
            .replace("mold_boad_", "Mold board: ")
    )

@st.cache_resource
def get_tree_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner=False)
def get_background_encoded(
    data_path: str,
    feature_cols_expanded: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
    max_samples: int = 200,
):
    if not os.path.exists(data_path):
        return pd.DataFrame(columns=feature_cols_expanded)

    df_raw_local = pd.read_csv(data_path)
    df_raw_local.columns = [str(c).replace("\u3000", " ").strip() for c in df_raw_local.columns]

    rename_dict_local = {
        "weed_remain_gndvi(%)": "weed_remain_gndvi",
        "weed_after_dm(g/m2)": "weed_after_dm",
        "weed_before_gndvi(%)": "weed_before_gndvi",
        "weed_before_dm(g/m2)": "weed_before_dm",
        "soy_cover_mean(%)": "soy_cover_mean",
        "soy_cover_SD": "soy_cover_sd",
        "Until_culti_temp(℃）": "until_culti_temp",
        "Until_culti_temp(℃)": "until_culti_temp",
        "after_sowing _temp(℃）": "after_sowing_temp",
        "after_sowing_temp": "after_sowing_temp",
        "rain_1day(mm)": "rain_1d",
        "rain_3day(mm)": "rain_3d",
        "rain_7day(mm)": "rain_7d",
        "sun_time_1day(h)": "sun_time_1d",
        "sun_time_3day(h)": "sun_time_3d",
        "sun_time_7day(h)": "sun_time_7d",
        "sun_enagy_1day(MJ/m2)": "sun_energy_1d",
        "sun_enagy_3day(MJ/m2)": "sun_energy_3d",
        "sun_enagy_7day(MJ/m2)": "sun_energy_7d",
        "temp_avg(℃）": "temp_avg",
        "hum_avg（％）": "hum_avg",
        "air_speed_avg(m/s)": "air_speed_avg",
        "work speed(m/s)": "work_speed",
        "work_speed": "work_speed",
        "work rate(h/ha)": "work_rate",
        "work_rate": "work_rate",
        "row_space(m)": "row_space",
        "row_space": "row_space",
    }
    df_local = df_raw_local.rename(columns=rename_dict_local)

    df_enc = encode_dataframe_for_model(
        df_local,
        feature_cols_expanded=feature_cols_expanded,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
    )

    if df_enc.empty:
        return df_enc

    if len(df_enc) > max_samples:
        df_enc = df_enc.sample(n=max_samples, random_state=42)

    return df_enc


def compute_shap_values(explainer, X: pd.DataFrame):
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    return shap_values


def plot_local_shap_bar(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 12,
    title: str = "局所SHAP"
):
    vals = np.array(shap_values_row).flatten()
    if len(vals) == 0:
        st.info("SHAP値を表示できません。")
        return

    abs_order = np.argsort(np.abs(vals))[::-1][:top_n]
    plot_vals = vals[abs_order]
    plot_names = [clean_feature_name(feature_names[i]) for i in abs_order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    y_pos = np.arange(len(plot_names))
    ax.barh(y_pos, plot_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    ax.axvline(0, linewidth=1)

    for i, v in enumerate(plot_vals):
        ax.text(v, i, f" {v:.3f}", va="center")

    plt.tight_layout()
    st.pyplot(fig)


def plot_global_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "全体重要度（平均|SHAP|）"
):
    if shap_values.size == 0:
        st.info("全体SHAP重要度を表示できません。")
        return

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    vals = mean_abs[order]
    names = [clean_feature_name(feature_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)

    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center")

    plt.tight_layout()
    st.pyplot(fig)


# =========================
# 2. モデル読み込み
# =========================
st.sidebar.write("モデル読み込み中...")

if not os.path.exists(COLS_PATH):
    st.error(f"特徴量列ファイルが見つかりません: {COLS_PATH}")
    st.stop()

feature_cols_expanded = load_feature_cols(COLS_PATH)

for p in [MODEL_G_PATH, MODEL_D_RAW_PATH, MODEL_D_LOG_PATH]:
    if not os.path.exists(p):
        st.error(f"モデルファイルが見つかりません: {p}")
        st.stop()

rf_gndvi = joblib.load(MODEL_G_PATH)
rf_dm_raw = joblib.load(MODEL_D_RAW_PATH)
rf_dm_log = joblib.load(MODEL_D_LOG_PATH)

# =========================
# 3. データ読み込み（レンジ＆デフォルト生成用）
# =========================
df = None
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
    df_raw.columns = [str(c).replace("\u3000", " ").strip() for c in df_raw.columns]

    rename_dict = {
        "weed_remain_gndvi(%)": "weed_remain_gndvi",
        "weed_after_dm(g/m2)": "weed_after_dm",
        "weed_before_gndvi(%)": "weed_before_gndvi",
        "weed_before_dm(g/m2)": "weed_before_dm",
        "soy_cover_mean(%)": "soy_cover_mean",
        "soy_cover_SD": "soy_cover_sd",
        "Until_culti_temp(℃）": "until_culti_temp",
        "Until_culti_temp(℃)": "until_culti_temp",
        "after_sowing _temp(℃）": "after_sowing_temp",
        "after_sowing_temp": "after_sowing_temp",
        "rain_1day(mm)": "rain_1d",
        "rain_3day(mm)": "rain_3d",
        "rain_7day(mm)": "rain_7d",
        "sun_time_1day(h)": "sun_time_1d",
        "sun_time_3day(h)": "sun_time_3d",
        "sun_time_7day(h)": "sun_time_7d",
        "sun_enagy_1day(MJ/m2)": "sun_energy_1d",
        "sun_enagy_3day(MJ/m2)": "sun_energy_3d",
        "sun_enagy_7day(MJ/m2)": "sun_energy_7d",
        "temp_avg(℃）": "temp_avg",
        "hum_avg（％）": "hum_avg",
        "air_speed_avg(m/s)": "air_speed_avg",
        "work speed(m/s)": "work_speed",
        "work_speed": "work_speed",
        "work rate(h/ha)": "work_rate",
        "work_rate": "work_rate",
        "row_space(m)": "row_space",
        "row_space": "row_space",
    }
    df = df_raw.rename(columns=rename_dict)

# =========================
# 4. 入力特徴量
# =========================
numeric_cols = [
    "weed_before_gndvi",
    "weed_before_dm",
    "soy_cover_mean",
    "until_culti_temp",
    "after_sowing_temp",
    "rain_1d", "rain_3d", "rain_7d",
    "sun_time_1d", "sun_time_3d", "sun_time_7d",
    "sun_energy_1d", "sun_energy_3d", "sun_energy_7d",
    "temp_avg",
    "hum_avg",
    "air_speed_avg",
    "work_speed",
    "work_rate",
    "row_space",
    "soy_cover_sd",
]
cat_cols = ["green_manure", "culti_type", "mold_boad"]

defaults = {
    "weed_before_gndvi": 20.0,
    "weed_before_dm": 50.0,
    "soy_cover_mean": 50.0,
    "soy_cover_sd": 10.0,
    "until_culti_temp": 300.0,
    "after_sowing_temp": 400.0,
    "rain_1d": 0.0, "rain_3d": 5.0, "rain_7d": 10.0,
    "sun_time_1d": 6.0, "sun_time_3d": 18.0, "sun_time_7d": 42.0,
    "sun_energy_1d": 15.0, "sun_energy_3d": 45.0, "sun_energy_7d": 105.0,
    "temp_avg": 20.0,
    "hum_avg": 70.0,
    "air_speed_avg": 2.0,
    "work_speed": 1.4,
    "work_rate": 1.0,
    "row_space": 0.75,
    "green_manure": "barley",
    "culti_type": "Rotaly",
    "mold_boad": "OFF",
}

if df is not None:
    green_choices = sorted(df["green_manure"].dropna().astype(str).unique()) if "green_manure" in df.columns else ["barley", "hairy vetch", "mix", "no"]
    culti_choices = sorted(df["culti_type"].dropna().astype(str).unique()) if "culti_type" in df.columns else ["tine", "Rotaly"]
    mold_choices = sorted(df["mold_boad"].dropna().astype(str).unique()) if "mold_boad" in df.columns else ["OFF", "On"]
else:
    green_choices = ["barley", "hairy vetch", "mix", "no"]
    culti_choices = ["tine", "Rotaly"]
    mold_choices = ["OFF", "On"]

ranges = {}
for c in numeric_cols:
    if df is not None and c in df.columns:
        ranges[c] = make_slider_range(df[c], defaults[c] * 0.5, defaults[c] * 1.5, defaults[c])
    else:
        if defaults[c] >= 0:
            ranges[c] = (0.0, max(1.0, defaults[c] * 2.0), defaults[c])
        else:
            ranges[c] = (defaults[c] * 2.0, defaults[c] * 0.5, defaults[c])

if df is not None and "weed_remain_gndvi" in df.columns:
    g_min = float(pd.to_numeric(df["weed_remain_gndvi"], errors="coerce").min())
    g_max = float(pd.to_numeric(df["weed_remain_gndvi"], errors="coerce").max())
else:
    g_min, g_max = 0.0, 100.0

if df is not None and "weed_after_dm" in df.columns:
    d_min = float(pd.to_numeric(df["weed_after_dm"], errors="coerce").min())
    d_max = float(pd.to_numeric(df["weed_after_dm"], errors="coerce").max())
else:
    d_min, d_max = 0.0, 200.0

reco_df = None
if os.path.exists(RECO_PATH):
    reco_df = pd.read_csv(RECO_PATH)
    if "pred_weed_after_dm" in reco_df.columns and (df is None or "weed_after_dm" not in df.columns):
        d_min = float(pd.to_numeric(reco_df["pred_weed_after_dm"], errors="coerce").min())
        d_max = float(pd.to_numeric(reco_df["pred_weed_after_dm"], errors="coerce").max())

# =========================
# 5. UI（サイドバー）
# =========================
st.sidebar.title("条件入力（ML_AL7）")
st.sidebar.caption("中耕前条件、作業条件、機械条件、気象条件を入力してください。")

use_log1p = st.sidebar.toggle("乾物重は log1p モデルを使用", value=True)
st.sidebar.caption("ON: drymatter_log1p_rf_model.joblib（出力は expm1 で g/m²に戻します）")

st.sidebar.markdown("### ① 中耕前条件")
weed_before_gndvi = st.sidebar.slider("weed_before_gndvi（%）", *ranges["weed_before_gndvi"])
weed_before_dm = st.sidebar.slider("weed_before_dm（g/m²）", *ranges["weed_before_dm"])
soy_cover_mean = st.sidebar.slider("soy_cover_mean（%）", *ranges["soy_cover_mean"])
soy_cover_sd = st.sidebar.slider("soy_cover_sd（被覆率SD）", *ranges["soy_cover_sd"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ② 作業条件")

try:
    default_rs = float(ranges["row_space"][2])
except Exception:
    default_rs = 0.75

default_rs = min(max(default_rs, 0.65), 0.78)

row_space = st.sidebar.slider(
    "row_space（m）",
    min_value=0.65,
    max_value=0.78,
    value=default_rs,
    step=0.01
)

work_speed = st.sidebar.slider("work_speed（m/s）", *ranges["work_speed"])
work_rate = st.sidebar.slider("work_rate（h/ha）", *ranges["work_rate"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ③ 機械・緑肥条件")
green_manure = st.sidebar.selectbox(
    "green_manure",
    green_choices,
    index=green_choices.index(defaults["green_manure"]) if defaults["green_manure"] in green_choices else 0
)
culti_type = st.sidebar.selectbox(
    "culti_type",
    culti_choices,
    index=culti_choices.index(defaults["culti_type"]) if defaults["culti_type"] in culti_choices else 0
)
mold_boad = st.sidebar.selectbox(
    "mold_boad",
    mold_choices,
    index=mold_choices.index(defaults["mold_boad"]) if defaults["mold_boad"] in mold_choices else 0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ④ 気象・積算")
rain_1d = st.sidebar.slider("rain_1d（mm）", *ranges["rain_1d"])
rain_3d = st.sidebar.slider("rain_3d（mm）", *ranges["rain_3d"])
rain_7d = st.sidebar.slider("rain_7d（mm）", *ranges["rain_7d"])

sun_time_1d = st.sidebar.slider("sun_time_1d（h）", *ranges["sun_time_1d"])
sun_time_3d = st.sidebar.slider("sun_time_3d（h）", *ranges["sun_time_3d"])
sun_time_7d = st.sidebar.slider("sun_time_7d（h）", *ranges["sun_time_7d"])

sun_energy_1d = st.sidebar.slider("sun_energy_1d（MJ/m²）", *ranges["sun_energy_1d"])
sun_energy_3d = st.sidebar.slider("sun_energy_3d（MJ/m²）", *ranges["sun_energy_3d"])
sun_energy_7d = st.sidebar.slider("sun_energy_7d（MJ/m²）", *ranges["sun_energy_7d"])

temp_avg = st.sidebar.slider("temp_avg（℃）", *ranges["temp_avg"])
hum_avg = st.sidebar.slider("hum_avg（%）", *ranges["hum_avg"])
air_speed_avg = st.sidebar.slider("air_speed_avg（m/s）", *ranges["air_speed_avg"])

until_culti_temp = st.sidebar.slider("until_culti_temp（℃・日）", *ranges["until_culti_temp"])
after_sowing_temp = st.sidebar.slider("after_sowing_temp（℃・日）", *ranges["after_sowing_temp"])

controls = {
    "weed_before_gndvi": weed_before_gndvi,
    "weed_before_dm": weed_before_dm,
    "soy_cover_mean": soy_cover_mean,
    "soy_cover_sd": soy_cover_sd,
    "row_space": row_space,
    "work_speed": work_speed,
    "work_rate": work_rate,
    "green_manure": green_manure,
    "culti_type": culti_type,
    "mold_boad": mold_boad,
    "rain_1d": rain_1d, "rain_3d": rain_3d, "rain_7d": rain_7d,
    "sun_time_1d": sun_time_1d, "sun_time_3d": sun_time_3d, "sun_time_7d": sun_time_7d,
    "sun_energy_1d": sun_energy_1d, "sun_energy_3d": sun_energy_3d, "sun_energy_7d": sun_energy_7d,
    "temp_avg": temp_avg,
    "hum_avg": hum_avg,
    "air_speed_avg": air_speed_avg,
    "until_culti_temp": until_culti_temp,
    "after_sowing_temp": after_sowing_temp,
}

# =========================
# 6. 予測
# =========================
row_dum = build_row_dum(
    controls=controls,
    feature_cols_expanded=feature_cols_expanded,
    numeric_cols=numeric_cols,
    cat_cols=cat_cols,
)

pred_g = float(rf_gndvi.predict(row_dum)[0])
pred_d = predict_drymatter(rf_dm_raw, rf_dm_log, row_dum, use_log1p=use_log1p)

g_norm = (pred_g - g_min) / (g_max - g_min + 1e-9)
d_norm = (pred_d - d_min) / (d_max - d_min + 1e-9)
score = 0.5 * g_norm + 0.5 * d_norm
score_clipped = float(np.clip(score, 0.0, 1.0))
bar_value = 1.0 - score_clipped

# =========================
# 7. SHAP準備
# =========================
background_X = get_background_encoded(
    data_path=DATA_PATH,
    feature_cols_expanded=feature_cols_expanded,
    numeric_cols=numeric_cols,
    cat_cols=cat_cols,
    max_samples=200,
)

g_explainer = None
d_explainer = None
g_shap_row = None
d_shap_row = None
g_shap_global = None
d_shap_global = None
feature_names = list(row_dum.columns)

try:
    g_explainer = get_tree_explainer(rf_gndvi)
    g_shap_row = compute_shap_values(g_explainer, row_dum)[0]

    if not background_X.empty:
        g_shap_global = compute_shap_values(g_explainer, background_X)
except Exception as e:
    g_shap_row = None
    g_shap_global = None
    g_shap_error = str(e)
else:
    g_shap_error = None

try:
    current_dm_model = rf_dm_log if use_log1p else rf_dm_raw
    d_explainer = get_tree_explainer(current_dm_model)
    d_shap_row = compute_shap_values(d_explainer, row_dum)[0]

    if not background_X.empty:
        d_shap_global = compute_shap_values(d_explainer, background_X)
except Exception as e:
    d_shap_row = None
    d_shap_global = None
    d_shap_error = str(e)
else:
    d_shap_error = None

# =========================
# 8. 画面レイアウト
# =========================
tab_pred, tab_map, tab_reco, tab_shap = st.tabs(
    ["予測結果", "ヒートマップ", "レコメンド一覧", "SHAP説明"]
)

with tab_pred:
    st.title("中耕除草効果予測アプリ（ML_AL7：NGRDI × 乾物重）")
    st.caption(f"モデル: NGRDI = Random Forest / Dry matter = {'Random Forest (log1p)' if use_log1p else 'Random Forest (raw)'}")
    st.caption("研究用デモ版")

    st.markdown("""
### このアプリについて
このアプリは、中耕除草前の雑草量・ダイズ被覆率・作業条件・気象条件などを入力すると、  
**中耕後の残草率** と **残草乾物重** を機械学習モデルで予測します。

### 出力される指標
- **予測残草率（NGRDIベース, %）**
- **予測残草乾物重（g/m²）**
- **総合スコア（0に近いほど良い条件）**

### 利用上の注意
このアプリの出力は、学習データに基づく**推定値**です。  
現場での判断や結論づけには、追加の検証と実測確認を併用してください。
""")

    st.markdown("#### 入力条件")
    display_controls = format_controls_for_display(controls)
    st.dataframe(display_controls, use_container_width=True)

    st.markdown("#### 予測結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("予測残草率（NGRDIベース, %）", f"{pred_g:.2f}")
    with col2:
        st.metric("予測残草乾物重（g/m²）", f"{pred_d:.2f}")
    with col3:
        st.metric("総合スコア", f"{score_clipped:.3f}")

    st.markdown("#### 総合評価")
    st.progress(bar_value)
    st.write(f"総合スコア: **{score_clipped:.3f}** （0に近いほど良い条件）")

    st.markdown("#### 結果のダウンロード")
    download_row = {
        **controls,
        "pred_weed_remain_gndvi": pred_g,
        "pred_weed_after_dm": pred_d,
        "g_norm": g_norm,
        "d_norm": d_norm,
        "total_score": score_clipped,
        "drymatter_model": "log1p" if use_log1p else "raw",
    }
    download_df = pd.DataFrame([download_row])
    csv_data = download_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="この条件と予測結果をCSVでダウンロード",
        data=csv_data,
        file_name="culti_prediction_result.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("※ 推定値です。現場適用や結論づけには追加検証を推奨します。")


# ---------- ヒートマップ ----------
def plot_heatmap_mean(df_hm, row_key, col_key, value_key, title, xlabel, ylabel):
    if df_hm is None or df_hm.empty:
        st.info("recommendation_grid_results.csv が見つからないため、ヒートマップを表示できません。")
        return

    tmp = df_hm.copy()
    tmp[row_key] = tmp[row_key]
    tmp[col_key] = tmp[col_key].astype(str)

    pivot = (
        tmp.groupby([col_key, row_key])[value_key]
        .mean()
        .reset_index()
        .pivot(index=col_key, columns=row_key, values=value_key)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{safe_float(c):.2f}" for c in pivot.columns], rotation=45, ha="right")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("平均 total_score（小さいほど良い）")
    st.pyplot(fig)


with tab_map:
    st.header("ヒートマップ（推薦結果ベース）")
    st.caption("値が小さいほど望ましい条件です。")

    if reco_df is None:
        st.info("recommendation_grid_results.csv が見つかりません。学習スクリプトで出力されたCSVを配置してください。")
    else:
        st.markdown("##### 条間 × 緑肥（平均 total_score）")
        plot_heatmap_mean(
            reco_df,
            "row_space",
            "green_manure",
            "total_score",
            "row_space × green_manure",
            "row_space (m)",
            "green_manure"
        )

        st.markdown("##### 条間 × 中耕機種類（平均 total_score）")
        plot_heatmap_mean(
            reco_df,
            "row_space",
            "culti_type",
            "total_score",
            "row_space × culti_type",
            "row_space (m)",
            "culti_type"
        )


with tab_reco:
    st.header("レコメンド一覧（total_score 小さい順）")
    st.caption("値が小さいほど望ましい条件です。")

    if reco_df is None:
        st.info("recommendation_grid_results.csv が見つかりません。")
    else:
        st.dataframe(
            reco_df.sort_values("total_score").head(100),
            use_container_width=True
        )


with tab_shap:
    st.header("SHAP説明")
    st.caption("SHAP値は、各入力項目が予測値をどの方向にどれだけ動かしたかを示す指標です。")

    st.markdown("""
- **プラス方向**：予測値を押し上げる要因  
- **マイナス方向**：予測値を押し下げる要因  
- **全体重要度**：学習データ全体で見た平均的な影響の大きさ  
""")

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "今回の条件（残草率）",
        "今回の条件（乾物重）",
        "全体重要度（残草率）",
        "全体重要度（乾物重）",
    ])

    with subtab1:
        st.subheader("今回の入力条件に対するSHAP（残草率モデル）")
        if g_shap_error is not None:
            st.warning(f"SHAP計算でエラーが発生しました（残草率モデル）: {g_shap_error}")
        elif g_shap_row is None:
            st.info("残草率モデルのSHAP値を表示できません。")
        else:
            st.write("上位の寄与項目ほど、この条件での予測残草率に強く影響しています。")
            plot_local_shap_bar(
                g_shap_row,
                feature_names=feature_names,
                top_n=12,
                title="Local SHAP (weed coverage model)"
            )

    with subtab2:
        st.subheader("今回の入力条件に対するSHAP（乾物重モデル）")
        if d_shap_error is not None:
            st.warning(f"SHAP計算でエラーが発生しました（乾物重モデル）: {d_shap_error}")
        elif d_shap_row is None:
            st.info("乾物重モデルのSHAP値を表示できません。")
        else:
            st.write("上位の寄与項目ほど、この条件での予測残草乾物重に強く影響しています。")
            plot_local_shap_bar(
                d_shap_row,
                feature_names=feature_names,
                top_n=12,
                title=f"Local SHAP（weed dry weight model: {'log1p' if use_log1p else 'raw'}）"
            )

    with subtab3:
        st.subheader("学習データ全体でのSHAP重要度（残草率モデル）")
        if g_shap_error is not None:
            st.warning(f"SHAP計算でエラーが発生しました（残草率モデル）: {g_shap_error}")
        elif g_shap_global is None or len(g_shap_global) == 0:
            st.info("全体SHAP重要度を表示できません。")
        else:
            st.write("平均|SHAP|が大きい項目ほど、学習データ全体で重要な変数です。")
            plot_global_shap_bar(
                g_shap_global,
                feature_names=feature_names,
                top_n=15,
                title="全体重要度（残草率モデル）"
            )

    with subtab4:
        st.subheader("学習データ全体でのSHAP重要度（乾物重モデル）")
        if d_shap_error is not None:
            st.warning(f"SHAP計算でエラーが発生しました（乾物重モデル）: {d_shap_error}")
        elif d_shap_global is None or len(d_shap_global) == 0:
            st.info("全体SHAP重要度を表示できません。")
        else:
            st.write("平均|SHAP|が大きい項目ほど、学習データ全体で重要な変数です。")
            plot_global_shap_bar(
                d_shap_global,
                feature_names=feature_names,
                top_n=15,
                title=f"全体重要度（乾物重モデル: {'log1p' if use_log1p else 'raw'}）"
            )