
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import threading

# Import the original biology/weather/ML engine from the adapted core.
from ant_engine import (
    load_excel, fetch_weather_forecast, fetch_antwiki_api,
    HybridEnsemble, expert_score, find_peak_window,
    EXCEL_FILE, ANTWIKI_CACHE_FILE, CSV_HISTORY,
    DEFAULT_LAT, DEFAULT_LON
)

st.set_page_config(
    page_title="ANTS FLY PRO CLOUD",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {padding-top: 1rem;}
.block-container {max-width: 1450px; padding-top: 1.2rem;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 22px;
    background: linear-gradient(135deg, #111827, #172554);
    border: 1px solid rgba(255,255,255,.10);
    margin-bottom: 1rem;
}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.35rem 0 0; color:#a9b7d0;}
.card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(30,41,59,.72);
    border: 1px solid rgba(255,255,255,.08);
    min-height: 105px;
}
.card .label {font-size:.75rem; color:#94a3b8; font-weight:700;}
.card .value {font-size:1.65rem; font-weight:800; margin-top:.25rem;}
.small-muted {color:#94a3b8; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def get_excel_data():
    return load_excel(EXCEL_FILE)

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon):
    return fetch_weather_forecast(float(lat), float(lon))

@st.cache_data(ttl=86400, show_spinner=False)
def get_antwiki(species):
    return fetch_antwiki_api(species)

def load_cache():
    if ANTWIKI_CACHE_FILE.exists():
        try:
            return json.loads(ANTWIKI_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        ANTWIKI_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

def load_history():
    if CSV_HISTORY.exists():
        try:
            return pd.read_csv(CSV_HISTORY)
        except Exception:
            pass
    return pd.DataFrame()

def save_feedback(label, species, weather):
    now = datetime.now()
    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "species": species,
        "month": now.month,
        "hour": now.hour,
        "temp": weather.get("temp", 28.0),
        "humidity": weather.get("humidity", 75.0),
        "rain_prob": weather.get("rain_prob", 0.0),
        "rain": weather.get("rain", 0.0),
        "wind": weather.get("wind", 5.0),
        "pressure_msl": weather.get("pressure_msl", 1010.0),
        "dew_point": weather.get("dew_point", 23.0),
        "uv_index": weather.get("uv_index", 0.0),
        "shortwave_radiation": weather.get("shortwave_radiation", 0.0),
        "delta_temp": weather.get("delta_temp", 0.0),
        "delta_humidity": weather.get("delta_humidity", 0.0),
        "delta_pressure": weather.get("delta_pressure", 0.0),
        "pressure_drop_3h": weather.get("pressure_drop_3h", 0.0),
        "dew_point_spread": weather.get("dew_point_spread", 5.0),
        "condensation_index": weather.get("condensation_index", 0.0),
        "post_rain_index": weather.get("post_rain_index", 0.0),
        "heat_solar_index": weather.get("heat_solar_index", 0.0),
        "label": int(label),
    }
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_HISTORY, index=False, encoding="utf-8-sig")
    return df

def hybrid_score(engine, h, species, excel_data, antwiki_cache):
    expert, reasons = expert_score(h, species, excel_data, antwiki_cache)
    row = {"month": h["time"].month, "hour": h["time"].hour, **h}
    ml = engine.predict_ml(row)
    if ml is None:
        return expert, reasons, expert, None
    w_ml = engine.ml_weight
    final = w_ml * ml + (1.0 - w_ml) * expert
    reasons = [f"Hybrid: ML {w_ml*100:.0f}% + Expert {(1-w_ml)*100:.0f}%", *reasons[:5]]
    return final, reasons, expert, ml

# ---------- Initial data ----------
excel_data = get_excel_data()
species_list = ["Tất cả loài"] + list(excel_data.get("species", {}).keys())

if "lat" not in st.session_state:
    st.session_state.lat = DEFAULT_LAT
if "lon" not in st.session_state:
    st.session_state.lon = DEFAULT_LON
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "antwiki_cache" not in st.session_state:
    st.session_state.antwiki_cache = load_cache()

engine = HybridEnsemble()
history = st.session_state.history
if not history.empty:
    engine.train(history)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🐜 ANTS FLY PRO")
    st.caption("BIOLOGY × PHYSICS × HYBRID AI")
    st.divider()

    st.markdown("### 📍 Vị trí")
    lat = st.number_input("Latitude", value=float(st.session_state.lat), format="%.6f")
    lon = st.number_input("Longitude", value=float(st.session_state.lon), format="%.6f")
    st.session_state.lat, st.session_state.lon = lat, lon

    st.markdown("### 🐜 Loài kiến")
    species = st.selectbox("Chọn loài", species_list)

    if st.button("⟳ CẬP NHẬT THỜI TIẾT", use_container_width=True):
        get_weather.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📝 GHI NHẬN THỰC TẾ")
    st.caption("Phản hồi thực tế được dùng để tăng trọng số ML.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐜 BAY", use_container_width=True):
            if "current_weather" in st.session_state:
                st.session_state.history = save_feedback(
                    1, species, st.session_state.current_weather
                )
                st.success("Đã ghi nhận: KIẾN BAY")
                st.rerun()
    with c2:
        if st.button("🚫 KHÔNG", use_container_width=True):
            if "current_weather" in st.session_state:
                st.session_state.history = save_feedback(
                    0, species, st.session_state.current_weather
                )
                st.warning("Đã ghi nhận: KHÔNG BAY")
                st.rerun()

    st.divider()
    st.markdown("### 🤖 Trạng thái AI")
    st.info(engine.status_text())
    st.caption(f"Số mẫu lịch sử: {len(history)}")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🐜 ANTS FLY PRO CLOUD</h1>
    <p>Hybrid Biology + Physics + XGBoost + CatBoost + HistGradientBoosting + AntWiki</p>
</div>
""", unsafe_allow_html=True)

# ---------- Weather ----------
with st.spinner("Đang lấy dữ liệu thời tiết..."):
    hourly = get_weather(lat, lon)

if not hourly:
    st.error("Không lấy được dữ liệu thời tiết. Kiểm tra kết nối/API.")
    st.stop()

now = datetime.now()
start = now.replace(minute=0, second=0, microsecond=0)
future = [x for x in hourly if x["time"] >= start][:24]

if not future:
    st.warning("Không có dữ liệu dự báo 24 giờ.")
    st.stop()

current = future[0]
st.session_state.current_weather = dict(current)

# ---------- Prediction ----------
scored = []
for h in future:
    sc, reasons, expert, ml = hybrid_score(
        engine, h, species, excel_data, st.session_state.antwiki_cache
    )
    scored.append((sc, h, reasons))

peak = max(scored, key=lambda x: x[0])
best_sc, best_h, best_reasons = peak
window = find_peak_window(scored, width=3)

window_str = (
    f"{window['start']:%H:%M} – {window['end']:%H:%M}"
    if window else best_h["time"].strftime("%H:%M")
)

# ---------- KPI cards ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="card"><div class="label">XÁC SUẤT ĐỈNH</div>'
        f'<div class="value">{best_sc*100:.1f}%</div></div>',
        unsafe_allow_html=True
    )
with k2:
    st.markdown(
        f'<div class="card"><div class="label">PEAK WINDOW</div>'
        f'<div class="value">{window_str}</div></div>',
        unsafe_allow_html=True
    )
with k3:
    st.markdown(
        f'<div class="card"><div class="label">GIỜ ĐỈNH</div>'
        f'<div class="value">{best_h["time"]:%H:%M}</div></div>',
        unsafe_allow_html=True
    )
with k4:
    st.markdown(
        f'<div class="card"><div class="label">ML / EXPERT</div>'
        f'<div class="value">{engine.ml_weight*100:.0f}% / {engine.expert_weight*100:.0f}%</div></div>',
        unsafe_allow_html=True
    )

st.write("")

# ---------- Current conditions ----------
st.subheader("🌦️ Điều kiện hiện tại / giờ gần nhất")
w1, w2, w3, w4, w5, w6 = st.columns(6)
w1.metric("🌡️ Nhiệt độ", f'{current["temp"]:.1f} °C')
w2.metric("💧 Độ ẩm", f'{current["humidity"]:.0f}%')
w3.metric("🌧️ Mưa", f'{current["rain_prob"]:.0f}%')
w4.metric("🌀 Gió", f'{current["wind"]:.1f} km/h')
w5.metric("📉 Áp suất", f'{current["pressure_msl"]:.1f} hPa')
w6.metric("💦 Điểm sương", f'{current["dew_point"]:.1f} °C')

with st.expander("⚙️ Physics Features", expanded=False):
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("ΔT / 3h", f'{current["delta_temp"]:+.1f} °C')
    p2.metric("ΔRH / 3h", f'{current["delta_humidity"]:+.1f}%')
    p3.metric("Giảm áp / 3h", f'{current["pressure_drop_3h"]:.1f} hPa')
    p4.metric("Ngưng tụ", f'{current["condensation_index"]:.2f}')

    p5, p6, p7 = st.columns(3)
    p5.metric("Sau mưa", f'{current["post_rain_index"]:.2f}')
    p6.metric("UV", f'{current["uv_index"]:.1f}')
    p7.metric("Bức xạ", f'{current["shortwave_radiation"]:.0f} W/m²')

# ---------- Chart ----------
st.subheader("📈 Xác suất kiến bay — 24 giờ")
chart_df = pd.DataFrame({
    "Giờ": [h["time"] for _, h, _ in scored],
    "Xác suất (%)": [s * 100 for s, _, _ in scored]
}).set_index("Giờ")
st.line_chart(chart_df, y="Xác suất (%)", height=360)

# ---------- Detailed table ----------
st.subheader("📊 Dự báo chi tiết 24 giờ")
table_rows = []
for sc, h, reasons in scored:
    level = (
        "🔥 Rất cao" if sc >= 0.80 else
        "🟠 Cao" if sc >= 0.65 else
        "🟡 Trung bình" if sc >= 0.45 else
        "⚪ Thấp"
    )
    table_rows.append({
        "Giờ": h["time"].strftime("%d/%m %H:00"),
        "°C": round(h["temp"], 1),
        "RH%": round(h["humidity"]),
        "MSL hPa": round(h["pressure_msl"], 1),
        "Dew °C": round(h["dew_point"], 1),
        "Mưa %": round(h["rain_prob"]),
        "Gió km/h": round(h["wind"], 1),
        "AI": f"{sc*100:.1f}%",
        "Mức": level,
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ---------- Reasons ----------
st.subheader("🧠 Vì sao AI chọn khung giờ này?")
st.write(" • ".join(best_reasons))

# ---------- AntWiki ----------
with st.expander("🔬 AntWiki / dữ liệu sinh học"):
    st.write(f"Loài đang chọn: **{species}**")
    if species != "Tất cả loài":
        if species in st.session_state.antwiki_cache:
            data = st.session_state.antwiki_cache[species]
            st.write(data.get("flight_data", "Không có mô tả"))
            st.write("Tháng tìm thấy:", data.get("months", []))
        else:
            st.caption("Loài này chưa có cache AntWiki. Dữ liệu Excel vẫn được ưu tiên.")

# ---------- Footer ----------
st.divider()
st.caption(
    "ANTS FLY PRO CLOUD • Dữ liệu thời tiết từ Open-Meteo • "
    "Dự báo là xác suất/ước lượng, không phải bảo đảm kiến sẽ bay."
)
