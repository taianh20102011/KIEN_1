import os
import re
import math
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

# Optional ML backends: the app still runs if one/both are unavailable.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Kiến.xlsx"
if not EXCEL_FILE.exists():
    candidate = BASE_DIR / "Kiến(1).xlsx"
    if candidate.exists():
        EXCEL_FILE = candidate

CSV_HISTORY = BASE_DIR / "lich_su_quan_sat.csv"
ANTWIKI_CACHE_FILE = BASE_DIR / "antwiki_cache.json"

DEFAULT_LAT = 10.8231
DEFAULT_LON = 106.6297



# ============================================================
# ANTWIKI API INTEGRATION
# ============================================================
def extract_english_months(text: str) -> list[int]:
    """Bóc tách tháng bay giao hoan (bằng tiếng Anh) từ đoạn văn bản AntWiki"""
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
    }
    found = set()
    for word in re.findall(r'\b[A-Za-z]+\b', text.lower()):
        if word in months:
            found.add(months[word])
    return sorted(list(found))

def fetch_antwiki_api(species_name: str) -> dict:
    """Gọi API Antwiki để kéo thông tin sinh học"""
    url = "https://www.antwiki.org/wiki/api.php"
    title = species_name.strip().capitalize().replace(" ", "_")
    params = {
        "action": "query", "prop": "revisions", "titles": title,
        "rvprop": "content", "rvslots": "main", "format": "json"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = list(pages.values())[0]
        
        if "missing" in page:
            return {"error": "Missing"}
            
        wikitext = page["revisions"][0]["slots"]["main"]["*"]
        
        flight_sentences = []
        for sentence in re.split(r'(?<=[.!?]) +', wikitext):
            if re.search(r'(nuptial flight|mating flight|swarm|alates|after rain)', sentence, re.IGNORECASE):
                clean_s = re.sub(r"<.*?>", "", sentence)
                clean_s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", clean_s)
                flight_sentences.append(clean_s.strip())
                
        flight_data = " | ".join(flight_sentences)
        months = extract_english_months(flight_data)
        
        return {
            "species": species_name,
            "flight_data": flight_data,
            "months": months
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# EXCEL KNOWLEDGE BASE
# ============================================================
def parse_month_range(text: str) -> list[int]:
    result = set()
    nums = [int(x) for x in re.findall(r"(?:tháng|T)\s*(\d{1,2})", text, re.I)]
    if len(nums) >= 2:
        start, end = nums[0], nums[1]
        if 1 <= start <= 12 and 1 <= end <= 12:
            if start <= end:
                result.update(range(start, end + 1))
            else:
                result.update(list(range(start, 13)) + list(range(1, end + 1)))
    elif len(nums) == 1 and 1 <= nums[0] <= 12:
        result.add(nums[0])
    return sorted(result)

def normalize(s: str) -> str:
    s = str(s).lower().replace("đ", "d")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_excel(path: Path) -> dict:
    if not path.exists():
        return {"monthly": {}, "species": {}, "no_month": [], "environment": ""}

    try:
        df = pd.read_excel(path, sheet_name=0, header=None)
        rows = df.fillna("").astype(str)

        monthly = {}
        for _, row in rows.iterrows():
            a = row.iloc[0].strip()
            b = row.iloc[1].strip() if len(row) > 1 else ""
            m = re.fullmatch(r"T(1[0-2]|[1-9])", a, flags=re.I)
            if m:
                month = int(m.group(1))
                monthly[month] = [x.strip() for x in re.split(r",\s*", b) if x.strip()]

        species = {}
        for i in range(len(rows)):
            text = rows.iloc[i, 0].strip()
            if not text:
                continue
            if text.endswith(":") and "(" in text and i + 1 < len(rows):
                name = text[:-1].strip()
                time_text = rows.iloc[i + 1, 0].strip()
                mm = re.search(r"Thời gian:\s*(.+?)\.", time_text, flags=re.I)
                if mm:
                    species[name] = {
                        "time_text": mm.group(1).strip(),
                        "months": parse_month_range(mm.group(1)),
                    }

        no_month = []
        for i, text in enumerate(rows.iloc[:, 0].tolist()):
            if "KIẾN KHÔNG CÓ THÁNG BAY CỰ THỂ" in text.upper():
                if i + 1 < len(rows):
                    no_month = [
                        x.strip() for x in re.split(r",\s*", rows.iloc[i + 1, 0]) if x.strip()
                    ]

        environment = " ".join(x.strip() for x in rows.iloc[:, 0].tolist() if x.strip())
        return {
            "monthly": monthly,
            "species": species,
            "no_month": no_month,
            "environment": environment,
        }
    except Exception as e:
        print(f"Lỗi đọc Excel: {e}")
        return {"monthly": {}, "species": {}, "no_month": [], "environment": ""}

def species_month_score(species: str | None, month: int, data: dict, antwiki_cache: dict) -> tuple[float, str]:
    """Cập nhật để trả về tuple gồm: (Điểm số, Lý do) có tích hợp AntWiki"""
    if not species or species == "Tất cả loài":
        if month in (4, 5, 6, 7): return 1.0, "Quy luật chung (Cao điểm)"
        if month in (2, 3, 8): return 0.65, "Quy luật chung (Mùa phụ)"
        return 0.25, "Quy luật chung (Mùa thấp điểm)"

    q = normalize(species)
    
    # 1. Khớp từ Excel
    for name, info in data["species"].items():
        if q in normalize(name) or normalize(name) in q:
            if month in info["months"]:
                return 1.0, "Đúng mùa bay (Theo Kiến.xlsx)"
            else:
                return 0.12, "Lệch mùa bay (Theo Kiến.xlsx)"

    for m, names in data["monthly"].items():
        if m == month:
            for name in names:
                if q in normalize(name) or normalize(name) in q:
                    return 1.0, "Đúng tháng bay (Theo Kiến.xlsx)"

    # 2. Khớp từ dữ liệu API AntWiki
    for name, info in antwiki_cache.items():
        if q in normalize(name) or normalize(name) in q:
            if month in info.get("months", []):
                return 0.95, "Dữ liệu sinh học cào từ AntWiki"

    # 3. Mặc định cho loài không có tháng
    for name in data["no_month"]:
        if q in normalize(name) or normalize(name) in q:
            return 0.55, "Chưa xác định rõ tháng bay"

    return 0.35, "Không có dữ liệu tháng"


# ============================================================
# WEATHER + PHYSICS FEATURES
# ============================================================
WEATHER_HOURLY = ",".join([
    "temperature_2m", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "weather_code", "wind_speed_10m", "pressure_msl",
    "dew_point_2m", "uv_index", "shortwave_radiation",
])

def fetch_weather_forecast(lat: float, lon: float) -> list[dict]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "hourly": WEATHER_HOURLY,
        "forecast_days": 2, "past_days": 1, "timezone": "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        h = r.json()["hourly"]

        result = []
        for i, iso in enumerate(h["time"]):
            result.append({
                "time": datetime.fromisoformat(iso),
                "temp": h["temperature_2m"][i],
                "humidity": h["relative_humidity_2m"][i],
                "rain_prob": h["precipitation_probability"][i],
                "rain": h["precipitation"][i],
                "code": h["weather_code"][i],
                "wind": h["wind_speed_10m"][i],
                "pressure_msl": h["pressure_msl"][i],
                "dew_point": h["dew_point_2m"][i],
                "uv_index": h["uv_index"][i],
                "shortwave_radiation": h["shortwave_radiation"][i],
            })
        return add_physics_features(result)
    except Exception as e:
        print(f"Lỗi kết nối thời tiết: {e}")
        return []

def add_physics_features(rows: list[dict]) -> list[dict]:
    if not rows: return rows

    rows = sorted(rows, key=lambda x: x["time"])
    by_time = {x["time"]: x for x in rows}

    for i, row in enumerate(rows):
        prev3 = by_time.get(row["time"] - timedelta(hours=3))
        if prev3:
            row["delta_temp"] = row["temp"] - prev3["temp"]
            row["delta_humidity"] = row["humidity"] - prev3["humidity"]
            row["delta_pressure"] = row["pressure_msl"] - prev3["pressure_msl"]
            row["pressure_drop_3h"] = max(0.0, -row["delta_pressure"])
        else:
            row["delta_temp"] = 0.0
            row["delta_humidity"] = 0.0
            row["delta_pressure"] = 0.0
            row["pressure_drop_3h"] = 0.0

        row["dew_point_spread"] = max(0.0, row["temp"] - row["dew_point"])
        row["condensation_index"] = max(0.0, min(1.0, 1.0 - row["dew_point_spread"] / 8.0))
        row["post_rain_index"] = max(0.0, min(1.0, 0.45 * min(row["rain_prob"] / 100.0, 1.0) + 0.30 * max(0.0, min(row["delta_humidity"] / 15.0, 1.0)) + 0.25 * row["condensation_index"]))
        row["heat_solar_index"] = max(0.0, min(1.0, 0.50 * max(0.0, min((row["temp"] - 27.0) / 7.0, 1.0)) + 0.50 * max(0.0, min(row["shortwave_radiation"] / 700.0, 1.0))))

    return rows


# ============================================================
# HYBRID ENSEMBLE
# ============================================================
FEATURES = [
    "month", "hour", "temp", "humidity", "rain_prob", "rain", "wind",
    "pressure_msl", "dew_point", "uv_index", "shortwave_radiation",
    "delta_temp", "delta_humidity", "delta_pressure", "pressure_drop_3h", 
    "dew_point_spread", "condensation_index", "post_rain_index", "heat_solar_index",
]

def numeric_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "month": 1, "hour": 12, "temp": 28.0, "humidity": 75.0,
        "rain_prob": 0.0, "rain": 0.0, "wind": 5.0,
        "pressure_msl": 1010.0, "dew_point": 23.0, "uv_index": 0.0,
        "shortwave_radiation": 0.0, "delta_temp": 0.0,
        "delta_humidity": 0.0, "delta_pressure": 0.0,
        "pressure_drop_3h": 0.0, "dew_point_spread": 5.0,
        "condensation_index": 0.0, "post_rain_index": 0.0,
        "heat_solar_index": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns: out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    return out

class HybridEnsemble:
    def __init__(self):
        self.models = []
        self.is_trained = False
        self.n_samples = 0

        if HAS_XGBOOST:
            self.models.append(("XGBoost", XGBClassifier(n_estimators=260, max_depth=5, learning_rate=0.045, subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=2)))
        if HAS_CATBOOST:
            self.models.append(("CatBoost", CatBoostClassifier(iterations=260, depth=6, learning_rate=0.05, loss_function="Logloss", verbose=False, random_seed=42)))
        self.models.append(("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)))

    @property
    def ml_weight(self) -> float:
        return min(0.80, max(0.0, (self.n_samples - 10) / 50.0))

    @property
    def expert_weight(self) -> float:
        return 1.0 - self.ml_weight

    def train(self, history: pd.DataFrame):
        self.is_trained = False
        self.n_samples = len(history)

        if len(history) < 10: return
        df = numeric_history(history)
        if len(df) < 10 or df["label"].nunique() < 2: return

        X = df[FEATURES]
        y = df["label"].astype(int)

        trained = []
        for name, model in self.models:
            try:
                model.fit(X, y)
                trained.append((name, model))
            except Exception as e:
                print(f"{name} không huấn luyện được: {e}")

        self.models = trained
        self.is_trained = bool(trained)

    def predict_ml(self, row: dict) -> float | None:
        if not self.is_trained: return None
        values = {f: row.get(f, 0.0) for f in FEATURES}
        values["month"] = int(values["month"])
        values["hour"] = int(values["hour"])
        X = pd.DataFrame([values])[FEATURES]

        probs = []
        for name, model in self.models:
            try: probs.append(float(model.predict_proba(X)[0][1]))
            except Exception as e: print(f"Lỗi predict {name}: {e}")

        return sum(probs) / len(probs) if probs else None

    def status_text(self) -> str:
        backend = " + ".join(name for name, _ in self.models) if self.models else "chưa có backend"
        if not self.is_trained: return f"Heuristic Expert • cần ≥10 mẫu • Backend: {backend}"
        return f"Hybrid Soft Voting • ML {self.ml_weight*100:.0f}% / Expert {self.expert_weight*100:.0f}% • {backend} • {self.n_samples} mẫu"


# ============================================================
# BIOLOGICAL EXPERT SYSTEM
# ============================================================
def expert_score(h: dict, selected_species: str, excel_data: dict, antwiki_cache: dict) -> tuple[float, list[str]]:
    month = h["time"].month
    hour = h["time"].hour
    temp = float(h["temp"])
    humidity = float(h["humidity"])
    rain_prob = float(h["rain_prob"])
    rain = float(h["rain"])
    wind = float(h["wind"])

    s_month, month_reason = species_month_score(selected_species, month, excel_data, antwiki_cache)
    score = 0.0
    reasons = [month_reason] # Tự động nhúng lý do từ AntWiki hoặc Excel

    if 25 <= temp <= 33: score += 0.16; reasons.append("nhiệt độ 25–33°C")
    elif 23 <= temp <= 35: score += 0.08
    if humidity >= 80: score += 0.16; reasons.append("ẩm rất cao ≥80%")
    elif humidity >= 70: score += 0.09
    if 14 <= hour <= 16: score += 0.10; reasons.append("khung mưa rào 14–16h")
    if 17 <= hour <= 20: score += 0.20; reasons.append("khung chiều tối 17–20h")
    if rain_prob >= 70 or rain > 0: score += 0.10; reasons.append("tín hiệu mưa")
    
    if h["pressure_drop_3h"] >= 2.0: score += 0.10; reasons.append("áp suất giảm ≥2 hPa/3h")
    elif h["pressure_drop_3h"] >= 1.0: score += 0.05; reasons.append("áp suất đang giảm")
    if h["dew_point_spread"] <= 2.0: score += 0.10; reasons.append("điểm sương sát nhiệt độ")
    elif h["dew_point_spread"] <= 4.0: score += 0.05
    if h["condensation_index"] >= 0.75: score += 0.05; reasons.append("độ ngưng tụ cao")
    if h["delta_humidity"] >= 8: score += 0.08; reasons.append("độ ẩm tăng mạnh so với 3h trước")
    elif h["delta_humidity"] >= 4: score += 0.04
    if h["delta_temp"] <= -1.5: score += 0.05; reasons.append("nhiệt độ hạ nhanh sau nóng/mưa")
    if h["post_rain_index"] >= 0.70: score += 0.10; reasons.append("mẫu hình sau mưa ẩm oi")
    if h["uv_index"] >= 6 and h["shortwave_radiation"] >= 500: score += 0.04; reasons.append("ban ngày nắng/UV mạnh")
    if wind >= 25: score -= 0.08; reasons.append("gió khá mạnh")
    elif wind <= 12: score += 0.03

    total = max(0.0, min(1.0, 0.55 * s_month + 0.45 * min(score / 1.0, 1.0)))
    return total, reasons


# ============================================================
# SLIDING WINDOW PEAK SEARCH
# ============================================================
def find_peak_window(scored: list[tuple[float, dict, list[str]]], width: int = 3):
    if not scored: return None
    scored = sorted(scored, key=lambda x: x[1]["time"])
    n = len(scored)
    width = max(1, min(width, n))
    best = None

    for i in range(n - width + 1):
        window = scored[i:i + width]
        scores = [x[0] for x in window]
        mean_score = sum(scores) / len(scores)
        peak_score = max(scores)
        continuity = 1.0 - (max(scores) - min(scores))
        objective = 0.60 * mean_score + 0.30 * peak_score + 0.10 * continuity

        candidate = (objective, window)
        if best is None or candidate[0] > best[0]: best = candidate

    objective, window = best
    start = window[0][1]["time"]
    end = window[-1][1]["time"] + timedelta(hours=1)
    peak = max(window, key=lambda x: x[0])
    return {"objective": objective, "start": start, "end": end, "peak": peak, "window": window}


