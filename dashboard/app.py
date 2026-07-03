"""
Ozone Sorting Center Digital Twin - Ящик Шрёдингера
Интерактивный дашборд для имитационной модели сортировочного центра.

Запуск:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================================
# Конфигурация страницы
# ============================================================================

st.set_page_config(
    page_title="Ozone Sorting Center Digital Twin",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = "data/out"

# ============================================================================
# Игровой CSS — синяя палитра, клеточный фон, пиксельный шрифт
# ============================================================================

GAME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

/* --- Клеточный синий фон --- */
.stApp {
    background-color: #0a1428;
    background-image:
        linear-gradient(rgba(64, 156, 255, 0.15) 1px, transparent 1px),
        linear-gradient(90deg, rgba(64, 156, 255, 0.15) 1px, transparent 1px),
        linear-gradient(rgba(64, 156, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(64, 156, 255, 0.05) 1px, transparent 1px);
    background-size: 80px 80px, 80px 80px, 16px 16px, 16px 16px;
    color: #cfe4ff;
}

/* --- Заголовки: пиксельный шрифт --- */
h1, h2, h3, h4 {
    font-family: 'Press Start 2P', monospace !important;
    color: #7cc0ff !important;
    text-shadow: 2px 2px 0px #0a1428, 3px 3px 0px #1a3a6a;
    letter-spacing: 1px;
    line-height: 1.6 !important;
}
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1.0rem !important; }
h4 { font-size: 0.85rem !important; }

/* --- Основной текст --- */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    font-family: 'VT323', 'Courier New', monospace;
    font-size: 1.05rem;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background-color: #061024;
    background-image:
        linear-gradient(rgba(64, 156, 255, 0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(64, 156, 255, 0.12) 1px, transparent 1px);
    background-size: 16px 16px;
    border-right: 3px solid #409cff;
}
[data-testid="stSidebar"] * {
    color: #cfe4ff !important;
}

/* --- Метрики: игровые панели --- */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #0f2447 0%, #0a1830 100%);
    border: 3px solid #409cff;
    border-radius: 0;
    padding: 16px;
    box-shadow: 4px 4px 0 #1a3a6a, inset 0 0 0 1px #7cc0ff;
    transition: transform 0.1s;
}
[data-testid="stMetric"]:hover {
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0 #1a3a6a, inset 0 0 0 1px #7cc0ff;
}
[data-testid="stMetricLabel"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    color: #7cc0ff !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 1.3rem !important;
    color: #ffffff !important;
    text-shadow: 0 0 8px #409cff, 2px 2px 0 #0a1428;
}
[data-testid="stMetricDelta"] {
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
}

/* --- Кнопки: пиксельные --- */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.7rem !important;
    background: linear-gradient(180deg, #1e5bb0 0%, #0f3a7a 100%);
    color: #ffffff !important;
    border: 3px solid #7cc0ff;
    border-radius: 0;
    padding: 10px 18px;
    box-shadow: 3px 3px 0 #1a3a6a, inset 0 0 0 1px #cfe4ff;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(180deg, #2a72d0 0%, #1e5bb0 100%);
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 #1a3a6a, inset 0 0 0 1px #ffffff;
    color: #ffffff !important;
}
.stButton > button:active {
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0 #1a3a6a;
}

/* --- Селекты и инпуты --- */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #0a1830 !important;
    color: #cfe4ff !important;
    border: 2px solid #409cff !important;
    border-radius: 0 !important;
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
}

/* --- Вкладки: игровое меню --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: transparent;
    border-bottom: 3px solid #409cff;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.65rem !important;
    background: linear-gradient(180deg, #0f2447 0%, #0a1830 100%);
    color: #7cc0ff !important;
    border: 3px solid #1a3a6a;
    border-bottom: none;
    border-radius: 0;
    padding: 12px 18px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #2a72d0 0%, #1e5bb0 100%) !important;
    color: #ffffff !important;
    border-color: #7cc0ff !important;
    box-shadow: inset 0 0 0 1px #cfe4ff;
}

/* --- Заголовок страницы --- */
.stApp h1:first-of-type {
    text-align: center;
    padding: 20px;
    background: linear-gradient(180deg, #0f2447 0%, #0a1830 100%);
    border: 4px solid #409cff;
    box-shadow: inset 0 0 0 2px #1a3a6a, 6px 6px 0 #1a3a6a;
    margin-bottom: 24px;
}

/* --- Alerts --- */
.stAlert {
    background-color: #0f2447 !important;
    border: 3px solid #409cff !important;
    border-radius: 0 !important;
    color: #cfe4ff !important;
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
    box-shadow: 3px 3px 0 #1a3a6a;
}

/* --- Expander --- */
.streamlit-expanderHeader {
    background-color: #0f2447 !important;
    border: 2px solid #409cff !important;
    border-radius: 0 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.65rem !important;
    color: #7cc0ff !important;
}

/* --- Caption --- */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'VT323', monospace !important;
    color: #7cc0ff !important;
    font-size: 1.05rem !important;
}

/* --- DataFrame --- */
.stDataFrame {
    border: 2px solid #409cff;
    border-radius: 0;
}

/* --- Разделитель --- */
hr {
    border: none;
    border-top: 3px dashed #409cff;
    margin: 24px 0;
    opacity: 0.6;
}

/* --- Plotly wrapper --- */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, #0f2447 0%, #0a1830 100%);
    border: 3px solid #409cff;
    padding: 8px;
    box-shadow: 4px 4px 0 #1a3a6a;
}
</style>
"""

st.markdown(GAME_CSS, unsafe_allow_html=True)

# Игровая палитра для графиков
GAME_COLORS = ["#7cc0ff", "#409cff", "#1e5bb0", "#a8d8ff", "#2a72d0", "#cfe4ff"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10, 24, 48, 0.6)",
    font=dict(family="VT323, monospace", size=16, color="#cfe4ff"),
    xaxis=dict(gridcolor="rgba(64, 156, 255, 0.2)", zerolinecolor="#409cff"),
    yaxis=dict(gridcolor="rgba(64, 156, 255, 0.2)", zerolinecolor="#409cff"),
)


# ============================================================================
# Кэшированные загрузчики
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Ошибка чтения {path}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Ошибка чтения {path}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_all(data_dir: str) -> dict:
    base = Path(data_dir)
    return {
        "ops_min": load_csv(str(base / "operations_minutely.csv")),
        "ops_agg": load_csv(str(base / "operations_agg.csv")),
        "queues": load_csv(str(base / "queues.csv")),
        "utilization": load_csv(str(base / "utilization.csv")),
        "summary": load_json(str(base / "summary.json")),
    }


# ============================================================================
# Утилиты
# ============================================================================

def format_seconds(sec) -> str:
    if sec is None:
        return "—"
    try:
        sec = int(sec)
    except Exception:
        return "—"
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h} ч {m} мин"
    if m > 0:
        return f"{m} мин {s} с"
    return f"{s} с"


def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    if isinstance(x, float):
        return f"{x:,.2f}".replace(",", " ")
    return str(x)


def is_monotonic_growing(series: pd.Series) -> bool:
    """Грубая проверка монотонного роста очереди."""
    s = series.dropna().values
    if len(s) < 10:
        return False
    diffs = np.diff(s)
    frac_nonneg = np.mean(diffs >= -1e-9)
    growth = (s[-1] - s[0]) / (abs(s[0]) + 1e-9)
    return frac_nonneg > 0.9 and growth > 0.5


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("[ НАСТРОЙКИ ]")


def find_data_dirs() -> list[str]:
    candidates = []
    data_root = Path("data")
    if data_root.exists():
        for p in sorted(data_root.iterdir()):
            if p.is_dir() and p.name.startswith("out"):
                if any(p.iterdir()):
                    candidates.append(str(p))
    return candidates


available_dirs = find_data_dirs()
if available_dirs:
    default_idx = 0
    if DEFAULT_DATA_DIR in available_dirs:
        default_idx = available_dirs.index(DEFAULT_DATA_DIR)
    data_dir = st.sidebar.selectbox(
        "Папка с результатами",
        options=available_dirs,
        index=default_idx,
    )
    custom_dir = st.sidebar.text_input("...или укажите свою", value="")
    if custom_dir.strip():
        data_dir = custom_dir.strip()
else:
    data_dir = st.sidebar.text_input(
        "Папка с результатами", value=DEFAULT_DATA_DIR
    )

window_choice = st.sidebar.selectbox(
    "Окно агрегации",
    options=["1min", "1h", "12h", "24h"],
    index=1,
)

if st.sidebar.button("ОБНОВИТЬ ДАННЫЕ"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(">> Ozone Sorting Center Digital Twin <<")

# ============================================================================
# Загрузка данных
# ============================================================================

data = load_all(data_dir)
ops_min = data["ops_min"]
ops_agg = data["ops_agg"]
queues = data["queues"]
utilization = data["utilization"]
summary = data["summary"]

# ============================================================================
# Заголовок
# ============================================================================

st.title("OZONE SORTING CENTER DIGITAL TWIN")
st.caption(f">> Данные из: `{data_dir}`")

if all(v is None for v in [ops_min, ops_agg, queues, utilization, summary]):
    st.error(
        f"[X] В папке `{data_dir}` не найдено ни одного файла. "
        "Выберите корректную папку в sidebar."
    )
    st.stop()

if any(v is None for v in [ops_min, ops_agg, queues, utilization, summary]):
    missing = [k for k, v in data.items() if v is None]
    st.warning(f"[!] Не удалось загрузить: {', '.join(missing)}")

# ============================================================================
# Вкладки
# ============================================================================

tab_overview, tab_throughput, tab_resources, tab_queues, tab_insights = st.tabs(
    ["ОБЗОР", "ПРОПУСКНАЯ СПОСОБНОСТЬ", "РЕСУРСЫ",
     "ОЧЕРЕДИ", "АНАЛИТИКА"]
)

# ----------------------------------------------------------------------------
# Вкладка "Обзор"
# ----------------------------------------------------------------------------
with tab_overview:
    st.header("Обзор работы центра")

    totals = (summary or {}).get("totals", {}) or {}
    balance = (summary or {}).get("balance", {}) or {}
    horizon = (summary or {}).get("horizon_sec", None)

    items_in = balance.get("items_in", totals.get("infeed_items", 0)) or 0
    items_shipped = balance.get("items_shipped",
                                totals.get("shipped_items", 0)) or 0
    items_sorted = balance.get("items_sorted",
                               totals.get("sorted_items", 0)) or 0
    items_nonsort = balance.get("items_nonsort",
                                totals.get("nonsort_items", 0)) or 0

    wip = max(0, items_in - items_shipped)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Товаров принято", fmt_num(items_in))
    col2.metric("Товаров отгружено", fmt_num(items_shipped))
    col3.metric("WIP (в системе)", fmt_num(wip))
    col4.metric("Время прогона", format_seconds(horizon))

    st.markdown("---")

    # Проверка баланса
    st.subheader("Проверка баланса потока")

    expected = items_sorted + items_nonsort
    diff = items_in - expected
    rel_err = abs(diff) / max(items_in, 1) * 100

    cbal1, cbal2 = st.columns([2, 1])
    with cbal1:
        bal_df = pd.DataFrame({
            "Показатель": ["items_in", "items_sorted + items_nonsort"],
            "Значение": [items_in, expected],
        })
        fig_bal = px.bar(
            bal_df, x="Показатель", y="Значение",
            text="Значение", color="Показатель",
            color_discrete_sequence=["#7cc0ff", "#1e5bb0"],
        )
        fig_bal.update_traces(
            textposition="outside",
            marker_line_color="#cfe4ff",
            marker_line_width=2,
        )
        fig_bal.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_bal, use_container_width=True)

    with cbal2:
        st.metric("Погрешность баланса", f"{rel_err:.2f}%",
                  delta=f"{diff:+d}")
        if rel_err < 1:
            st.success("[OK] Баланс сходится (< 1%)")
        elif rel_err < 5:
            st.warning("[!] Небольшой дисбаланс (< 5%)")
        else:
            st.error("[X] Значительный дисбаланс потока")

    st.markdown("---")

    # Круговая диаграмма распределения потока
    st.subheader("Распределение потока товаров")

    flow_data = pd.DataFrame({
        "Категория": ["Отсортированные (sorter)",
                      "Non-sort (ручная сортировка)",
                      "WIP (в системе)"],
        "Количество": [items_sorted, items_nonsort, wip],
    })
    flow_data = flow_data[flow_data["Количество"] > 0]

    if not flow_data.empty:
        fig_pie = px.pie(
            flow_data, values="Количество", names="Категория",
            color_discrete_sequence=GAME_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label+value",
            marker=dict(line=dict(color="#0a1428", width=3)),
        )
        fig_pie.update_layout(height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("[i] КОНФИГУРАЦИЯ ПРОГОНА"):
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.write("**Режим назначения:**",
                     (summary or {}).get("assignment_mode", "—"))
            st.write("**Горизонт (сек):**", horizon)
        with cfg2:
            cfg = (summary or {}).get("config", {})
            if cfg:
                st.json(cfg, expanded=False)