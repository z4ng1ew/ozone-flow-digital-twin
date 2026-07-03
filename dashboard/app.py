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

















































# ----------------------------------------------------------------------------
# Вкладка "Пропускная способность"
# ----------------------------------------------------------------------------
with tab_throughput:
    st.header("Пропускная способность системы")

    if ops_min is None or ops_min.empty:
        st.warning("[!] Нет данных по операциям (operations_minutely.csv)")
    else:
        df = ops_min.copy()

        # Определяем колонку времени
        time_col = None
        for c in ["timestamp", "time", "t", "minute", "ts"]:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            st.error("[X] В operations_minutely.csv не найдена колонка времени")
        else:
            # Приводим к datetime, если возможно
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                use_datetime = True
            except Exception:
                use_datetime = False

            # Определяем колонку с количеством товаров/операций
            value_candidates = [
                "items", "items_count", "count", "throughput",
                "n_items", "processed", "value"
            ]
            value_col = None
            for c in value_candidates:
                if c in df.columns:
                    value_col = c
                    break

            if value_col is None:
                # Берём первую числовую колонку (не time_col)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != time_col]
                if numeric_cols:
                    value_col = numeric_cols[0]

            # Определяем колонку операции/этапа
            op_col = None
            for c in ["operation", "op", "stage", "process", "type"]:
                if c in df.columns:
                    op_col = c
                    break

            # --- KPI ---
            st.subheader("Ключевые показатели")

            total_ops = df[value_col].sum() if value_col else 0
            n_minutes = len(df) if not use_datetime else max(
                1, int((df[time_col].max() - df[time_col].min()).total_seconds() / 60)
            )
            avg_per_min = total_ops / max(n_minutes, 1)
            peak_per_min = df[value_col].max() if value_col else 0
            avg_per_hour = avg_per_min * 60

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Всего операций", fmt_num(int(total_ops)))
            k2.metric("Средн. в минуту", fmt_num(round(avg_per_min, 1)))
            k3.metric("Средн. в час", fmt_num(round(avg_per_hour, 0)))
            k4.metric("Пик за минуту", fmt_num(int(peak_per_min)))

            st.markdown("---")

            # --- Ресэмплинг по выбранному окну ---
            st.subheader(f"Динамика операций (окно: {window_choice})")

            plot_df = df.copy()

            if use_datetime:
                plot_df = plot_df.set_index(time_col)
                freq_map = {"1min": "1min", "1h": "1h", "12h": "12h", "24h": "24h"}
                freq = freq_map.get(window_choice, "1h")

                if op_col and op_col in plot_df.columns:
                    resampled = (
                        plot_df.groupby(op_col)[value_col]
                        .resample(freq).sum()
                        .reset_index()
                    )
                    fig_line = px.line(
                        resampled, x=time_col, y=value_col, color=op_col,
                        color_discrete_sequence=GAME_COLORS,
                        markers=True,
                    )
                else:
                    resampled = plot_df[value_col].resample(freq).sum().reset_index()
                    fig_line = px.line(
                        resampled, x=time_col, y=value_col,
                        color_discrete_sequence=["#7cc0ff"],
                        markers=True,
                    )
            else:
                # Без datetime — просто по индексу
                if op_col and op_col in plot_df.columns:
                    fig_line = px.line(
                        plot_df, x=time_col, y=value_col, color=op_col,
                        color_discrete_sequence=GAME_COLORS,
                    )
                else:
                    fig_line = px.line(
                        plot_df, x=time_col, y=value_col,
                        color_discrete_sequence=["#7cc0ff"],
                    )

            fig_line.update_traces(line=dict(width=3))
            fig_line.update_layout(
                height=420,
                xaxis_title="Время",
                yaxis_title="Операций",
                hovermode="x unified",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")

            # --- Распределение по операциям ---
            if op_col and op_col in df.columns and value_col:
                st.subheader("Распределение по типам операций")

                by_op = (
                    df.groupby(op_col)[value_col].sum()
                    .sort_values(ascending=True).reset_index()
                )

                c1, c2 = st.columns([3, 2])
                with c1:
                    fig_bar = px.bar(
                        by_op, x=value_col, y=op_col, orientation="h",
                        text=value_col,
                        color=value_col,
                        color_continuous_scale=[
                            [0, "#1e5bb0"], [0.5, "#409cff"], [1, "#7cc0ff"]
                        ],
                    )
                    fig_bar.update_traces(
                        textposition="outside",
                        marker_line_color="#cfe4ff",
                        marker_line_width=2,
                    )
                    fig_bar.update_layout(
                        height=420,
                        xaxis_title="Всего операций",
                        yaxis_title="Тип операции",
                        showlegend=False,
                        coloraxis_showscale=False,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with c2:
                    st.markdown("**ТОП операций:**")
                    top_df = by_op.sort_values(value_col, ascending=False).head(10)
                    top_df.columns = ["Операция", "Количество"]
                    top_df["Доля, %"] = (
                        top_df["Количество"] / top_df["Количество"].sum() * 100
                    ).round(1)
                    st.dataframe(
                        top_df, use_container_width=True, hide_index=True
                    )

                st.markdown("---")

            # --- Агрегированная сводка ---
            if ops_agg is not None and not ops_agg.empty:
                st.subheader("Агрегированная сводка")
                with st.expander("[i] ПОКАЗАТЬ ТАБЛИЦУ ops_agg"):
                    st.dataframe(ops_agg, use_container_width=True, hide_index=True)

            # --- Гистограмма нагрузки по минутам ---
            if value_col:
                st.subheader("Гистограмма нагрузки")

                fig_hist = px.histogram(
                    df, x=value_col, nbins=40,
                    color_discrete_sequence=["#409cff"],
                )
                fig_hist.update_traces(
                    marker_line_color="#cfe4ff",
                    marker_line_width=2,
                )
                fig_hist.update_layout(
                    height=360,
                    xaxis_title="Операций за интервал",
                    yaxis_title="Частота (кол-во интервалов)",
                    bargap=0.05,
                    **PLOTLY_LAYOUT,
                )

                # Средняя и пиковая линии
                fig_hist.add_vline(
                    x=avg_per_min, line_dash="dash", line_color="#7cc0ff",
                    line_width=3,
                    annotation_text=f"Средн: {avg_per_min:.1f}",
                    annotation_position="top",
                    annotation_font_color="#7cc0ff",
                )
                fig_hist.add_vline(
                    x=peak_per_min, line_dash="dot", line_color="#ffffff",
                    line_width=3,
                    annotation_text=f"Пик: {int(peak_per_min)}",
                    annotation_position="top",
                    annotation_font_color="#ffffff",
                )
                st.plotly_chart(fig_hist, use_container_width=True)






































































































































































































# ----------------------------------------------------------------------------
# Вкладка "Ресурсы и узкие места"
# ----------------------------------------------------------------------------
with tab_resources:
    st.header("Ресурсы и узкие места")

    if utilization is None or utilization.empty:
        st.warning("[!] Нет данных по утилизации (utilization.csv)")
    else:
        df_u = utilization.copy()

        # Автодетект колонок
        resource_col = None
        for c in ["resource", "station", "process", "stage", "name", "type"]:
            if c in df_u.columns:
                resource_col = c
                break

        util_col = None
        for c in ["utilization", "util", "busy_ratio", "load", "usage"]:
            if c in df_u.columns:
                util_col = c
                break

        # fallback — первая числовая колонка
        if util_col is None:
            numeric_cols = df_u.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                util_col = numeric_cols[0]

        if resource_col is None or util_col is None:
            st.error("[X] Не удалось определить колонки ресурса/утилизации")
            st.dataframe(df_u.head(), use_container_width=True)
        else:
            # Если утилизация в долях (0..1) — конвертируем в проценты
            max_val = df_u[util_col].max()
            if max_val is not None and max_val <= 1.5:
                df_u["_util_pct"] = df_u[util_col] * 100
            else:
                df_u["_util_pct"] = df_u[util_col]

            # Агрегируем по ресурсу (на случай, если строк несколько)
            agg_u = (
                df_u.groupby(resource_col)["_util_pct"]
                .mean().reset_index()
                .sort_values("_util_pct", ascending=False)
            )
            agg_u.columns = [resource_col, "Утилизация, %"]

            # --- KPI ---
            st.subheader("Ключевые показатели")

            n_resources = len(agg_u)
            avg_util = agg_u["Утилизация, %"].mean()
            max_util = agg_u["Утилизация, %"].max()
            n_overloaded = int((agg_u["Утилизация, %"] >= 90).sum())
            n_idle = int((agg_u["Утилизация, %"] < 30).sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ресурсов", fmt_num(n_resources))
            k2.metric("Средн. утилизация", f"{avg_util:.1f}%")
            k3.metric("Перегружено (>=90%)", fmt_num(n_overloaded),
                      delta="узкое место" if n_overloaded > 0 else "OK",
                      delta_color="inverse" if n_overloaded > 0 else "normal")
            k4.metric("Простаивает (<30%)", fmt_num(n_idle))

            st.markdown("---")

            # --- Барчарт утилизации ---
            st.subheader("Утилизация по ресурсам")

            def util_color(pct: float) -> str:
                # Синие оттенки: чем выше нагрузка, тем ярче/белее
                if pct >= 90:
                    return "#ffffff"   # критично — белый
                elif pct >= 70:
                    return "#7cc0ff"   # высоко
                elif pct >= 40:
                    return "#409cff"   # средне
                else:
                    return "#1e5bb0"   # низко

            agg_u_sorted = agg_u.sort_values("Утилизация, %", ascending=True)
            colors = [util_color(v) for v in agg_u_sorted["Утилизация, %"]]

            fig_util = px.bar(
                agg_u_sorted,
                x="Утилизация, %",
                y=resource_col,
                orientation="h",
                text=agg_u_sorted["Утилизация, %"].round(1).astype(str) + "%",
            )
            fig_util.update_traces(
                marker_color=colors,
                marker_line_color="#cfe4ff",
                marker_line_width=2,
                textposition="outside",
                textfont=dict(color="#cfe4ff", family="VT323, monospace", size=14),
            )
            fig_util.update_layout(
                height=max(360, 28 * len(agg_u_sorted) + 100),
                xaxis_title="Утилизация, %",
                yaxis_title="Ресурс",
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            # Пороговые линии
            fig_util.add_vline(
                x=70, line_dash="dot", line_color="#7cc0ff", line_width=2,
                annotation_text="70%", annotation_position="top",
                annotation_font_color="#7cc0ff",
            )
            fig_util.add_vline(
                x=90, line_dash="dash", line_color="#ffffff", line_width=2,
                annotation_text="90% КРИТ", annotation_position="top",
                annotation_font_color="#ffffff",
            )
            fig_util.update_xaxes(range=[0, max(100, max_util * 1.1)])
            st.plotly_chart(fig_util, use_container_width=True)

            st.markdown("---")

            # --- Тепловая карта / категории нагрузки ---
            c1, c2 = st.columns([3, 2])

            with c1:
                st.subheader("Категории нагрузки")

                def categorize(pct):
                    if pct >= 90: return "КРИТИЧНО (>=90%)"
                    if pct >= 70: return "ВЫСОКО (70-90%)"
                    if pct >= 40: return "СРЕДНЕ (40-70%)"
                    return "НИЗКО (<40%)"

                agg_u["Категория"] = agg_u["Утилизация, %"].apply(categorize)
                cat_counts = (
                    agg_u["Категория"].value_counts()
                    .reindex([
                        "КРИТИЧНО (>=90%)", "ВЫСОКО (70-90%)",
                        "СРЕДНЕ (40-70%)", "НИЗКО (<40%)"
                    ], fill_value=0)
                    .reset_index()
                )
                cat_counts.columns = ["Категория", "Ресурсов"]

                fig_cat = px.bar(
                    cat_counts, x="Категория", y="Ресурсов",
                    text="Ресурсов",
                    color="Категория",
                    color_discrete_map={
                        "КРИТИЧНО (>=90%)": "#ffffff",
                        "ВЫСОКО (70-90%)": "#7cc0ff",
                        "СРЕДНЕ (40-70%)": "#409cff",
                        "НИЗКО (<40%)": "#1e5bb0",
                    },
                )
                fig_cat.update_traces(
                    textposition="outside",
                    marker_line_color="#cfe4ff",
                    marker_line_width=2,
                )
                fig_cat.update_layout(
                    height=380,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Кол-во ресурсов",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            with c2:
                st.subheader("ТОП узких мест")

                bottlenecks = agg_u.sort_values(
                    "Утилизация, %", ascending=False
                ).head(10).copy()
                bottlenecks["Утилизация, %"] = bottlenecks["Утилизация, %"].round(1)
                bottlenecks = bottlenecks[[resource_col, "Утилизация, %", "Категория"]]
                st.dataframe(bottlenecks, use_container_width=True, hide_index=True)

                if n_overloaded > 0:
                    top_bn = bottlenecks.iloc[0]
                    st.error(
                        f"[!] УЗКОЕ МЕСТО: {top_bn[resource_col]} — "
                        f"{top_bn['Утилизация, %']}%"
                    )
                else:
                    st.success("[OK] Критичных узких мест не обнаружено")

            st.markdown("---")

            # --- Полная таблица ---
            with st.expander("[i] ПОЛНАЯ ТАБЛИЦА УТИЛИЗАЦИИ"):
                display_df = agg_u.copy()
                display_df["Утилизация, %"] = display_df["Утилизация, %"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)











































# ----------------------------------------------------------------------------
# Вкладка "Очереди"
# ----------------------------------------------------------------------------
with tab_queues:
    st.header("Очереди в системе")

    if queues is None or queues.empty:
        st.warning("[!] Нет данных по очередям (queues.csv)")
    else:
        df_q = queues.copy()

        # Автодетект колонок
        time_col = None
        for c in ["timestamp", "time", "t", "minute", "ts"]:
            if c in df_q.columns:
                time_col = c
                break

        queue_col = None
        for c in ["queue", "station", "resource", "name", "stage", "process"]:
            if c in df_q.columns:
                queue_col = c
                break

        length_col = None
        for c in ["length", "queue_length", "size", "count", "items", "n"]:
            if c in df_q.columns:
                length_col = c
                break

        # fallback — первая числовая, не time
        if length_col is None:
            numeric_cols = df_q.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != time_col]
            if numeric_cols:
                length_col = numeric_cols[0]

        if length_col is None:
            st.error("[X] Не удалось определить колонку длины очереди")
            st.dataframe(df_q.head(), use_container_width=True)
        else:
            # Приводим время
            use_datetime = False
            if time_col:
                try:
                    df_q[time_col] = pd.to_datetime(df_q[time_col])
                    use_datetime = True
                except Exception:
                    use_datetime = False

            # --- KPI ---
            st.subheader("Ключевые показатели")

            avg_len = df_q[length_col].mean()
            max_len = df_q[length_col].max()
            final_len = 0
            if time_col and queue_col:
                last_time = df_q[time_col].max()
                final_len = df_q[df_q[time_col] == last_time][length_col].sum()
            elif queue_col:
                final_len = df_q.groupby(queue_col)[length_col].last().sum()
            else:
                final_len = df_q[length_col].iloc[-1]

            n_queues = df_q[queue_col].nunique() if queue_col else 1

            # Проверка на growing queues
            growing_queues = []
            if queue_col and time_col:
                for q_name, grp in df_q.groupby(queue_col):
                    grp_sorted = grp.sort_values(time_col)
                    if is_monotonic_growing(grp_sorted[length_col]):
                        growing_queues.append(q_name)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Очередей", fmt_num(n_queues))
            k2.metric("Средн. длина", fmt_num(round(avg_len, 1)))
            k3.metric("Максимум", fmt_num(int(max_len)))
            k4.metric(
                "Растущих очередей", fmt_num(len(growing_queues)),
                delta="накопление" if growing_queues else "стабильно",
                delta_color="inverse" if growing_queues else "normal",
            )

            if growing_queues:
                st.error(
                    "[!] ОБНАРУЖЕН РОСТ ОЧЕРЕДЕЙ: " +
                    ", ".join(str(q) for q in growing_queues[:5]) +
                    (f" и ещё {len(growing_queues) - 5}"
                     if len(growing_queues) > 5 else "")
                )
            else:
                st.success("[OK] Система стабильна — очереди не растут монотонно")

            st.markdown("---")

            # --- Динамика очередей во времени ---
            st.subheader("Динамика длины очередей")

            if time_col:
                plot_df = df_q.copy()

                # Ресэмплинг по окну
                if use_datetime and queue_col:
                    freq_map = {"1min": "1min", "1h": "1h", "12h": "12h", "24h": "24h"}
                    freq = freq_map.get(window_choice, "1h")
                    plot_df = (
                        plot_df.set_index(time_col)
                        .groupby(queue_col)[length_col]
                        .resample(freq).mean()
                        .reset_index()
                    )

                if queue_col:
                    fig_dyn = px.line(
                        plot_df, x=time_col, y=length_col, color=queue_col,
                        color_discrete_sequence=GAME_COLORS,
                        markers=False,
                    )
                else:
                    fig_dyn = px.line(
                        plot_df, x=time_col, y=length_col,
                        color_discrete_sequence=["#7cc0ff"],
                    )

                fig_dyn.update_traces(line=dict(width=2.5))
                fig_dyn.update_layout(
                    height=450,
                    xaxis_title="Время",
                    yaxis_title="Длина очереди",
                    hovermode="x unified",
                    legend=dict(
                        bgcolor="rgba(15, 36, 71, 0.8)",
                        bordercolor="#409cff",
                        borderwidth=2,
                    ),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_dyn, use_container_width=True)

                st.markdown("---")

            # --- Сравнение очередей: средняя/пик ---
            if queue_col:
                st.subheader("Сравнение очередей")

                stats_df = (
                    df_q.groupby(queue_col)[length_col]
                    .agg(["mean", "max", "std"])
                    .round(2)
                    .reset_index()
                )
                stats_df.columns = [queue_col, "Средняя", "Пик", "СКО"]
                stats_df = stats_df.sort_values("Пик", ascending=False)

                c1, c2 = st.columns([3, 2])

                with c1:
                    top_stats = stats_df.head(15).sort_values("Пик", ascending=True)

                    fig_cmp = px.bar(
                        top_stats,
                        x=["Средняя", "Пик"],
                        y=queue_col,
                        orientation="h",
                        barmode="group",
                        color_discrete_sequence=["#409cff", "#7cc0ff"],
                    )
                    # px.bar с несколькими x требует melt — сделаем через melt
                    melted = top_stats.melt(
                        id_vars=queue_col,
                        value_vars=["Средняя", "Пик"],
                        var_name="Метрика",
                        value_name="Значение",
                    )
                    fig_cmp = px.bar(
                        melted, x="Значение", y=queue_col,
                        color="Метрика", orientation="h", barmode="group",
                        color_discrete_map={
                            "Средняя": "#409cff",
                            "Пик": "#7cc0ff",
                        },
                    )
                    fig_cmp.update_traces(
                        marker_line_color="#cfe4ff",
                        marker_line_width=1.5,
                    )
                    fig_cmp.update_layout(
                        height=max(400, 30 * len(top_stats) + 100),
                        xaxis_title="Длина очереди",
                        yaxis_title="Очередь",
                        legend=dict(
                            bgcolor="rgba(15, 36, 71, 0.8)",
                            bordercolor="#409cff",
                            borderwidth=2,
                        ),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                with c2:
                    st.markdown("**ТОП критичных очередей:**")
                    display_stats = stats_df.head(10).copy()
                    if growing_queues:
                        display_stats["Статус"] = display_stats[queue_col].apply(
                            lambda x: "РАСТЁТ" if x in growing_queues else "OK"
                        )
                    st.dataframe(
                        display_stats, use_container_width=True, hide_index=True
                    )

                st.markdown("---")

            # --- Тепловая карта (если много очередей + время) ---
            if queue_col and time_col and use_datetime:
                st.subheader("Тепловая карта загрузки очередей")

                heat_df = df_q.copy()
                freq_map = {"1min": "1min", "1h": "1h", "12h": "12h", "24h": "24h"}
                freq = freq_map.get(window_choice, "1h")

                heat_pivot = (
                    heat_df.set_index(time_col)
                    .groupby(queue_col)[length_col]
                    .resample(freq).mean()
                    .reset_index()
                    .pivot(index=queue_col, columns=time_col, values=length_col)
                    .fillna(0)
                )

                if not heat_pivot.empty and heat_pivot.shape[1] > 1:
                    fig_heat = px.imshow(
                        heat_pivot,
                        color_continuous_scale=[
                            [0.0, "#0a1428"],
                            [0.3, "#1e5bb0"],
                            [0.6, "#409cff"],
                            [0.85, "#7cc0ff"],
                            [1.0, "#ffffff"],
                        ],
                        aspect="auto",
                    )
                    fig_heat.update_layout(
                        height=max(300, 25 * len(heat_pivot) + 100),
                        xaxis_title="Время",
                        yaxis_title="Очередь",
                        coloraxis_colorbar=dict(
                            title="Длина",
                            tickfont=dict(color="#cfe4ff"),
                            titlefont=dict(color="#7cc0ff"),
                        ),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                st.markdown("---")

            # --- Гистограмма распределения длин ---
            st.subheader("Распределение длин очередей")

            fig_hist_q = px.histogram(
                df_q, x=length_col, nbins=40,
                color=queue_col if queue_col else None,
                color_discrete_sequence=GAME_COLORS,
                barmode="overlay" if queue_col else "relative",
                opacity=0.75,
            )
            fig_hist_q.update_traces(
                marker_line_color="#cfe4ff",
                marker_line_width=1.5,
            )
            fig_hist_q.update_layout(
                height=380,
                xaxis_title="Длина очереди",
                yaxis_title="Частота",
                bargap=0.05,
                legend=dict(
                    bgcolor="rgba(15, 36, 71, 0.8)",
                    bordercolor="#409cff",
                    borderwidth=2,
                ),
                **PLOTLY_LAYOUT,
            )
            fig_hist_q.add_vline(
                x=avg_len, line_dash="dash", line_color="#7cc0ff", line_width=3,
                annotation_text=f"Средн: {avg_len:.1f}",
                annotation_position="top",
                annotation_font_color="#7cc0ff",
            )
            st.plotly_chart(fig_hist_q, use_container_width=True)

            with st.expander("[i] ПОЛНАЯ ТАБЛИЦА ОЧЕРЕДЕЙ"):
                st.dataframe(df_q.head(500), use_container_width=True, hide_index=True)
                st.caption(f">> Показаны первые 500 строк из {len(df_q)}")



































































                # ----------------------------------------------------------------------------
# Вкладка "Аналитические выводы"
# ----------------------------------------------------------------------------
with tab_insights:
    st.header("Аналитические выводы")

    st.markdown(
        ">> Автоматический анализ прогона: узкие места, стабильность, "
        "рекомендации."
    )

    insights = []      # список (уровень, заголовок, описание)
    LEVEL_OK = "ok"
    LEVEL_WARN = "warn"
    LEVEL_CRIT = "crit"
    LEVEL_INFO = "info"

    # ========================================================================
    # 1. Анализ баланса потока
    # ========================================================================
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
    expected = items_sorted + items_nonsort
    rel_err = abs(items_in - expected) / max(items_in, 1) * 100

    if items_in > 0:
        if rel_err < 1:
            insights.append((
                LEVEL_OK, "Баланс потока сходится",
                f"Погрешность {rel_err:.2f}% (< 1%). Модель корректна: "
                f"items_in ({fmt_num(items_in)}) ≈ sorted + nonsort "
                f"({fmt_num(expected)})."
            ))
        elif rel_err < 5:
            insights.append((
                LEVEL_WARN, "Небольшой дисбаланс потока",
                f"Погрешность {rel_err:.2f}%. Возможны потери на этапах "
                "или незакрытые операции. Рекомендуется проверить события."
            ))
        else:
            insights.append((
                LEVEL_CRIT, "Значительный дисбаланс потока",
                f"Погрешность {rel_err:.2f}%. Разница между входом и суммой "
                f"выходов: {items_in - expected:+d}. Проверить логику модели."
            ))

    # ========================================================================
    # 2. WIP и пропускная способность
    # ========================================================================
    if items_in > 0 and items_shipped > 0:
        ship_ratio = items_shipped / items_in * 100
        wip_ratio = wip / items_in * 100

        if ship_ratio >= 95:
            insights.append((
                LEVEL_OK, "Система успевает отгружать",
                f"Отгружено {ship_ratio:.1f}% товаров, WIP всего {wip_ratio:.1f}%. "
                "Пропускная способность соответствует нагрузке."
            ))
        elif ship_ratio >= 75:
            insights.append((
                LEVEL_WARN, "Умеренное накопление WIP",
                f"Отгружено {ship_ratio:.1f}%, в системе застряло "
                f"{fmt_num(wip)} товаров ({wip_ratio:.1f}%). "
                "Рекомендуется расширить bottleneck-ресурсы."
            ))
        else:
            insights.append((
                LEVEL_CRIT, "Система не справляется с потоком",
                f"Отгружено всего {ship_ratio:.1f}%. WIP = {fmt_num(wip)} "
                f"({wip_ratio:.1f}% от принятого). "
                "Требуется увеличить мощности узких мест."
            ))

    if horizon and items_shipped:
        throughput_per_hour = items_shipped / (horizon / 3600)
        insights.append((
            LEVEL_INFO, "Средняя производительность",
            f"Система отгружает ~{fmt_num(round(throughput_per_hour, 0))} "
            f"товаров/час за горизонт {format_seconds(horizon)}."
        ))

    # ========================================================================
    # 3. Non-sort доля
    # ========================================================================
    if items_sorted + items_nonsort > 0:
        nonsort_ratio = items_nonsort / (items_sorted + items_nonsort) * 100

        if nonsort_ratio < 5:
            insights.append((
                LEVEL_OK, "Низкая доля non-sort",
                f"Non-sort всего {nonsort_ratio:.1f}%. "
                "Автоматическая сортировка работает эффективно."
            ))
        elif nonsort_ratio < 15:
            insights.append((
                LEVEL_INFO, "Умеренная доля non-sort",
                f"Non-sort составляет {nonsort_ratio:.1f}%. "
                "В пределах нормы, но есть потенциал оптимизации."
            ))
        else:
            insights.append((
                LEVEL_WARN, "Высокая доля ручной сортировки",
                f"Non-sort = {nonsort_ratio:.1f}% ({fmt_num(items_nonsort)} шт.). "
                "Возможные причины: некорректные габариты, сбои сортировщика."
            ))

    # ========================================================================
    # 4. Анализ утилизации ресурсов
    # ========================================================================
    if utilization is not None and not utilization.empty:
        df_u = utilization.copy()

        resource_col = None
        for c in ["resource", "station", "process", "stage", "name", "type"]:
            if c in df_u.columns:
                resource_col = c
                break

        util_col = None
        for c in ["utilization", "util", "busy_ratio", "load", "usage"]:
            if c in df_u.columns:
                util_col = c
                break
        if util_col is None:
            numeric_cols = df_u.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                util_col = numeric_cols[0]

        if resource_col and util_col:
            max_val = df_u[util_col].max()
            if max_val is not None and max_val <= 1.5:
                df_u["_util_pct"] = df_u[util_col] * 100
            else:
                df_u["_util_pct"] = df_u[util_col]

            agg_u = (
                df_u.groupby(resource_col)["_util_pct"]
                .mean().reset_index()
                .sort_values("_util_pct", ascending=False)
            )

            overloaded = agg_u[agg_u["_util_pct"] >= 90]
            high = agg_u[(agg_u["_util_pct"] >= 70) & (agg_u["_util_pct"] < 90)]
            idle = agg_u[agg_u["_util_pct"] < 30]

            if len(overloaded) > 0:
                top_bn = overloaded.iloc[0]
                bn_list = ", ".join(
                    f"{r[resource_col]} ({r['_util_pct']:.0f}%)"
                    for _, r in overloaded.head(3).iterrows()
                )
                insights.append((
                    LEVEL_CRIT, f"Критическая нагрузка: {len(overloaded)} ресурс(ов)",
                    f"Узкие места (>=90%): {bn_list}. "
                    f"Главное: **{top_bn[resource_col]}** — "
                    f"{top_bn['_util_pct']:.1f}%. Рекомендуется добавить мощности."
                ))
            elif len(high) > 0:
                insights.append((
                    LEVEL_WARN, f"Высокая нагрузка: {len(high)} ресурс(ов)",
                    f"Ресурсы в зоне 70-90%: работают на пределе, "
                    "нет запаса под пиковую нагрузку."
                ))
            else:
                insights.append((
                    LEVEL_OK, "Нет перегруженных ресурсов",
                    "Все ресурсы работают в пределах 90%. Запас мощности есть."
                ))

            if len(idle) > 0:
                idle_list = ", ".join(str(r) for r in idle[resource_col].head(3))
                insights.append((
                    LEVEL_INFO, f"Простаивающие ресурсы: {len(idle)}",
                    f"Ресурсы с загрузкой < 30%: {idle_list}"
                    f"{' и др.' if len(idle) > 3 else ''}. "
                    "Возможно, избыточная мощность или неверная маршрутизация."
                ))

    # ========================================================================
    # 5. Анализ очередей
    # ========================================================================
    if queues is not None and not queues.empty:
        df_q = queues.copy()

        time_col_q = None
        for c in ["timestamp", "time", "t", "minute", "ts"]:
            if c in df_q.columns:
                time_col_q = c
                break

        queue_col_q = None
        for c in ["queue", "station", "resource", "name", "stage", "process"]:
            if c in df_q.columns:
                queue_col_q = c
                break

        length_col_q = None
        for c in ["length", "queue_length", "size", "count", "items", "n"]:
            if c in df_q.columns:
                length_col_q = c
                break
        if length_col_q is None:
            numeric_cols = df_q.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != time_col_q]
            if numeric_cols:
                length_col_q = numeric_cols[0]

        if length_col_q and queue_col_q and time_col_q:
            try:
                df_q[time_col_q] = pd.to_datetime(df_q[time_col_q])
            except Exception:
                pass

            growing = []
            for q_name, grp in df_q.groupby(queue_col_q):
                grp_sorted = grp.sort_values(time_col_q)
                if is_monotonic_growing(grp_sorted[length_col_q]):
                    growing.append(q_name)

            if growing:
                growing_str = ", ".join(str(q) for q in growing[:5])
                if len(growing) > 5:
                    growing_str += f" и ещё {len(growing) - 5}"
                insights.append((
                    LEVEL_CRIT, f"Растущие очереди: {len(growing)}",
                    f"Очереди с монотонным ростом: {growing_str}. "
                    "Это признак системной нестабильности — приход превышает "
                    "обработку. Рост будет продолжаться."
                ))
            else:
                insights.append((
                    LEVEL_OK, "Очереди стабильны",
                    "Не обнаружено очередей с монотонным ростом. "
                    "Система находится в устойчивом режиме."
                ))

            # Экстремальные пики
            max_q_len = df_q[length_col_q].max()
            avg_q_len = df_q[length_col_q].mean()
            if avg_q_len > 0 and max_q_len / avg_q_len > 5:
                worst_q = (
                    df_q.groupby(queue_col_q)[length_col_q].max()
                    .idxmax()
                )
                insights.append((
                    LEVEL_WARN, "Резкие пики в очередях",
                    f"Пик ({int(max_q_len)}) в {max_q_len / avg_q_len:.1f}x выше "
                    f"средней ({avg_q_len:.1f}). Худшая очередь: "
                    f"**{worst_q}**. Проверить всплески потока."
                ))

    # ========================================================================
    # Рендер insights в игровом стиле
    # ========================================================================

    if not insights:
        st.info("[i] Нет данных для аналитических выводов")
    else:
        # Сводка по количеству
        n_crit = sum(1 for lvl, _, _ in insights if lvl == LEVEL_CRIT)
        n_warn = sum(1 for lvl, _, _ in insights if lvl == LEVEL_WARN)
        n_ok = sum(1 for lvl, _, _ in insights if lvl == LEVEL_OK)
        n_info = sum(1 for lvl, _, _ in insights if lvl == LEVEL_INFO)

        st.subheader("Сводка проверок")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Критично", fmt_num(n_crit),
                  delta="проблемы" if n_crit > 0 else "OK",
                  delta_color="inverse" if n_crit > 0 else "normal")
        s2.metric("Предупреждения", fmt_num(n_warn))
        s3.metric("Всё OK", fmt_num(n_ok))
        s4.metric("Инфо", fmt_num(n_info))

        # Общий вердикт
        if n_crit > 0:
            st.error(
                f"[X] СИСТЕМА НЕСТАБИЛЬНА — обнаружено {n_crit} критических "
                "проблем. Требуется вмешательство."
            )
        elif n_warn > 0:
            st.warning(
                f"[!] СИСТЕМА РАБОТАЕТ С ЗАМЕЧАНИЯМИ — {n_warn} предупреждений. "
                "Есть потенциал оптимизации."
            )
        else:
            st.success(
                "[OK] СИСТЕМА СТАБИЛЬНА — критических проблем не обнаружено."
            )

        st.markdown("---")

        # Стили карточек insight
        LEVEL_STYLES = {
            LEVEL_CRIT: {
                "border": "#ff5577",
                "bg": "linear-gradient(180deg, #3a0f1e 0%, #1a0a12 100%)",
                "accent": "#ff88a0",
                "label": "КРИТИЧНО",
                "shadow": "#5a1428",
            },
            LEVEL_WARN: {
                "border": "#ffcc44",
                "bg": "linear-gradient(180deg, #3a2a0f 0%, #1a140a 100%)",
                "accent": "#ffe088",
                "label": "ПРЕДУПРЕЖДЕНИЕ",
                "shadow": "#5a4014",
            },
            LEVEL_OK: {
                "border": "#44dd88",
                "bg": "linear-gradient(180deg, #0f3a24 0%, #0a1a14 100%)",
                "accent": "#88ffbb",
                "label": "OK",
                "shadow": "#145a34",
            },
            LEVEL_INFO: {
                "border": "#409cff",
                "bg": "linear-gradient(180deg, #0f2447 0%, #0a1830 100%)",
                "accent": "#7cc0ff",
                "label": "ИНФО",
                "shadow": "#1a3a6a",
            },
        }

        # Сортируем: сначала критичные, потом warn, ok, info
        order = {LEVEL_CRIT: 0, LEVEL_WARN: 1, LEVEL_OK: 2, LEVEL_INFO: 3}
        insights_sorted = sorted(insights, key=lambda x: order.get(x[0], 99))

        st.subheader(f"Детальные выводы ({len(insights_sorted)})")

        for level, title, desc in insights_sorted:
            style = LEVEL_STYLES.get(level, LEVEL_STYLES[LEVEL_INFO])

            card_html = f"""
            <div style="
                background: {style['bg']};
                border: 3px solid {style['border']};
                padding: 16px 20px;
                margin-bottom: 16px;
                box-shadow: 4px 4px 0 {style['shadow']},
                            inset 0 0 0 1px {style['accent']};
                position: relative;
            ">
                <div style="
                    font-family: 'Press Start 2P', monospace;
                    font-size: 0.65rem;
                    color: {style['accent']};
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-bottom: 8px;
                    text-shadow: 2px 2px 0 #0a1428;
                ">
                    [ {style['label']} ]
                </div>
                <div style="
                    font-family: 'Press Start 2P', monospace;
                    font-size: 0.85rem;
                    color: #ffffff;
                    margin-bottom: 10px;
                    line-height: 1.5;
                    text-shadow: 2px 2px 0 #0a1428;
                ">
                    {title}
                </div>
                <div style="
                    font-family: 'VT323', monospace;
                    font-size: 1.15rem;
                    color: #cfe4ff;
                    line-height: 1.4;
                ">
                    {desc}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")

        # ====================================================================
        # Рекомендации
        # ====================================================================
        st.subheader("Рекомендации")

        recommendations = []

        if n_crit > 0:
            for lvl, title, _ in insights_sorted:
                if lvl != LEVEL_CRIT:
                    continue
                if "нагрузка" in title.lower() or "узк" in title.lower():
                    recommendations.append(
                        ">> Увеличить число параллельных линий/сотрудников "
                        "на узких местах или ускорить обработку."
                    )
                elif "очеред" in title.lower() or "растущ" in title.lower():
                    recommendations.append(
                        ">> Снизить входной поток или расширить мощность "
                        "обработчиков, читающих из растущих очередей."
                    )
                elif "не справ" in title.lower() or "wip" in title.lower():
                    recommendations.append(
                        ">> Требуется масштабирование: пропускная способность "
                        "ниже входного потока."
                    )
                elif "баланс" in title.lower():
                    recommendations.append(
                        ">> Проверить логику модели: возможна потеря товаров "
                        "или незакрытые события."
                    )

        if n_warn > 0 and not recommendations:
            recommendations.append(
                ">> Мониторить ресурсы в зоне 70-90% — при росте нагрузки "
                "они станут узким местом."
            )

        if not recommendations and n_crit == 0 and n_warn == 0:
            recommendations.append(
                ">> Система работает оптимально. Можно рассмотреть увеличение "
                "входного потока для проверки запаса мощности."
            )

        # Уникальные
        recommendations = list(dict.fromkeys(recommendations))

        for rec in recommendations:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(180deg, #0f2447 0%, #0a1830 100%);
                    border-left: 5px solid #7cc0ff;
                    padding: 12px 18px;
                    margin-bottom: 10px;
                    font-family: 'VT323', monospace;
                    font-size: 1.15rem;
                    color: #cfe4ff;
                    box-shadow: 3px 3px 0 #1a3a6a;
                ">
                    {rec}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ====================================================================
        # Итоговая техническая справка
        # ====================================================================
        with st.expander("[i] ТЕХНИЧЕСКИЕ ДЕТАЛИ ПРОГОНА"):
            tech = {
                "Горизонт симуляции": format_seconds(horizon),
                "Товаров принято": fmt_num(items_in),
                "Товаров отгружено": fmt_num(items_shipped),
                "Отсортированных": fmt_num(items_sorted),
                "Non-sort": fmt_num(items_nonsort),
                "WIP на конец": fmt_num(wip),
                "Погрешность баланса": f"{rel_err:.3f}%",
                "Режим назначения": (summary or {}).get("assignment_mode", "—"),
            }
            tech_df = pd.DataFrame(
                [(k, v) for k, v in tech.items()],
                columns=["Параметр", "Значение"],
            )
            st.dataframe(tech_df, use_container_width=True, hide_index=True)