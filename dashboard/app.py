"""
Ozone Sorting Center Digital Twin — Ящик Шрёдингера
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
    page_title="Ozone Sorting Center Digital Twin — Ящик Шрёдингера",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = "data/out"

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

st.sidebar.title("⚙️ Настройки")

# Автоопределение доступных папок data/out*
def find_data_dirs() -> list[str]:
    candidates = []
    data_root = Path("data")
    if data_root.exists():
        for p in sorted(data_root.iterdir()):
            if p.is_dir() and p.name.startswith("out"):
                # непустая?
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

if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("📦 Ozone Sorting Center Digital Twin")

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

st.title("📦 Ozone Sorting Center Digital Twin — Ящик Шрёдингера")
st.caption(f"Данные из: `{data_dir}`")

if all(v is None for v in [ops_min, ops_agg, queues, utilization, summary]):
    st.error(
        f"❌ В папке `{data_dir}` не найдено ни одного файла. "
        "Выберите корректную папку в sidebar."
    )
    st.stop()

if any(v is None for v in [ops_min, ops_agg, queues, utilization, summary]):
    missing = [k for k, v in data.items() if v is None]
    st.warning(f"⚠️ Не удалось загрузить: {', '.join(missing)}")

# ============================================================================
# Вкладки
# ============================================================================

tab_overview, tab_throughput, tab_resources, tab_queues, tab_insights = st.tabs(
    ["🎯 Обзор", "📈 Пропускная способность", "🏭 Ресурсы и узкие места",
     "🚦 Очереди", "💡 Аналитические выводы"]
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
    col1.metric("📥 Товаров принято", fmt_num(items_in))
    col2.metric("🚚 Товаров отгружено", fmt_num(items_shipped))
    col3.metric("⏳ WIP (в системе)", fmt_num(wip))
    col4.metric("⏱ Время прогона", format_seconds(horizon))

    st.markdown("---")

    # Проверка баланса
    st.subheader("⚖️ Проверка баланса потока")

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
            color_discrete_sequence=["#2E86AB", "#A23B72"],
        )
        fig_bal.update_traces(textposition="outside")
        fig_bal.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bal, use_container_width=True)

    with cbal2:
        st.metric("Погрешность баланса", f"{rel_err:.2f}%",
                  delta=f"{diff:+d}")
        if rel_err < 1:
            st.success("✅ Баланс сходится (< 1%)")
        elif rel_err < 5:
            st.warning("⚠️ Небольшой дисбаланс (< 5%)")
        else:
            st.error("❌ Значительный дисбаланс потока")

    st.markdown("---")

    # Круговая диаграмма распределения потока
    st.subheader("🥧 Распределение потока товаров")

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
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside",
                              textinfo="percent+label+value")
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("ℹ️ Конфигурация прогона"):
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.write("**Режим назначения:**",
                     (summary or {}).get("assignment_mode", "—"))
            st.write("**Горизонт (сек):**", horizon)
        with cfg2:
            cfg = (summary or {}).get("config", {})
            if cfg:
                st.json(cfg, expanded=False)

# ----------------