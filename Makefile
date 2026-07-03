.PHONY: install smoke run day dashboard test docker scenarios clean help

help:
    @echo "Команды проекта Робозон:"
    @echo "  make install    — установить зависимости"
    @echo "  make smoke      — быстрый прогон 2 ч (проверка)"
    @echo "  make run        — полный прогон 24 ч"
    @echo "  make scenarios  — все what-if сценарии"
    @echo "  make dashboard  — Streamlit UI"
    @echo "  make test       — pytest"
    @echo "  make docker     — сборка+прогон в контейнере"

install:
    pip install -r requirements.txt

smoke:
    python -m src.run --hours 2 --out data/out_smoke

run day:
    python -m src.run --config config.yaml --out data/out

# ---- Ключевые сценарии для отчёта ----
scenarios: scenario-base scenario-tight scenario-manual scenario-strict scenario-peak

scenario-base:
    python -m src.run --config config.yaml --out data/out_base

scenario-tight:
    python -m src.run --set sorters.count=10 --out data/out_tight

scenario-manual:
    python -m src.run --set infeed.mode=manual --set infeed.stations=100 --out data/out_manual

scenario-strict:
    python -m src.run --set sorters.assignment_mode=strict --out data/out_strict

scenario-peak:
    python -m src.run --set flow.items_per_hour=120000 --out data/out_peak

dashboard:
    streamlit run dashboard/app.py

test:
    python -m pytest tests/ -v

docker:
    docker build -t robozon-sc . && \
    docker run --rm -v $(PWD)/data/out:/app/data/out robozon-sc

clean:
    rm -rf data/out_* __pycache__ .pytest_cache
    find . -type d -name __pycache__ -exec rm -rf {} +
