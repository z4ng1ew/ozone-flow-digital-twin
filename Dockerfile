# Воспроизводимая среда для экспертов (критерий 6.2, 10 баллов)
FROM python:3.12-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Питон-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# По умолчанию — суточный прогон
CMD ["python", "-m", "src.run", "--config", "config.yaml", "--out", "data/out"]
