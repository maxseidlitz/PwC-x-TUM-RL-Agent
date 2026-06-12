FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten für Playwright
RUN apt-get update && apt-get install -y \
    wget curl gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright-Browser installieren
RUN playwright install chromium --with-deps

COPY . .

RUN mkdir -p /app/logs

CMD ["python", "scheduler.py"]
