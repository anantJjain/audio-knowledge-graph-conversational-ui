FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "app.server:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "4"]
