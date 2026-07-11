FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/model.pkl
EXPOSE 5000

WORKDIR /app/src
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
