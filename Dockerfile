FROM python:3.11-slim
# libgomp: OpenMP runtime needed by LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY params.yaml .
COPY src/ src/
# Baked artifact: image tag == model version, no runtime dependency on MLflow.
COPY models/model.joblib models/model.joblib
RUN useradd -m app && chown -R app /app
USER app
ENV PORT=8080
CMD exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
