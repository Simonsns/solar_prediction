# - Builder
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# - Runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local

RUN addgroup --system appuser && adduser --system --group appuser

# ephemeral file log 
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Code copy
COPY --chown=appuser:appuser ./src ./src
COPY --chown=appuser:appuser ./app ./app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
USER appuser

# main in app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]