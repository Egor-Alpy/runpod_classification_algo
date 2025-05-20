# Dockerfile для улучшенной RAG-системы v2.0
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Установка системных зависимостей и утилит для мониторинга
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl jq htop \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip и установка основных инструментов
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Установка Python зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Загрузка Qdrant для локального запуска
RUN mkdir -p /app/qdrant_data && \
    cd /app && \
    wget -q https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    rm qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    chmod +x /app/qdrant

# Копирование скриптов запуска и основного кода
COPY start.sh .
COPY rag_system.py .
COPY data/ ./data/
RUN chmod +x start.sh

# Создание необходимых директорий
RUN mkdir -p /app/data /app/logs /app/cache

# Переменные окружения по умолчанию
ENV EMBEDDING_MODEL_PATH=intfloat/multilingual-e5-large
ENV LLM_MODEL_PATH=ai-forever/mGPT
ENV QDRANT_URL=http://localhost:6333
ENV COLLECTION_NAME=technical_documents
ENV BATCH_SIZE=100
ENV PORT=8000
ENV USE_GPU=true
ENV LOAD_8BIT=true
ENV WORKER_THREADS=4
ENV PYTHONUNBUFFERED=1

# Открываем порты для API и Qdrant
EXPOSE 8000 6333 6334

# Запускаем healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Команда запуска
CMD ["./start.sh", "--port", "8000", "--host", "0.0.0.0"]