# Dockerfile для RAG-системы на базе Hugging Face и Qdrant
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN pip3 install --no-cache-dir --upgrade pip

# Установка Python зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Загрузка Qdrant для локального запуска
RUN mkdir -p /app/qdrant_data && \
    cd /app && \
    wget -q https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    rm qdrant-x86_64-unknown-linux-gnu.tar.gz

# Копирование скриптов запуска и основного кода
COPY start.sh .
COPY rag_system.py .
RUN chmod +x start.sh

# Создание директории для данных
RUN mkdir -p /app/data

# Переменные окружения по умолчанию
ENV EMBEDDING_MODEL_PATH=intfloat/multilingual-e5-large
ENV LLM_MODEL_PATH=ai-forever/mGPT
ENV QDRANT_URL=http://localhost:6333
ENV COLLECTION_NAME=technical_documents
ENV BATCH_SIZE=100
ENV PORT=8000
ENV USE_GPU=true
ENV LOAD_8BIT=true

# Открываем порты для API и Qdrant
EXPOSE 8000 6333 6334

# Команда запуска
CMD ["./start.sh", "--embedding-model", "${EMBEDDING_MODEL_PATH}", "--llm-model", "${LLM_MODEL_PATH}", "--qdrant-url", "${QDRANT_URL}", "--collection-name", "${COLLECTION_NAME}", "--batch-size", "${BATCH_SIZE}", "--port", "${PORT}", "--host", "0.0.0.0", "--use-gpu"]