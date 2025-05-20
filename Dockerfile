# Dockerfile для RAG-системы на базе Hugging Face и Qdrant
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Предварительная загрузка моделей
# Это ускорит первый запуск и сделает образ автономным
RUN mkdir -p /app/models && \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='BAAI/bge-large-ru-v1.5', local_dir='/app/models/bge-large-ru-v1.5'); \
    snapshot_download(repo_id='IlyaGusev/saiga_mistral_7b_qlora', local_dir='/app/models/saiga_mistral_7b_qlora')"

# Копирование кода
COPY . .

# Переменные окружения по умолчанию
ENV EMBEDDING_MODEL_PATH=/app/models/bge-large-ru-v1.5
ENV LLM_MODEL_PATH=/app/models/saiga_mistral_7b_qlora
ENV QDRANT_URL=http://qdrant:6333
ENV COLLECTION_NAME=technical_documents
ENV BATCH_SIZE=100
ENV PORT=8000
ENV USE_GPU=true
ENV LOAD_8BIT=true

# Команда запуска
CMD ["sh", "-c", "python3 rag_system.py \
    --embedding-model ${EMBEDDING_MODEL_PATH} \
    --llm-model ${LLM_MODEL_PATH} \
    --qdrant-url ${QDRANT_URL} \
    --collection-name ${COLLECTION_NAME} \
    --batch-size ${BATCH_SIZE} \
    --port ${PORT} \
    --host 0.0.0.0 \
    ${USE_GPU:+--use-gpu} \
    ${LOAD_8BIT:+--no-8bit}"]
