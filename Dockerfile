FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Копирование списка зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Переменные окружения по умолчанию
ENV EMBEDDING_MODEL=BAAI/bge-large-ru-v1.5
ENV LLM_MODEL=IlyaGusev/saiga_mistral_7b_qlora
ENV QDRANT_URL=http://qdrant:6333
ENV QDRANT_COLLECTION=medical_products
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV DEBUG=False
ENV USE_GPU=True
ENV QUANTIZE_LLM=True

# Предварительная загрузка моделей
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='BAAI/bge-large-ru-v1.5'); \
    snapshot_download(repo_id='IlyaGusev/saiga_mistral_7b_qlora')"

# Запуск приложения
CMD ["python3", "api.py"]