#!/bin/bash

# Скрипт для настройки RAG-системы на RunPod.io

# Обновление пакетов
apt-get update
apt-get install -y git docker.io docker-compose

# Запуск Docker
systemctl start docker
systemctl enable docker

# Клонирование репозитория с кодом
git clone https://github.com/yourusername/rag-system-50m.git
cd rag-system-50m

# Создание структуры папок
mkdir -p data/json
mkdir -p models

# Настройка переменных окружения
cat > .env << EOF
EMBEDDING_MODEL=BAAI/bge-large-ru-v1.5
LLM_MODEL=IlyaGusev/saiga_mistral_7b_qlora
QDRANT_COLLECTION=medical_products
USE_GPU=true
QUANTIZE_LLM=true
DEBUG=false
EOF

# Запуск системы через Docker Compose
docker-compose up -d

echo "RAG-система успешно запущена!"
echo "API доступен по адресу: http://localhost:8000"
echo "Документация API: http://localhost:8000/docs"