import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Общие настройки
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_products")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Настройки моделей
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-ru-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "IlyaGusev/saiga_mistral_7b_qlora")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
QUANTIZE_LLM = os.getenv("QUANTIZE_LLM", "True").lower() == "true"

# API настройки
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Настройки RAG
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))