import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import config


class ModelManager:
    """
    Класс для управления моделями и векторной базой данных
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.llm = None
        self.embeddings = None
        self.qdrant_client = None
        self.vectorstore = None

    def load_embedding_model(self):
        """Загружает модель эмбеддингов"""
        print(f"Загрузка модели эмбеддингов {config.EMBEDDING_MODEL}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(f"Модель эмбеддингов загружена на устройство {self.device}")
        return self.embeddings

    def load_llm_model(self):
        """Загружает языковую модель"""
        print(f"Загрузка языковой модели {config.LLM_MODEL}...")

        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)

        # Настройки для квантизации модели
        model_kwargs = {
            "device_map": "auto",
        }

        if config.QUANTIZE_LLM:
            model_kwargs.update({
                "load_in_8bit": True,
                "torch_dtype": torch.float16,
            })

        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            **model_kwargs
        )

        # Создаем пайплайн для генерации текста
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.15,
        )

        # Обертка LangChain
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        print(f"Языковая модель загружена")
        return self.llm

    def init_qdrant(self):
        """Инициализирует подключение к Qdrant"""
        print(f"Подключение к Qdrant по адресу {config.QDRANT_URL}...")

        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=300  # Увеличенный таймаут для работы с большими объемами данных
        )

        # Проверяем наличие коллекции
        try:
            collection_info = self.qdrant_client.get_collection(config.QDRANT_COLLECTION)
            print(f"Коллекция {config.QDRANT_COLLECTION} найдена")
        except Exception as e:
            print(f"Создание новой коллекции {config.QDRANT_COLLECTION}...")

            # Определяем размерность эмбеддингов
            if not self.embeddings:
                self.load_embedding_model()

            # Тестовый текст для определения размерности
            test_text = "Тестовый текст для определения размерности эмбеддингов"
            embedding = self.embeddings.embed_query(test_text)
            dimension = len(embedding)

            # Создаем коллекцию с шардированием для больших объемов данных
            self.qdrant_client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config={
                    "size": dimension,
                    "distance": "Cosine"
                },
                optimizers_config={
                    "indexing_threshold": 20000,  # Порог для запуска индексации
                },
                hnsw_config={
                    "m": 16,  # Число исходящих ребер в графе
                    "ef_construct": 100,  # Размер динамического списка при построении
                    "full_scan_threshold": 10000  # Порог для полного сканирования
                },
                shard_number=10,  # 10 шардов для распределения нагрузки
                replication_factor=1  # Для продакшена рекомендуется 2-3
            )

        # Инициализируем векторное хранилище LangChain
        if not self.embeddings:
            self.load_embedding_model()

        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=config.QDRANT_COLLECTION,
            embeddings=self.embeddings
        )

        print("Подключение к Qdrant успешно установлено")
        return self.vectorstore

    def initialize_all(self):
        """Инициализирует все компоненты"""
        self.load_embedding_model()
        self.load_llm_model()
        self.init_qdrant()

        return {
            "embeddings": self.embeddings,
            "llm": self.llm,
            "vectorstore": self.vectorstore,
            "qdrant_client": self.qdrant_client
        }