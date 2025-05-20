"""
RAG-система для работы с 50 млн JSON-записей технической документации
(основана на Hugging Face и Qdrant с запуском на RunPod.io)

Данная система реализует архитектуру RAG (Retrieval Augmented Generation)
и позволяет обрабатывать большие объемы структурированных данных JSON,
динамически добавляя к запросам временные данные.

Используется FastAPI вместо Flask для лучшей производительности и валидации данных.
"""

import os
import json
import time
import uuid
import torch
import logging
import argparse
import re  # Добавлен import re для KTRU endpoint
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple, Union

# Библиотеки для работы с векторной БД
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Библиотеки для работы с эмбеддингами и LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Для веб-сервера
from fastapi import FastAPI, HTTPException, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)
logger = logging.getLogger(__name__)

# Настройки по умолчанию
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Более надежная мультиязычная модель
DEFAULT_LLM_MODEL = "ai-forever/mGPT"  # Стабильная русскоязычная модель
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 100

# Проверяем и запускаем Qdrant, если он не запущен
def ensure_qdrant_running():
    """Проверяет, запущен ли Qdrant, и запускает его при необходимости"""
    import socket
    import subprocess
    import time
    import os

    # Проверяем, открыт ли порт 6333
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 6333))
    sock.close()

    if result == 0:
        logger.info("Qdrant уже запущен на порту 6333")
        return True

    logger.info("Qdrant не запущен. Пытаемся запустить...")

    # Проверяем наличие исполняемого файла qdrant
    qdrant_path = "./qdrant"
    if os.path.isfile(qdrant_path) and os.access(qdrant_path, os.X_OK):
        try:
            # Запускаем Qdrant в фоновом режиме
            subprocess.Popen([qdrant_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Ждем запуска
            max_attempts = 10
            for i in range(max_attempts):
                time.sleep(2)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 6333))
                sock.close()
                if result == 0:
                    logger.info(f"Qdrant успешно запущен после {i+1} попыток")
                    return True

            logger.error(f"Qdrant не запустился после {max_attempts} попыток")
            return False
        except Exception as e:
            logger.error(f"Ошибка при запуске Qdrant: {e}")
            return False
    else:
        logger.error(f"Исполняемый файл Qdrant не найден по пути {qdrant_path}")
        return False

class RAGSystem:
    """
    Основной класс RAG-системы, объединяющий создание эмбеддингов,
    хранение векторов и генерацию ответов
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = "technical_documents",
        use_gpu: bool = True,
        load_in_8bit: bool = True,
        vector_size: int = 1024,  # Размерность для моделей E5
    ):
        """
        Инициализация компонентов RAG-системы

        Args:
            embedding_model_name: Имя модели для эмбеддингов
            llm_model_name: Имя модели для генерации ответов
            qdrant_url: URL Qdrant сервера
            collection_name: Имя коллекции в Qdrant
            use_gpu: Использовать ли GPU для инференса
            load_in_8bit: Загружать ли LLM в 8-битном режиме для экономии памяти
            vector_size: Размерность векторов эмбеддингов
        """
        # Убедимся, что Qdrant запущен
        ensure_qdrant_running()

        self.collection_name = collection_name
        self.vector_size = vector_size

        # Инициализация клиента Qdrant с отключенной проверкой совместимости
        logger.info(f"Подключение к Qdrant по адресу {qdrant_url}")
        self.qdrant_client = QdrantClient(url=qdrant_url, check_compatibility=False)

        # Создание коллекции, если её еще нет
        self._ensure_collection_exists()

        # Инициализация модели эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")

        # Инициализируем SentenceTransformer с явным указанием устройства
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

        # Инициализация LLM для генерации ответов
        if llm_model_name:
            logger.info(f"Загрузка языковой модели: {llm_model_name}")
            self._init_llm(llm_model_name, use_gpu, load_in_8bit)
        else:
            logger.info("Языковая модель не указана, генерация ответов недоступна")
            self.llm = None
            self.tokenizer = None
            self.generation_pipeline = None

    def _ensure_collection_exists(self):
        """
        Создает коллекцию в Qdrant, если она еще не существует
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Коллекция {self.collection_name} уже существует")
        except Exception as e:
            logger.info(f"Создание коллекции {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                # Оптимизация для быстрого поиска по метаданным
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000  # Автоиндексация после 20K точек
                )
            )

    def _init_llm(self, model_name, use_gpu, load_in_8bit):
        """
        Инициализирует языковую модель для генерации ответов
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Опции загрузки модели
            model_kwargs = {}
            if use_gpu and torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16
                if load_in_8bit:
                    model_kwargs["load_in_8bit"] = True

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

            # Создание пайплайна для генерации текста
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации языковой модели: {e}")
            # Продолжаем работу без генерации ответов
            self.llm = None
            self.tokenizer = None
            self.generation_pipeline = None
            logger.warning("Система будет работать только в режиме поиска без генерации ответов")

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Добавляет документы в векторную базу данных

        Args:
            documents: Список словарей JSON для добавления
            batch_size: Размер пакета для вставки
        """
        total_docs = len(documents)
        logger.info(f"Начало обработки {total_docs} документов")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            batch_points = []

            for doc in batch:
                # Создаем текстовое представление документа для эмбеддинга
                text = self._prepare_text_from_document(doc)

                # Генерируем эмбеддинг с помощью SentenceTransformer
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)

                # Создаем уникальный ID, если его нет
                doc_id = str(doc.get("_id", {}).get("$oid", uuid.uuid4()))

                # Добавляем точку в пакет
                batch_points.append(
                    PointStruct(
                        id=doc_id,
                        vector=embedding.tolist(),
                        payload=self._prepare_payload(doc)
                    )
                )

            # Вставляем пакет документов
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )

            logger.info(f"Обработано {min(i+batch_size, total_docs)}/{total_docs} документов")

    def _prepare_text_from_document(self, doc: Dict[str, Any]) -> str:
        """
        Преобразует документ JSON в текстовое представление для эмбеддинга

        Args:
            doc: Документ JSON

        Returns:
            str: Текстовое представление документа
        """
        text = f"Код КТРУ: {doc.get('ktru_code', '')}\n"
        text += f"Название: {doc.get('title', '')}\n"

        # Добавляем описание, если оно есть
        description = doc.get('description', '')
        if description:
            text += f"Описание: {description}\n"

        # Добавляем атрибуты
        attributes = doc.get('attributes', [])
        if attributes:
            text += "Характеристики:\n"
            for attr in attributes:
                name = attr.get('attr_name', '')
                values = [v.get('value', '') for v in attr.get('attr_values', [])]
                units = [v.get('value_unit', '') for v in attr.get('attr_values', []) if v.get('value_unit')]

                # Форматируем значения атрибутов
                values_str = ", ".join(values)
                if any(units):
                    values_str += f" ({', '.join(filter(None, units))})"

                text += f"- {name}: {values_str}\n"

        return text

    def _prepare_payload(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает метаданные для сохранения в Qdrant

        Args:
            doc: Документ JSON

        Returns:
            Dict[str, Any]: Метаданные для сохранения
        """
        # Извлекаем основные поля
        payload = {
            "ktru_code": doc.get("ktru_code", ""),
            "title": doc.get("title", ""),
            "description": doc.get("description", ""),
            "unit": doc.get("unit", ""),
            "version": doc.get("version", ""),
            "updated_at": doc.get("updated_at", ""),
            "source_link": doc.get("source_link", "")
        }

        # Добавляем атрибуты в виде плоской структуры для эффективного поиска
        attributes = {}
        for attr in doc.get("attributes", []):
            attr_name = attr.get("attr_name", "")
            attr_values = [v.get("value", "") for v in attr.get("attr_values", [])]
            attributes[attr_name] = attr_values

        payload["attributes"] = attributes

        # Добавляем ключевые слова, если есть
        keywords = doc.get("keywords", [])
        if keywords:
            payload["keywords"] = keywords

        return payload

    def search(
        self,
        query: str,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу

        Args:
            query: Текстовый запрос
            filter_params: Параметры фильтрации (опционально)
            limit: Максимальное количество результатов

        Returns:
            List[Dict[str, Any]]: Список найденных документов с их метаданными
        """
        # Создаем эмбеддинг запроса с SentenceTransformer
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)

        # Настраиваем фильтр для поиска, если он передан
        search_filter = None
        if filter_params:
            search_filter = models.Filter(
                must=self._build_filter_conditions(filter_params)
            )

        # Выполняем поиск
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            filter=search_filter
        )

        # Преобразуем результаты в список документов
        documents = []
        for result in search_result:
            doc = {
                "id": result.id,
                "score": result.score,
                **result.payload
            }
            documents.append(doc)

        return documents

    def _build_filter_conditions(self, filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Строит условия фильтрации для Qdrant на основе параметров фильтра

        Args:
            filter_params: Параметры фильтрации

        Returns:
            List[Dict[str, Any]]: Список условий фильтрации
        """
        conditions = []

        for key, value in filter_params.items():
            # Проверяем, является ли значение словарем с операторами ($eq, $gt, etc)
            if isinstance(value, dict) and all(k.startswith('$') for k in value.keys()):
                for op, op_value in value.items():
                    if op == '$eq':
                        conditions.append(models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=op_value)
                        ))
                    elif op == '$gt':
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gt=op_value)
                        ))
                    elif op == '$gte':
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gte=op_value)
                        ))
                    elif op == '$lt':
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lt=op_value)
                        ))
                    elif op == '$lte':
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lte=op_value)
                        ))
            # Для атрибутов мы используем поиск по ключам и значениям
            elif key == 'attributes' and isinstance(value, dict):
                for attr_name, attr_value in value.items():
                    conditions.append(models.FieldCondition(
                        key=f"attributes.{attr_name}",
                        match=models.MatchValue(value=attr_value)
                    ))
            # Для простых значений используем точное соответствие
            else:
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                ))

        return conditions

    def generate_response(
        self,
        query: str,
        temporal_data: Optional[Dict[str, Any]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        num_documents: int = 5
    ) -> Dict[str, Any]:
        """
        Генерирует ответ на запрос, используя RAG

        Args:
            query: Запрос пользователя
            temporal_data: Временные данные для добавления к контексту (опционально)
            filter_params: Параметры фильтрации для поиска документов (опционально)
            num_documents: Количество документов для извлечения

        Returns:
            Dict[str, Any]: Ответ с контекстом и сгенерированным текстом
        """
        if not self.llm or not self.generation_pipeline:
            raise ValueError("Языковая модель не инициализирована")

        # Поиск релевантных документов
        documents = self.search(query, filter_params, num_documents)

        # Форматирование контекста из документов
        context = self._format_context_from_documents(documents)

        # Добавление временных данных, если они есть
        if temporal_data:
            temporal_context = self._format_temporal_data(temporal_data)
            context += f"\nВременные данные:\n{temporal_context}"

        # Создание промпта для модели
        prompt = self._create_prompt(query, context)

        # Генерация ответа
        try:
            generated_text = self.generation_pipeline(prompt)[0]['generated_text']
            # Извлечение только сгенерированного ответа, без промпта
            answer = self._extract_answer(generated_text, prompt)
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            answer = f"Не удалось сгенерировать ответ: {str(e)}"

        return {
            "query": query,
            "context": context,
            "answer": answer,
            "retrieved_documents": documents
        }

    def _format_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Форматирует контекст из найденных документов

        Args:
            documents: Список документов

        Returns:
            str: Отформатированный контекст
        """
        context = ""

        for i, doc in enumerate(documents):
            context += f"Документ {i+1}:\n"
            context += f"Код КТРУ: {doc.get('ktru_code', '')}\n"
            context += f"Название: {doc.get('title', '')}\n"

            description = doc.get('description', '')
            if description:
                context += f"Описание: {description}\n"

            unit = doc.get('unit', '')
            if unit:
                context += f"Единица измерения: {unit}\n"

            # Добавляем атрибуты
            attributes = doc.get('attributes', {})
            if attributes:
                context += "Характеристики:\n"
                for attr_name, attr_values in attributes.items():
                    values_str = ", ".join(attr_values) if isinstance(attr_values, list) else str(attr_values)
                    context += f"- {attr_name}: {values_str}\n"

            context += "\n"

        return context

    def _format_temporal_data(self, temporal_data: Dict[str, Any]) -> str:
        """
        Форматирует временные данные

        Args:
            temporal_data: Словарь с временными данными

        Returns:
            str: Отформатированные временные данные
        """
        formatted_data = ""
        for key, value in temporal_data.items():
            formatted_data += f"{key}: {value}\n"
        return formatted_data

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для модели

        Args:
            query: Запрос пользователя
            context: Контекст с информацией

        Returns:
            str: Готовый промпт
        """
        # Универсальный формат промпта, который подойдет для разных моделей
        prompt = f"""Ты - ассистент по технической документации и каталогам товаров. 
Используй предоставленную информацию, чтобы ответить на вопрос.
Если ты не можешь найти ответ в контексте, так и скажи.

Контекст:
{context}

Вопрос: {query}

Ответ:
"""
        return prompt

    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """
        Извлекает ответ модели из сгенерированного текста

        Args:
            generated_text: Полный сгенерированный текст
            prompt: Исходный промпт

        Returns:
            str: Извлеченный ответ
        """
        # Удаляем промпт из начала сгенерированного текста
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            # Ищем ответ после "Ответ:" или возвращаем весь текст
            parts = generated_text.split("Ответ:")
            if len(parts) > 1:
                answer = parts[1].strip()
            else:
                answer = generated_text

        return answer

# Функция для загрузки данных из JSON-файлов
def load_json_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Загружает данные из нескольких JSON-файлов

    Args:
        file_paths: Список путей к JSON-файлам

    Returns:
        List[Dict[str, Any]]: Список JSON-документов
    """
    all_documents = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Проверяем, является ли JSON списком или одним объектом
                if isinstance(data, list):
                    all_documents.extend(data)
                else:
                    all_documents.append(data)

                logger.info(f"Загружено документов из {file_path}: {len(data) if isinstance(data, list) else 1}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path}: {e}")

    return all_documents

# Определение Pydantic моделей для валидации данных
class SearchRequest(BaseModel):
    query: str
    filter_params: Optional[Dict[str, Any]] = None
    limit: int = 5

class TemporalData(BaseModel):
    class Config:
        extra = "allow"  # Разрешаем дополнительные поля

class GenerateRequest(BaseModel):
    query: str
    temporal_data: Optional[TemporalData] = None
    filter_params: Optional[Dict[str, Any]] = None
    num_documents: int = 5

class DocumentBase(BaseModel):
    class Config:
        extra = "allow"  # Разрешаем произвольную структуру JSON

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentBase]
    batch_size: int = DEFAULT_BATCH_SIZE

# Создание FastAPI-приложения для API
def create_app(rag_system: RAGSystem):
    """
    Создает FastAPI-приложение для обслуживания API

    Args:
        rag_system: Экземпляр RAG-системы

    Returns:
        FastAPI: Экземпляр FastAPI-приложения
    """
    app = FastAPI(
        title="RAG API для технической документации",
        description="API для поиска и генерации ответов на основе технической документации",
        version="1.0.0"
    )

    # Добавляем CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/search", summary="Поиск документов")
    async def search_endpoint(request: SearchRequest = Body(...)):
        try:
            results = rag_system.search(
                query=request.query,
                filter_params=request.filter_params,
                limit=request.limit
            )
            return {"results": results}
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/generate", summary="Генерация ответа на основе документов")
    async def generate_endpoint(request: GenerateRequest = Body(...)):
        try:
            if not rag_system.generation_pipeline:
                raise HTTPException(
                    status_code=400,
                    detail="Языковая модель не инициализирована. Возможен только поиск, но не генерация ответов."
                )

            response = rag_system.generate_response(
                query=request.query,
                temporal_data=request.temporal_data.dict() if request.temporal_data else None,
                filter_params=request.filter_params,
                num_documents=request.num_documents
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/add_documents", summary="Добавление документов в систему")
    async def add_documents_endpoint(request: AddDocumentsRequest = Body(...)):
        try:
            # Преобразуем Pydantic модели в словари для обработки
            documents = [doc.dict() for doc in request.documents]
            rag_system.add_documents(documents, request.batch_size)
            return {
                "status": "success",
                "message": f'Добавлено {len(documents)} документов'
            }
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Добавляем эндпоинт для проверки состояния системы
    @app.get("/health", summary="Проверка работоспособности системы")
    async def health_check():
        return {"status": "ok", "version": "1.0.0"}

    # Добавляем специальный эндпоинт для определения кода КТРУ
    @app.post("/api/ktru_code", summary="Определение кода КТРУ по описанию товара")
    async def ktru_code_endpoint(document: DocumentBase = Body(...)):
        try:
            if not rag_system.generation_pipeline:
                raise HTTPException(
                    status_code=400,
                    detail="Языковая модель не инициализирована. Невозможно определить код КТРУ."
                )

            # Формируем промпт специально для задачи определения кода КТРУ
            prompt = f"""Я предоставлю тебе JSON-файл с описанием товара. Твоя задача - определить единственный точный код КТРУ для этого товара. Если ты не можешь определить код с высокой уверенностью (более 95%), ответь только "код не найден".

## Правила определения:
1. Анализируй все поля JSON, особое внимание обрати на:
   - title (полное наименование товара)
   - description (описание товара)
   - category и parent_category (категории товара)
   - attributes (ключевые характеристики)
   - brand (производитель)
2. Для корректного определения кода КТРУ обязательно учитывай:
   - Точное соответствие типа товара
   - Размеры и технические характеристики
   - Специфические особенности товара, указанные в описании
3. Код КТРУ должен иметь формат XX.XX.XX.XXX-XXXXXXXX, где первые цифры соответствуют ОКПД2, а после дефиса - уникальный идентификатор в КТРУ.

## Формат ответа:
- Если определен один точный код с уверенностью >95%, выведи только этот код КТРУ, без пояснений
- Если невозможно определить точный код, выведи только фразу "код не найден"

JSON товара:
{json.dumps(document.dict(), ensure_ascii=False)}

Найди наиболее подходящий код КТРУ для этого товара:
"""
            # Поиск документов по описанию товара
            search_query = f"{document.dict().get('title', '')} {document.dict().get('description', '')}"
            documents = rag_system.search(
                query=search_query,
                limit=10  # Увеличиваем количество документов для повышения точности
            )

            # Форматируем контекст из найденных документов
            context = rag_system._format_context_from_documents(documents)

            # Генерируем ответ с специальным промптом напрямую через LLM
            generated_text = rag_system.generation_pipeline(prompt + "\nРеференсные данные из каталога КТРУ:\n" + context)[0]['generated_text']

            # Извлекаем только ответ модели
            answer = rag_system._extract_answer(generated_text, prompt)

            # Проверяем формат ответа
            if answer.strip() == "код не найден":
                return {"result": "код не найден", "confidence": "< 95%"}
            elif re.match(r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}', answer.strip()):
                return {"result": answer.strip(), "confidence": "> 95%"}
            else:
                return {"result": "код не найден", "confidence": "< 95%"}

        except Exception as e:
            logger.error(f"Ошибка при определении кода КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app

# Функция для проверки зависимостей
def check_dependencies():
    """Проверяет наличие необходимых зависимостей и при необходимости устанавливает их"""
    required_packages = {
        "sentence-transformers": "2.2.2",
        "qdrant-client": "1.5.0",
        "fastapi": "0.100.0",
        "uvicorn": "0.23.0",
        "torch": "2.1.0",
        "transformers": "4.37.2",
    }

    try:
        import pkg_resources
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing = []

        for package, min_version in required_packages.items():
            if package not in installed:
                missing.append(f"{package}>={min_version}")

        if missing:
            logger.warning(f"Отсутствуют необходимые пакеты: {', '.join(missing)}")
            logger.info("Установка отсутствующих пакетов...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            logger.info("Пакеты успешно установлены")

    except Exception as e:
        logger.error(f"Ошибка при проверке зависимостей: {e}")
        logger.warning("Продолжение работы с имеющимися пакетами")

# Основная функция для запуска системы
def main():
    # Проверяем зависимости
    check_dependencies()

    # Запускаем Qdrant, если он не запущен
    ensure_qdrant_running()

    parser = argparse.ArgumentParser(description='RAG-система для работы с технической документацией')
    parser.add_argument('--embedding-model', type=str, default=DEFAULT_EMBEDDING_MODEL,
                       help=f'Модель для эмбеддингов (по умолчанию: {DEFAULT_EMBEDDING_MODEL})')
    parser.add_argument('--llm-model', type=str, default=DEFAULT_LLM_MODEL,
                       help=f'Модель для генерации ответов (по умолчанию: {DEFAULT_LLM_MODEL})')
    parser.add_argument('--qdrant-url', type=str, default=DEFAULT_QDRANT_URL,
                       help=f'URL Qdrant сервера (по умолчанию: {DEFAULT_QDRANT_URL})')
    parser.add_argument('--collection-name', type=str, default='technical_documents',
                       help='Имя коллекции в Qdrant (по умолчанию: technical_documents)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Использовать GPU для инференса')
    parser.add_argument('--no-8bit', action='store_true',
                       help='Не использовать 8-битную квантизацию для LLM')
    parser.add_argument('--import-data', type=str, nargs='+',
                       help='Пути к JSON-файлам для импорта (опционально)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Размер пакета для вставки (по умолчанию: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--port', type=int, default=8000,
                       help='Порт для запуска API (по умолчанию: 8000)')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='Хост для запуска API (по умолчанию: 0.0.0.0)')

    args = parser.parse_args()

    # Инициализация RAG-системы
    try:
        rag_system = RAGSystem(
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection_name,
            use_gpu=args.use_gpu,
            load_in_8bit=not args.no_8bit
        )

        # Импорт данных из параметров командной строки
        data_files = []

        # Если указаны пути к файлам в аргументах
        if args.import_data:
            data_files.extend(args.import_data)

        # Автоматический поиск JSON-файлов в директории data
        if os.path.exists('data'):
            logger.info("Сканирование директории data на наличие JSON-файлов...")
            import glob
            json_files = glob.glob('data/**/*.json', recursive=True)
            if json_files:
                logger.info(f"Найдено {len(json_files)} JSON-файлов в директории data")
                data_files.extend(json_files)

        # Загрузка и импорт данных
        if data_files:
            logger.info(f"Загрузка данных из {len(data_files)} файлов")
            documents = load_json_data(data_files)
            if documents:
                logger.info(f"Импорт {len(documents)} документов в Qdrant")
                rag_system.add_documents(documents, args.batch_size)

        # Создание и запуск FastAPI-приложения
        app = create_app(rag_system)

        # Запускаем с помощью Uvicorn
        logger.info(f"Запуск сервера на {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    except Exception as e:
        logger.error(f"Ошибка при запуске системы: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
