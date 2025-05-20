"""
RAG-система для работы с миллионами JSON-записей технической документации
и определения кодов КТРУ для товаров в промышленных масштабах

Данная система реализует архитектуру RAG (Retrieval Augmented Generation)
и позволяет обрабатывать большие объемы структурированных данных JSON.
"""

import os
import json
import time
import uuid
import torch
import logging
import argparse
import re
import subprocess
import sys
import asyncio
import glob
import threading
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from queue import Queue

# Библиотеки для работы с векторной БД
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Библиотеки для работы с эмбеддингами и LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Для веб-сервера
from fastapi import FastAPI, HTTPException, Body, Query, Depends, BackgroundTasks, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Set
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
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_LLM_MODEL = "ai-forever/mGPT"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 100
DEFAULT_WORKER_THREADS = 4

# Глобальные переменные для мониторинга
class ServiceStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.documents_processed = 0
        self.ktru_requests = 0
        self.successful_ktru = 0
        self.failed_ktru = 0
        self.batch_processes = 0
        self.system_load = {
            "cpu": 0.0,
            "memory": 0.0,
            "gpu": 0.0 if torch.cuda.is_available() else None
        }
        # Очередь и результаты для пакетных задач
        self.batch_jobs = {}
        self.batch_results = {}
        self.task_queue = Queue()
        self.task_results = {}
        self.lock = threading.Lock()

    def update_system_load(self):
        """Обновляет информацию о загрузке системы"""
        try:
            import psutil
            self.system_load["cpu"] = psutil.cpu_percent()
            self.system_load["memory"] = psutil.virtual_memory().percent

            if torch.cuda.is_available():
                # Получаем информацию о загрузке GPU
                try:
                    allocated = torch.cuda.memory_allocated(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    self.system_load["gpu"] = allocated / total * 100
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о GPU: {e}")
                    self.system_load["gpu"] = 0.0
        except ImportError:
            logger.warning("Модуль psutil не установлен, информация о системе недоступна")
        except Exception as e:
            logger.error(f"Ошибка при обновлении информации о системе: {e}")

# Создаем глобальный объект статистики
service_stats = ServiceStats()

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

class BackgroundTasks:
    """Класс для управления фоновыми задачами"""

    def __init__(self, num_workers=DEFAULT_WORKER_THREADS):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.tasks = {}
        self.lock = threading.Lock()

    def submit_task(self, task_id, func, *args, **kwargs):
        """Добавляет задачу в пул на выполнение"""
        future = self.executor.submit(func, *args, **kwargs)
        with self.lock:
            self.tasks[task_id] = {
                "future": future,
                "start_time": time.time(),
                "status": "running"
            }
        return task_id

    def get_task_status(self, task_id):
        """Получает статус задачи"""
        with self.lock:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]
            future = task["future"]

            if future.done():
                if future.exception():
                    status = "failed"
                    result = str(future.exception())
                else:
                    status = "completed"
                    result = future.result()

                # Обновляем статус задачи
                if task["status"] != status:
                    task["status"] = status
                    task["end_time"] = time.time()

                return {
                    "task_id": task_id,
                    "status": status,
                    "result": result if status == "completed" else None,
                    "error": result if status == "failed" else None,
                    "duration": task["end_time"] - task["start_time"]
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "running",
                    "duration": time.time() - task["start_time"]
                }

    def cleanup_tasks(self, max_age=3600):
        """Очищает завершенные задачи старше max_age секунд"""
        current_time = time.time()
        with self.lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task["status"] in ["completed", "failed"]:
                    if "end_time" in task and current_time - task["end_time"] > max_age:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]

# Создаем экземпляр для управления фоновыми задачами
background_tasks = BackgroundTasks()

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
        """
        # Убедимся, что Qdrant запущен
        ensure_qdrant_running()

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.qdrant_url = qdrant_url
        self.use_gpu = use_gpu

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

        # Лок для обеспечения потокобезопасности
        self._lock = threading.RLock()

        # Кэширование эмбеддингов для оптимизации
        self.embedding_cache = {}
        self.embedding_cache_max_size = 10000  # Максимальный размер кэша

        # Счетчики для статистики
        self.stats = {
            "documents_total": 0,
            "documents_with_ktru": 0,
            "documents_without_ktru": 0,
            "search_requests": 0,
            "search_results": 0,
            "generation_requests": 0,
            "generation_successes": 0,
            "generation_failures": 0
        }

        # Обновляем начальную статистику
        self._update_stats_from_db()

    def _update_stats_from_db(self):
        """Обновляет статистику на основе данных из БД"""
        try:
            # Получаем общее количество документов
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.stats["documents_total"] = collection_info.vectors_count

            # Получаем количество документов с КТРУ
            search_result = self.qdrant_client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="ktru_code",
                            match=MatchValue(value={"$exists": True})
                        )
                    ]
                )
            )
            self.stats["documents_with_ktru"] = search_result.count

            # Вычисляем количество документов без КТРУ
            self.stats["documents_without_ktru"] = self.stats["documents_total"] - self.stats["documents_with_ktru"]

        except Exception as e:
            logger.error(f"Ошибка при обновлении статистики из БД: {e}")

    def get_stats(self):
        """Возвращает статистику системы"""
        # Обновляем текущую статистику
        self._update_stats_from_db()

        return {
            "qdrant": {
                "url": self.qdrant_url,
                "collection": self.collection_name,
                "documents_total": self.stats["documents_total"],
                "documents_with_ktru": self.stats["documents_with_ktru"],
                "documents_without_ktru": self.stats["documents_without_ktru"],
            },
            "embedding_model": str(self.embedding_model),
            "llm_model": str(self.llm) if self.llm else None,
            "gpu_enabled": self.use_gpu and torch.cuda.is_available(),
            "operations": {
                "search_requests": self.stats["search_requests"],
                "search_results": self.stats["search_results"],
                "generation_requests": self.stats["generation_requests"],
                "generation_successes": self.stats["generation_successes"],
                "generation_failures": self.stats["generation_failures"]
            }
        }

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

            # Обновляем статистику
            with self._lock:
                self.stats["documents_total"] += len(batch)

                # Подсчитываем документы с КТРУ и без
                docs_with_ktru = sum(1 for doc in batch if doc.get("ktru_code"))
                self.stats["documents_with_ktru"] += docs_with_ktru
                self.stats["documents_without_ktru"] += (len(batch) - docs_with_ktru)

                # Обновляем глобальную статистику
                service_stats.documents_processed += len(batch)

        return {"added": total_docs}

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
        # Обновляем статистику
        with self._lock:
            self.stats["search_requests"] += 1

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

        # Обновляем статистику
        with self._lock:
            self.stats["search_results"] += len(documents)

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
                    elif op == '$exists':
                        if op_value:
                            conditions.append(models.FieldCondition(
                                key=key,
                                match=models.MatchAny()
                            ))
                        else:
                            conditions.append(models.FieldCondition(
                                key=key,
                                match=models.IsNullCondition(is_null=True)
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
        # Обновляем статистику
        with self._lock:
            self.stats["generation_requests"] += 1

        if not self.llm or not self.generation_pipeline:
            with self._lock:
                self.stats["generation_failures"] += 1
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

            # Обновляем статистику успешной генерации
            with self._lock:
                self.stats["generation_successes"] += 1

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            answer = f"Не удалось сгенерировать ответ: {str(e)}"

            # Обновляем статистику неудачной генерации
            with self._lock:
                self.stats["generation_failures"] += 1

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

    def determine_ktru_code(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Определяет код КТРУ для одного товара

        Args:
            document: Документ с информацией о товаре

        Returns:
            Dict[str, Any]: Результат определения кода КТРУ
        """
        # Обновляем статистику
        service_stats.ktru_requests += 1

        if not self.generation_pipeline:
            service_stats.failed_ktru += 1
            raise ValueError("Языковая модель не инициализирована. Невозможно определить код КТРУ.")

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
{json.dumps(document, ensure_ascii=False)}

Найди наиболее подходящий код КТРУ для этого товара:
"""
        # Поиск документов по описанию товара
        search_query = f"{document.get('title', '')} {document.get('description', '')}"
        documents = self.search(
            query=search_query,
            limit=10  # Увеличиваем количество документов для повышения точности
        )

        # Форматируем контекст из найденных документов
        context = self._format_context_from_documents(documents)

        # Генерируем ответ с специальным промптом напрямую через LLM
        try:
            generated_text = self.generation_pipeline(prompt + "\nРеференсные данные из каталога КТРУ:\n" + context)[0]['generated_text']

            # Извлекаем только ответ модели
            answer = self._extract_answer(generated_text, prompt)

            # Проверяем формат ответа
            if answer.strip() == "код не найден":
                result = {"result": "код не найден", "confidence": "< 95%"}
                service_stats.failed_ktru += 1
            elif re.match(r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}', answer.strip()):
                result = {"result": answer.strip(), "confidence": "> 95%"}
                service_stats.successful_ktru += 1
            else:
                result = {"result": "код не найден", "confidence": "< 95%"}
                service_stats.failed_ktru += 1

            return result

        except Exception as e:
            logger.error(f"Ошибка при определении кода КТРУ: {e}")
            service_stats.failed_ktru += 1
            return {"result": "ошибка", "error": str(e), "confidence": "0%"}

    def determine_ktru_codes_batch(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
        """
        Определяет коды КТРУ для батча товаров

        Args:
            documents: Список документов с информацией о товарах
            batch_size: Размер подбатча для параллельной обработки

        Returns:
            Dict[str, Any]: Результаты определения кодов КТРУ и идентификатор задачи
        """
        # Создаем уникальный идентификатор задачи
        task_id = str(uuid.uuid4())

        # Создаем функцию для выполнения в фоне
        def process_batch():
            try:
                service_stats.batch_processes += 1

                results = []
                total = len(documents)
                processed = 0

                logger.info(f"Начало обработки пакетного определения КТРУ для {total} документов (task_id: {task_id})")

                # Обрабатываем документы небольшими группами
                for i in range(0, total, batch_size):
                    sub_batch = documents[i:i+batch_size]

                    # Определяем КТРУ для каждого документа в группе
                    for doc in sub_batch:
                        doc_id = doc.get("id", str(uuid.uuid4()))

                        # Определяем КТРУ для документа
                        ktru_result = self.determine_ktru_code(doc)

                        # Добавляем результат в общий список
                        results.append({
                            "document_id": doc_id,
                            "ktru_code": ktru_result.get("result"),
                            "confidence": ktru_result.get("confidence"),
                            "original_document": doc
                        })

                        processed += 1

                    # Обновляем прогресс
                    logger.info(f"Обработано {processed}/{total} документов для задачи {task_id}")

                # Сохраняем результаты
                with service_stats.lock:
                    service_stats.batch_results[task_id] = {
                        "status": "completed",
                        "total": total,
                        "processed": processed,
                        "results": results,
                        "completed_at": datetime.now().isoformat()
                    }

                logger.info(f"Завершена обработка пакетного определения КТРУ для задачи {task_id}")
                return results

            except Exception as e:
                logger.error(f"Ошибка при пакетном определении КТРУ (task_id: {task_id}): {e}")

                # Сохраняем информацию об ошибке
                with service_stats.lock:
                    service_stats.batch_results[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "total": len(documents),
                        "processed": 0,
                        "completed_at": datetime.now().isoformat()
                    }

                return {"error": str(e)}

        # Сохраняем информацию о задаче
        with service_stats.lock:
            service_stats.batch_jobs[task_id] = {
                "status": "queued",
                "total": len(documents),
                "created_at": datetime.now().isoformat()
            }

        # Запускаем задачу в фоновом режиме
        background_tasks.submit_task(task_id, process_batch)

        return {
            "task_id": task_id,
            "status": "queued",
            "total_documents": len(documents),
            "message": "Задача поставлена в очередь на выполнение"
        }

    def get_batch_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Получает статус выполнения пакетной задачи

        Args:
            task_id: Идентификатор задачи

        Returns:
            Dict[str, Any]: Статус задачи
        """
        # Проверяем статус задачи
        with service_stats.lock:
            # Проверяем, есть ли задача в завершенных
            if task_id in service_stats.batch_results:
                result = service_stats.batch_results[task_id]

                # Если задача уже завершена, возвращаем статус и, если нужно, результаты
                return {
                    "task_id": task_id,
                    "status": result["status"],
                    "total": result["total"],
                    "processed": result["processed"],
                    "completed_at": result["completed_at"],
                    "results_available": True
                }

            # Проверяем, есть ли задача в активных
            elif task_id in service_stats.batch_jobs:
                job = service_stats.batch_jobs[task_id]

                # Задача в процессе выполнения
                return {
                    "task_id": task_id,
                    "status": job["status"],
                    "total": job["total"],
                    "created_at": job["created_at"],
                    "message": "Задача выполняется"
                }

            # Задача не найдена
            else:
                return {
                    "task_id": task_id,
                    "status": "not_found",
                    "message": "Задача не найдена"
                }

    def get_batch_task_results(self, task_id: str) -> Dict[str, Any]:
        """
        Получает результаты выполнения пакетной задачи

        Args:
            task_id: Идентификатор задачи

        Returns:
            Dict[str, Any]: Результаты задачи
        """
        # Проверяем статус задачи
        with service_stats.lock:
            # Проверяем, есть ли задача в завершенных
            if task_id in service_stats.batch_results:
                result = service_stats.batch_results[task_id]

                # Если задача завершена успешно, возвращаем результаты
                if result["status"] == "completed":
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "total": result["total"],
                        "processed": result["processed"],
                        "completed_at": result["completed_at"],
                        "results": result["results"]
                    }

                # Если задача завершена с ошибкой
                else:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": result.get("error", "Неизвестная ошибка"),
                        "completed_at": result["completed_at"]
                    }

            # Задача не найдена или не завершена
            else:
                status = self.get_batch_task_status(task_id)

                if status["status"] == "not_found":
                    return {
                        "task_id": task_id,
                        "status": "not_found",
                        "message": "Задача не найдена"
                    }
                else:
                    return {
                        "task_id": task_id,
                        "status": status["status"],
                        "message": "Задача еще не завершена, результаты недоступны"
                    }

    def cancel_batch_task(self, task_id: str) -> Dict[str, Any]:
        """
        Отменяет выполнение пакетной задачи

        Args:
            task_id: Идентификатор задачи

        Returns:
            Dict[str, Any]: Результат операции
        """
        # Проверяем статус задачи
        task_status = background_tasks.get_task_status(task_id)

        if not task_status:
            return {
                "task_id": task_id,
                "status": "not_found",
                "message": "Задача не найдена"
            }

        # Если задача еще выполняется, пытаемся отменить
        if task_status["status"] == "running":
            # К сожалению, нельзя отменить уже запущенную задачу в ThreadPoolExecutor,
            # но можно пометить её как отмененную, чтобы не обрабатывать результаты
            with service_stats.lock:
                if task_id in service_stats.batch_jobs:
                    service_stats.batch_jobs[task_id]["status"] = "cancelled"

                    # Сохраняем информацию об отмене
                    service_stats.batch_results[task_id] = {
                        "status": "cancelled",
                        "total": service_stats.batch_jobs[task_id]["total"],
                        "processed": 0,
                        "completed_at": datetime.now().isoformat(),
                        "message": "Задача была отменена пользователем"
                    }

            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": "Задача помечена как отмененная"
            }

        # Если задача уже завершена
        else:
            return {
                "task_id": task_id,
                "status": task_status["status"],
                "message": f"Задача уже в статусе {task_status['status']}, отмена невозможна"
            }

    def count_documents(self, filter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Подсчитывает количество документов по заданным критериям

        Args:
            filter_params: Параметры фильтрации (опционально)

        Returns:
            Dict[str, Any]: Количество документов
        """
        try:
            # Настраиваем фильтр для подсчета, если он передан
            count_filter = None
            if filter_params:
                count_filter = Filter(
                    must=self._build_filter_conditions(filter_params)
                )

            # Выполняем подсчет
            count_result = self.qdrant_client.count(
                collection_name=self.collection_name,
                count_filter=count_filter
            )

            return {
                "count": count_result.count
            }

        except Exception as e:
            logger.error(f"Ошибка при подсчете документов: {e}")
            raise ValueError(f"Ошибка при подсчете документов: {e}")

    def cleanup_old_results(self, max_age_hours: int = 24) -> Dict[str, int]:
        """
        Очищает старые результаты выполнения пакетных задач

        Args:
            max_age_hours: Максимальный возраст результатов в часах

        Returns:
            Dict[str, int]: Количество очищенных результатов
        """
        with service_stats.lock:
            now = datetime.now()
            to_remove = []

            # Находим старые результаты
            for task_id, result in service_stats.batch_results.items():
                if "completed_at" in result:
                    completed_at = datetime.fromisoformat(result["completed_at"])
                    age_hours = (now - completed_at).total_seconds() / 3600

                    if age_hours > max_age_hours:
                        to_remove.append(task_id)

            # Удаляем старые результаты
            for task_id in to_remove:
                del service_stats.batch_results[task_id]

            # Также очищаем старые задачи
            old_jobs = []
            for task_id, job in service_stats.batch_jobs.items():
                if "created_at" in job:
                    created_at = datetime.fromisoformat(job["created_at"])
                    age_hours = (now - created_at).total_seconds() / 3600

                    if age_hours > max_age_hours:
                        old_jobs.append(task_id)

            # Удаляем старые задачи
            for task_id in old_jobs:
                if task_id in service_stats.batch_jobs:
                    del service_stats.batch_jobs[task_id]

            return {
                "removed_results": len(to_remove),
                "removed_jobs": len(old_jobs)
            }

    def update_document(self, doc_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обновляет документ в базе данных

        Args:
            doc_id: Идентификатор документа
            update_data: Данные для обновления

        Returns:
            Dict[str, Any]: Результат операции
        """
        try:
            # Получаем текущий документ
            search_result = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )

            if not search_result:
                return {
                    "status": "error",
                    "message": f"Документ с ID {doc_id} не найден"
                }

            # Получаем текущие данные документа
            current_doc = search_result[0].payload

            # Обновляем данные
            updated_doc = {**current_doc, **update_data}

            # Если изменился основной текст документа, нужно пересчитать эмбеддинг
            recalculate_embedding = (
                current_doc.get("title") != updated_doc.get("title") or
                current_doc.get("description") != updated_doc.get("description") or
                current_doc.get("attributes") != updated_doc.get("attributes")
            )

            if recalculate_embedding:
                # Создаем текстовое представление для эмбеддинга
                text = self._prepare_text_from_document(updated_doc)

                # Генерируем новый эмбеддинг
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)

                # Обновляем документ вместе с новым эмбеддингом
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=doc_id,
                            vector=embedding.tolist(),
                            payload=self._prepare_payload(updated_doc)
                        )
                    ]
                )

                return {
                    "status": "success",
                    "message": f"Документ с ID {doc_id} успешно обновлен с пересчетом эмбеддинга",
                    "document": updated_doc
                }
            else:
                # Обновляем только метаданные без пересчета эмбеддинга
                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload=self._prepare_payload(updated_doc),
                    points=[doc_id]
                )

                return {
                    "status": "success",
                    "message": f"Метаданные документа с ID {doc_id} успешно обновлены",
                    "document": updated_doc
                }

        except Exception as e:
            logger.error(f"Ошибка при обновлении документа {doc_id}: {e}")
            return {
                "status": "error",
                "message": f"Ошибка при обновлении документа: {str(e)}"
            }

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

class KtruCodeRequest(BaseModel):
    title: str
    description: Optional[str] = None
    attributes: Optional[List[Dict[str, Any]]] = None
    category: Optional[str] = None
    parent_category: Optional[str] = None
    brand: Optional[str] = None

    class Config:
        extra = "allow"  # Разрешаем дополнительные поля

    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        # Удаляем None значения для более чистого JSON
        return {k: v for k, v in result.items() if v is not None}

class BatchKtruCodeRequest(BaseModel):
    documents: List[KtruCodeRequest]
    batch_size: int = 10

    @validator('documents')
    def validate_documents(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Список документов не может быть пустым")

        if len(v) > 10000:
            raise ValueError("Слишком много документов для одного запроса. Максимум 10000.")

        return v

class CountRequest(BaseModel):
    filter_params: Optional[Dict[str, Any]] = None

class UpdateDocumentRequest(BaseModel):
    ktru_code: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # Разрешаем дополнительные поля

    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        # Удаляем None значения для более чистого JSON
        return {k: v for k, v in result.items() if v is not None}

class BatchTaskStatusRequest(BaseModel):
    task_id: str

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
        title="RAG API для технической документации и определения КТРУ",
        description="API для поиска, генерации ответов и определения КТРУ кодов для технической документации",
        version="2.0.0"
    )

    # Добавляем CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ============= Основные эндпоинты ===============

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
            result = rag_system.add_documents(documents, request.batch_size)
            return {
                "status": "success",
                "message": f'Добавлено {len(documents)} документов',
                **result
            }
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ============= Эндпоинты для КТРУ ===============

    @app.post("/api/ktru_code", summary="Определение кода КТРУ для одного товара")
    async def ktru_code_endpoint(document: KtruCodeRequest = Body(...)):
        try:
            if not rag_system.generation_pipeline:
                raise HTTPException(
                    status_code=400,
                    detail="Языковая модель не инициализирована. Невозможно определить код КТРУ."
                )

            result = rag_system.determine_ktru_code(document.dict())
            return result

        except Exception as e:
            logger.error(f"Ошибка при определении кода КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ktru_code/batch", summary="Определение кодов КТРУ для батча товаров")
    async def ktru_code_batch_endpoint(request: BatchKtruCodeRequest = Body(...)):
        try:
            if not rag_system.generation_pipeline:
                raise HTTPException(
                    status_code=400,
                    detail="Языковая модель не инициализирована. Невозможно определить коды КТРУ."
                )

            # Преобразуем Pydantic модели в словари
            documents = [doc.dict() for doc in request.documents]

            # Запускаем пакетную обработку
            result = rag_system.determine_ktru_codes_batch(documents, request.batch_size)

            # Возвращаем идентификатор задачи
            return result

        except Exception as e:
            logger.error(f"Ошибка при пакетном определении кодов КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/ktru_code/batch/{task_id}/status", summary="Получение статуса выполнения пакетной задачи определения КТРУ")
    async def ktru_code_batch_status_endpoint(task_id: str):
        try:
            result = rag_system.get_batch_task_status(task_id)
            return result

        except Exception as e:
            logger.error(f"Ошибка при получении статуса пакетной задачи определения КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/ktru_code/batch/{task_id}/results", summary="Получение результатов выполнения пакетной задачи определения КТРУ")
    async def ktru_code_batch_results_endpoint(task_id: str):
        try:
            result = rag_system.get_batch_task_results(task_id)

            # Если задача не завершена или не найдена, возвращаем ошибку
            if result["status"] not in ["completed", "failed"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Результаты недоступны. Статус задачи: {result['status']}"
                )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при получении результатов пакетной задачи определения КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ktru_code/batch/{task_id}/cancel", summary="Отмена выполнения пакетной задачи определения КТРУ")
    async def ktru_code_batch_cancel_endpoint(task_id: str):
        try:
            result = rag_system.cancel_batch_task(task_id)
            return result

        except Exception as e:
            logger.error(f"Ошибка при отмене пакетной задачи определения КТРУ: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ============= Эндпоинты для документов ===============

    @app.post("/api/documents/count", summary="Подсчет количества документов")
    async def count_documents_endpoint(request: CountRequest = Body(...)):
        try:
            result = rag_system.count_documents(request.filter_params)
            return result

        except Exception as e:
            logger.error(f"Ошибка при подсчете документов: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/documents/{doc_id}", summary="Обновление документа")
    async def update_document_endpoint(doc_id: str, request: UpdateDocumentRequest = Body(...)):
        try:
            result = rag_system.update_document(doc_id, request.dict())

            if result["status"] == "error":
                raise HTTPException(
                    status_code=400,
                    detail=result["message"]
                )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при обновлении документа: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ============= Эндпоинты для мониторинга ===============

    @app.get("/health", summary="Проверка работоспособности системы")
    async def health_check():
        # Обновляем информацию о загрузке системы
        service_stats.update_system_load()

        return {
            "status": "ok",
            "version": "2.0.0",
            "uptime": str(datetime.now() - service_stats.start_time),
            "system_load": service_stats.system_load,
            "documents_processed": service_stats.documents_processed,
            "ktru_requests": service_stats.ktru_requests,
            "successful_ktru": service_stats.successful_ktru,
            "failed_ktru": service_stats.failed_ktru,
            "batch_processes": service_stats.batch_processes,
            "active_batch_jobs": len(service_stats.batch_jobs)
        }

    @app.get("/stats", summary="Получение статистики системы")
    async def get_stats():
        # Получаем статистику Qdrant и других компонентов
        system_stats = rag_system.get_stats()

        # Дополняем глобальной статистикой
        system_stats["service"] = {
            "uptime": str(datetime.now() - service_stats.start_time),
            "documents_processed": service_stats.documents_processed,
            "ktru_requests": service_stats.ktru_requests,
            "successful_ktru": service_stats.successful_ktru,
            "failed_ktru": service_stats.failed_ktru,
            "batch_processes": service_stats.batch_processes,
            "active_batch_jobs": len(service_stats.batch_jobs),
            "completed_batch_jobs": len(service_stats.batch_results)
        }

        # Обновляем информацию о загрузке системы
        service_stats.update_system_load()
        system_stats["system_load"] = service_stats.system_load

        return system_stats

    @app.get("/stats/tasks", summary="Получение списка активных пакетных задач")
    async def get_active_tasks():
        with service_stats.lock:
            active_tasks = {}
            for task_id, job in service_stats.batch_jobs.items():
                if job["status"] == "queued" or job["status"] == "running":
                    active_tasks[task_id] = job

            return {
                "active_tasks_count": len(active_tasks),
                "tasks": active_tasks
            }

    @app.post("/maintenance/cleanup", summary="Очистка старых результатов пакетных задач")
    async def cleanup_old_results(max_age_hours: int = Query(24, description="Максимальный возраст результатов в часах")):
        try:
            result = rag_system.cleanup_old_results(max_age_hours)
            return result

        except Exception as e:
            logger.error(f"Ошибка при очистке старых результатов: {e}")
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
        "psutil": "5.9.0"  # Для мониторинга системы
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
    parser.add_argument('--worker-threads', type=int, default=DEFAULT_WORKER_THREADS,
                       help=f'Количество рабочих потоков для пакетной обработки (по умолчанию: {DEFAULT_WORKER_THREADS})')

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

        # Инициализация фоновых задач с нужным количеством потоков
        global background_tasks
        background_tasks = BackgroundTasks(num_workers=args.worker_threads)

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
