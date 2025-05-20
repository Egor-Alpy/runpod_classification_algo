import json
import os
from typing import Dict, List, Any, Union, Optional
import concurrent.futures
from tqdm import tqdm
import torch
from qdrant_client.models import VectorParams, PointStruct
import config
from models import ModelManager


class DocumentProcessor:
    """
    Класс для обработки и загрузки документов в Qdrant
    """

    def __init__(self, model_manager: ModelManager = None):
        if model_manager is None:
            model_manager = ModelManager()
            model_manager.load_embedding_model()
            model_manager.init_qdrant()

        self.model_manager = model_manager
        self.embeddings = model_manager.embeddings
        self.qdrant_client = model_manager.qdrant_client
        self.collection_name = config.QDRANT_COLLECTION

    def process_json_document(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует JSON-документ в формат для векторизации

        Args:
            item: Исходный JSON-документ

        Returns:
            Dict: Обработанный документ с текстом и метаданными
        """
        # Формируем текстовое представление для эмбеддинга
        text = f"Код КТРУ: {item.get('ktru_code', '')}\n"
        text += f"Название: {item.get('title', '')}\n"

        if item.get('description'):
            text += f"Описание: {item.get('description', '')}\n"

        # Добавляем атрибуты
        if 'attributes' in item:
            text += "Характеристики:\n"
            for attr in item['attributes']:
                attr_name = attr.get('attr_name', '')
                attr_values = []
                for v in attr.get('attr_values', []):
                    value = v.get('value', '')
                    unit = v.get('value_unit', '')
                    if unit:
                        attr_values.append(f"{value} {unit}")
                    else:
                        attr_values.append(value)

                text += f"- {attr_name}: {', '.join(attr_values)}\n"

        # Создаем метаданные для поиска
        metadata = {
            "ktru_code": item.get('ktru_code', ''),
            "title": item.get('title', ''),
            "unit": item.get('unit', ''),
            "updated_at": item.get('updated_at', '')
        }

        # Добавляем ключевые слова, если есть
        if item.get('keywords'):
            metadata["keywords"] = item.get('keywords', [])

        # Добавляем плоские атрибуты для фильтрации
        if 'attributes' in item:
            attrs = {}
            for attr in item['attributes']:
                attr_name = attr.get('attr_name', '')
                # Преобразуем название атрибута для использования в качестве ключа
                attr_key = attr_name.lower().replace(' ', '_')

                # Добавляем значения как список
                attr_values = [v.get('value', '') for v in attr.get('attr_values', [])]
                if attr_values:
                    attrs[attr_key] = attr_values

            metadata["attributes"] = attrs

        # Определяем уникальный ID
        doc_id = item.get('ktru_code', '')
        if not doc_id and '_id' in item and '$oid' in item['_id']:
            doc_id = item['_id']['$oid']

        return {
            "text": text,
            "metadata": metadata,
            "id": doc_id
        }

    def load_json_batch(self, json_file_path: str, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Загружает и обрабатывает пакет документов из JSON-файла

        Args:
            json_file_path: Путь к JSON-файлу
            batch_size: Размер пакета

        Returns:
            List: Список обработанных документов
        """
        processed_documents = []

        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)

                # Проверяем, является ли данные списком или одиночным объектом
                if not isinstance(data, list):
                    data = [data]

                # Обрабатываем документы
                for item in data:
                    processed_doc = self.process_json_document(item)
                    processed_documents.append(processed_doc)

                    # Если достигли размера пакета, возвращаем
                    if len(processed_documents) >= batch_size:
                        return processed_documents[:batch_size]

            except json.JSONDecodeError as e:
                print(f"Ошибка декодирования JSON в файле {json_file_path}: {e}")

        return processed_documents

    def process_documents_parallel(self,
                                   json_files: List[str],
                                   max_workers: int = config.MAX_WORKERS,
                                   batch_size: int = config.BATCH_SIZE) -> List[Dict[str, Any]]:
        """
        Параллельно обрабатывает документы из нескольких JSON-файлов

        Args:
            json_files: Список путей к JSON-файлам
            max_workers: Максимальное количество рабочих потоков
            batch_size: Размер пакета для каждого рабочего потока

        Returns:
            List: Список обработанных документов
        """
        processed_documents = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи загрузки
            future_to_file = {
                executor.submit(self.load_json_batch, file_path, batch_size): file_path
                for file_path in json_files
            }

            # Собираем результаты
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                               total=len(future_to_file),
                               desc="Обработка файлов"):
                file_path = future_to_file[future]
                try:
                    batch = future.result()
                    processed_documents.extend(batch)
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")

        return processed_documents

    def create_embeddings_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Создает эмбеддинги для пакета документов

        Args:
            documents: Список обработанных документов

        Returns:
            List: Список документов с добавленными эмбеддингами
        """
        texts = [doc["text"] for doc in documents]

        # Создаем эмбеддинги
        embeddings_list = self.embeddings.embed_documents(texts)

        # Добавляем эмбеддинги к документам
        for i, embedding in enumerate(embeddings_list):
            documents[i]["embedding"] = embedding

        return documents

    def upload_to_qdrant(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Загружает пакет документов в Qdrant

        Args:
            documents: Список документов с эмбеддингами

        Returns:
            Dict: Статистика загрузки
        """
        # Преобразуем в формат Qdrant
        points = []
        for doc in documents:
            point = PointStruct(
                id=doc["id"],
                vector=doc["embedding"],
                payload={"text": doc["text"], **doc["metadata"]}
            )
            points.append(point)

        # Загружаем в Qdrant
        operation_info = self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return {
            "uploaded": len(points),
            "operation_id": operation_info.operation_id
        }

    def load_documents(self,
                       json_files: List[str],
                       batch_size: int = config.BATCH_SIZE,
                       max_workers: int = config.MAX_WORKERS) -> Dict[str, Any]:
        """
        Полный процесс загрузки документов из JSON-файлов в Qdrant

        Args:
            json_files: Список путей к JSON-файлам
            batch_size: Размер пакета для обработки
            max_workers: Максимальное количество рабочих потоков

        Returns:
            Dict: Статистика загрузки
        """
        total_processed = 0
        total_uploaded = 0

        # Обрабатываем файлы пакетами для экономии памяти
        for i in range(0, len(json_files), max_workers):
            # Берем пакет файлов
            file_batch = json_files[i:i + max_workers]

            # Параллельно обрабатываем файлы
            processed_docs = self.process_documents_parallel(
                file_batch,
                max_workers=max_workers,
                batch_size=batch_size
            )

            total_processed += len(processed_docs)

            # Создаем эмбеддинги и загружаем в Qdrant
            for j in range(0, len(processed_docs), batch_size):
                doc_batch = processed_docs[j:j + batch_size]

                # Создаем эмбеддинги
                docs_with_embeddings = self.create_embeddings_batch(doc_batch)

                # Загружаем в Qdrant
                upload_result = self.upload_to_qdrant(docs_with_embeddings)
                total_uploaded += upload_result["uploaded"]

                print(f"Загружено {upload_result['uploaded']} документов. "
                      f"Всего: {total_uploaded}/{total_processed}")

        return {
            "total_processed": total_processed,
            "total_uploaded": total_uploaded,
            "files_processed": len(json_files)
        }

    def load_directory(self,
                       directory_path: str,
                       batch_size: int = config.BATCH_SIZE,
                       max_workers: int = config.MAX_WORKERS,
                       extension: str = ".json") -> Dict[str, Any]:
        """
        Загружает все JSON-файлы из директории в Qdrant

        Args:
            directory_path: Путь к директории с JSON-файлами
            batch_size: Размер пакета для обработки
            max_workers: Максимальное количество рабочих потоков
            extension: Расширение файлов для обработки

        Returns:
            Dict: Статистика загрузки
        """
        # Находим все JSON-файлы в директории
        json_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(extension):
                    json_files.append(os.path.join(root, file))

        print(f"Найдено {len(json_files)} файлов в директории {directory_path}")

        # Загружаем файлы
        return self.load_documents(
            json_files=json_files,
            batch_size=batch_size,
            max_workers=max_workers
        )

    def update_documents(self,
                         documents: List[Dict[str, Any]],
                         batch_size: int = config.BATCH_SIZE) -> Dict[str, Any]:
        """
        Обновляет существующие документы в Qdrant

        Args:
            documents: Список документов для обновления
            batch_size: Размер пакета

        Returns:
            Dict: Статистика обновления
        """
        updated_count = 0

        # Разделяем на батчи для обновления
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Создаем эмбеддинги
            docs_with_embeddings = self.create_embeddings_batch(batch)

            # Обновляем в Qdrant
            upload_result = self.upload_to_qdrant(docs_with_embeddings)
            updated_count += upload_result["uploaded"]

        return {
            "updated": updated_count,
            "total": len(documents)
        }

    def delete_documents(self, ids: List[str]) -> Dict[str, Any]:
        """
        Удаляет документы из Qdrant по ID

        Args:
            ids: Список ID документов для удаления

        Returns:
            Dict: Статистика удаления
        """
        operation_info = self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

        return {
            "deleted": len(ids),
            "operation_id": operation_info.operation_id
        }
