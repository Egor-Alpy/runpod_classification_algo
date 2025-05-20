from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uvicorn
import os
import json
from datetime import datetime
import config
from models import ModelManager
from document_processor import DocumentProcessor
from rag import RAGSystem

# Инициализация моделей и системы
model_manager = ModelManager()
model_manager.initialize_all()

document_processor = DocumentProcessor(model_manager)
rag_system = RAGSystem(model_manager)

# Создание FastAPI приложения
app = FastAPI(title="RAG-система для 50 млн JSON-документов",
              description="API для работы с RAG-системой на базе Hugging Face и Qdrant")


# Модели данных для API
class QueryRequest(BaseModel):
    query: str
    filter_params: Optional[Dict[str, Any]] = None
    temporal_data: Optional[Dict[str, Any]] = None
    num_documents: int = Field(default=config.TOP_K, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str
    filter_params: Optional[Dict[str, Any]] = None
    k: int = Field(default=config.TOP_K, ge=1, le=100)


class DocumentBase(BaseModel):
    ktru_code: str
    title: str
    description: Optional[str] = None
    unit: Optional[str] = None
    attributes: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[str]] = None
    updated_at: Optional[str] = None


class DocumentBatchRequest(BaseModel):
    documents: List[DocumentBase]


class DocumentDeleteRequest(BaseModel):
    ids: List[str]


class FileUploadRequest(BaseModel):
    file_path: str
    batch_size: Optional[int] = config.BATCH_SIZE
    max_workers: Optional[int] = config.MAX_WORKERS


# Маршруты API
@app.get("/")
async def root():
    return {
        "message": "RAG-система для 50 млн JSON-документов",
        "version": "1.0.0",
        "status": "active"
    }


@app.post("/api/generate")
async def generate_answer(request: QueryRequest):
    """
    Генерирует ответ на вопрос с использованием RAG
    """
    try:
        result = rag_system.process_query(
            query=request.query,
            filter_params=request.filter_params,
            temporal_data=request.temporal_data,
            num_documents=request.num_documents
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации ответа: {str(e)}")


@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """
    Выполняет поиск документов без генерации ответа
    """
    try:
        documents = rag_system.search_documents(
            query=request.query,
            filter_params=request.filter_params,
            k=request.k
        )
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске документов: {str(e)}")


@app.post("/api/add_documents")
async def add_documents(request: DocumentBatchRequest):
    """
    Добавляет пакет документов в систему
    """
    try:
        # Преобразуем модели Pydantic в словари
        raw_documents = [doc.model_dump() for doc in request.documents]

        # Устанавливаем дату обновления, если не указана
        for doc in raw_documents:
            if not doc.get("updated_at"):
                doc["updated_at"] = datetime.now().isoformat()

        # Обрабатываем документы
        processed_docs = [document_processor.process_json_document(doc) for doc in raw_documents]

        # Обновляем документы в Qdrant
        result = document_processor.update_documents(processed_docs)

        return {
            "status": "success",
            "updated": result["updated"],
            "total": result["total"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении документов: {str(e)}")


@app.post("/api/delete_documents")
async def delete_documents(request: DocumentDeleteRequest):
    """
    Удаляет документы из системы по ID
    """
    try:
        result = document_processor.delete_documents(request.ids)

        return {
            "status": "success",
            "deleted": result["deleted"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении документов: {str(e)}")


@app.post("/api/upload_file")
async def upload_file(request: FileUploadRequest, background_tasks: BackgroundTasks):
    """
    Загружает документы из JSON-файла в фоновом режиме
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"Файл не найден: {request.file_path}")

    # Запускаем загрузку в фоновом режиме
    background_tasks.add_task(
        document_processor.load_documents,
        [request.file_path],
        batch_size=request.batch_size,
        max_workers=request.max_workers
    )

    return {
        "status": "processing",
        "message": f"Начата загрузка файла {request.file_path}",
        "file_path": request.file_path
    }


@app.post("/api/upload_directory")
async def upload_directory(request: FileUploadRequest, background_tasks: BackgroundTasks):
    """
    Загружает все JSON-файлы из директории в фоновом режиме
    """
    if not os.path.isdir(request.file_path):
        raise HTTPException(status_code=404, detail=f"Директория не найдена: {request.file_path}")

    # Запускаем загрузку в фоновом режиме
    background_tasks.add_task(
        document_processor.load_directory,
        request.file_path,
        batch_size=request.batch_size,
        max_workers=request.max_workers
    )

    return {
        "status": "processing",
        "message": f"Начата загрузка файлов из директории {request.file_path}",
        "directory_path": request.file_path
    }


@app.get("/api/status")
async def get_status():
    """
    Возвращает статус системы
    """
    collection_info = model_manager.qdrant_client.get_collection(config.QDRANT_COLLECTION)

    return {
        "status": "active",
        "collection": collection_info.dict(),
        "embeddings_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "top_k": config.TOP_K,
        "device": model_manager.device
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )
