from typing import Dict, List, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import config
from models import ModelManager


class RAGSystem:
    """
    Реализация системы Retrieval-Augmented Generation (RAG)
    """

    def __init__(self, model_manager: ModelManager = None):
        if model_manager is None:
            model_manager = ModelManager()
            model_manager.initialize_all()

        self.model_manager = model_manager
        self.llm = model_manager.llm
        self.embeddings = model_manager.embeddings
        self.vectorstore = model_manager.vectorstore

        # Инициализация ретривера и RAG-цепочки
        self.retriever = self._create_retriever()
        self.rag_chain = self._create_rag_chain()

    def _create_retriever(self):
        """
        Создает ретривер для поиска документов
        """
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

        # Фильтр на основе эмбеддингов для повышения точности
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.7  # Порог сходства для фильтрации
        )

        # Создаем улучшенный ретривер с компрессией контекста
        retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )

        return retriever

    def _create_rag_chain(self):
        """
        Создает RAG-цепочку
        """
        # Определяем шаблон промпта для генерации
        template = """
Ты - помощник по медицинским товарам и оборудованию. Используй предоставленную информацию, чтобы ответить на вопрос пользователя.

Контекст:
{context}

Вопрос: {question}

Если в контексте недостаточно информации, так и скажи. Не придумывай информацию.

Ответ:
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Создаем RAG-цепочку
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" загружает все документы в промпт
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return rag_chain

    def process_query(self,
                      query: str,
                      filter_params: Optional[Dict[str, Any]] = None,
                      temporal_data: Optional[Dict[str, Any]] = None,
                      num_documents: int = config.TOP_K) -> Dict[str, Any]:
        """
        Обрабатывает запрос с возможностью фильтрации и временными данными

        Args:
            query: Запрос пользователя
            filter_params: Параметры фильтрации по атрибутам
            temporal_data: Временные данные, актуальные на момент запроса
            num_documents: Количество документов для поиска

        Returns:
            Dict: Результат запроса с ответом и источниками
        """
        # Модифицируем запрос с учетом временных данных
        if temporal_data:
            temporal_context = "\nАктуальные данные:\n"
            for key, value in temporal_data.items():
                if isinstance(value, dict):
                    temporal_context += f"- {key}:\n"
                    for subkey, subvalue in value.items():
                        temporal_context += f"  * {subkey}: {subvalue}\n"
                else:
                    temporal_context += f"- {key}: {value}\n"

            enhanced_query = f"{query}\n{temporal_context}"
        else:
            enhanced_query = query

        # Если есть параметры фильтрации, создаем новый ретривер
        if filter_params:
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": num_documents,
                    "filter": filter_params
                }
            )

            # Создаем временную цепочку с фильтрованным ретривером
            temp_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=filtered_retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.rag_chain.combine_documents_chain.llm_chain.prompt
                }
            )

            # Выполняем запрос
            result = temp_chain({"query": enhanced_query})
        else:
            # Используем основную цепочку
            result = self.rag_chain({"query": enhanced_query})

        # Форматируем результат
        source_documents = result.get("source_documents", [])
        sources = []

        for doc in source_documents:
            source = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source)

        return {
            "query": query,
            "enhanced_query": enhanced_query if temporal_data else None,
            "answer": result["result"],
            "source_documents": sources,
            "filter_params": filter_params,
            "temporal_data": temporal_data
        }

    def search_documents(self,
                         query: str,
                         filter_params: Optional[Dict[str, Any]] = None,
                         k: int = config.TOP_K) -> List[Dict[str, Any]]:
        """
        Выполняет поиск документов без генерации ответа

        Args:
            query: Запрос пользователя
            filter_params: Параметры фильтрации
            k: Количество документов для возврата

        Returns:
            List: Список найденных документов
        """
        if filter_params:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_params
            )
        else:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

        # Форматируем результаты
        documents = []
        for doc, score in results:
            document = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            documents.append(document)

        return documents
    