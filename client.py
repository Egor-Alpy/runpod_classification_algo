import requests
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown


class RAGClient:
    """
    Клиент для работы с API RAG-системы
    """

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.console = Console()

    def _make_request(self, method, endpoint, data=None):
        """
        Выполняет запрос к API
        """
        url = f"{self.base_url}{endpoint}"

        if method.lower() == "get":
            response = requests.get(url)
        elif method.lower() == "post":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Неподдерживаемый метод: {method}")

        return response.json()

    def generate_answer(self, query, filter_params=None, temporal_data=None, num_documents=5):
        """
        Генерирует ответ на вопрос с использованием RAG
        """
        data = {
            "query": query,
            "num_documents": num_documents
        }

        if filter_params:
            data["filter_params"] = filter_params

        if temporal_data:
            data["temporal_data"] = temporal_data

        result = self._make_request("post", "/api/generate", data)

        # Выводим результат
        self.console.print("\n[bold green]Вопрос:[/bold green]", query)

        if temporal_data:
            self.console.print("\n[bold yellow]Временные данные:[/bold yellow]")
            self.console.print(json.dumps(temporal_data, indent=2, ensure_ascii=False))

        self.console.print("\n[bold blue]Ответ:[/bold blue]")
        self.console.print(Markdown(result["answer"]))

        # Выводим источники
        if "source_documents" in result and result["source_documents"]:
            self.console.print("\n[bold]Источники:[/bold]")

            table = Table(show_header=True)
            table.add_column("№", style="dim")
            table.add_column("Код КТРУ", style="yellow")
            table.add_column("Название", style="cyan")
            table.add_column("Единица изм.", style="green")

            for i, doc in enumerate(result["source_documents"], 1):
                table.add_row(
                    str(i),
                    doc["metadata"].get("ktru_code", "-"),
                    doc["metadata"].get("title", "-"),
                    doc["metadata"].get("unit", "-")
                )

            self.console.print(table)

        return result

    def search_documents(self, query, filter_params=None, k=5):
        """
        Поиск документов без генерации ответа
        """
        data = {
            "query": query,
            "k": k
        }

        if filter_params:
            data["filter_params"] = filter_params

        result = self._make_request("post", "/api/search", data)

        # Выводим результат
        self.console.print("\n[bold green]Запрос:[/bold green]", query)

        if filter_params:
            self.console.print("\n[bold yellow]Фильтр:[/bold yellow]")
            self.console.print(json.dumps(filter_params, indent=2, ensure_ascii=False))

        # Выводим найденные документы
        if "documents" in result and result["documents"]:
            self.console.print("\n[bold]Найденные документы:[/bold]")

            table = Table(show_header=True)
            table.add_column("№", style="dim")
            table.add_column("Код КТРУ", style="yellow")
            table.add_column("Название", style="cyan")
            table.add_column("Релевантность", style="magenta")

            for i, doc in enumerate(result["documents"], 1):
                table.add_row(
                    str(i),
                    doc["metadata"].get("ktru_code", "-"),
                    doc["metadata"].get("title", "-"),
                    f"{doc['score']:.3f}"
                )

            self.console.print(table)

        return result

    def get_status(self):
        """
        Получение статуса системы
        """
        result = self._make_request("get", "/api/status")

        # Выводим информацию о системе
        self.console.print("\n[bold green]Статус RAG-системы:[/bold green]")
        self.console.print(f"Статус: [bold]{result['status']}[/bold]")
        self.console.print(f"Модель эмбеддингов: [yellow]{result['embeddings_model']}[/yellow]")
        self.console.print(f"Языковая модель: [blue]{result['llm_model']}[/blue]")
        self.console.print(f"Устройство: [magenta]{result['device']}[/magenta]")

        # Информация о коллекции
        self.console.print("\n[bold]Информация о коллекции:[/bold]")
        self.console.print(f"Имя: [cyan]{result['collection']['name']}[/cyan]")
        self.console.print(f"Количество точек: [green]{result['collection']['vectors_count']}[/green]")
        self.console.print(f"Количество сегментов: {result['collection']['segments_count']}")

        return result


# Если запускаем скрипт напрямую
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Клиент для RAG-системы")
    parser.add_argument("--url", default="http://localhost:8000", help="URL API")

    subparsers = parser.add_subparsers(dest="command", help="Команда")

    # Команда для генерации ответа
    generate_parser = subparsers.add_parser("generate", help="Генерация ответа")
    generate_parser.add_argument("query", help="Вопрос")
    generate_parser.add_argument("--filter", help="Фильтр (JSON)")
    generate_parser.add_argument("--temporal", help="Временные данные (JSON)")
    generate_parser.add_argument("--k", type=int, default=5, help="Количество документов")

    # Команда для поиска документов
    search_parser = subparsers.add_parser("search", help="Поиск документов")
    search_parser.add_argument("query", help="Запрос")
    search_parser.add_argument("--filter", help="Фильтр (JSON)")
    search_parser.add_argument("--k", type=int, default=5, help="Количество документов")

    # Команда для получения статуса
    status_parser = subparsers.add_parser("status", help="Статус системы")

    args = parser.parse_args()

    client = RAGClient(args.url)

    if args.command == "generate":
        filter_params = json.loads(args.filter) if args.filter else None
        temporal_data = json.loads(args.temporal) if args.temporal else None

        client.generate_answer(
            query=args.query,
            filter_params=filter_params,
            temporal_data=temporal_data,
            num_documents=args.k
        )
    elif args.command == "search":
        filter_params = json.loads(args.filter) if args.filter else None

        client.search_documents(
            query=args.query,
            filter_params=filter_params,
            k=args.k
        )
    elif args.command == "status":
        client.get_status()
    else:
        parser.print_help()
