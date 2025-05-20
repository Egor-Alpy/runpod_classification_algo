#!/bin/bash

# Скрипт для автоматизации запуска RAG-системы
# Проверяет необходимые компоненты и запускает их при необходимости

# Функция проверки и запуска Qdrant
check_and_start_qdrant() {
  echo "Проверка статуса Qdrant..."

  # Проверяем доступность Qdrant
  if curl -s http://localhost:6333 > /dev/null; then
    echo "Qdrant уже запущен на порту 6333"
  else
    echo "Qdrant не запущен. Запускаем..."

    # Проверяем, что исполняемый файл qdrant существует
    if [ -f "./qdrant" ] && [ -x "./qdrant" ]; then
      # Запускаем Qdrant в фоновом режиме
      ./qdrant &

      # Подождем секунду для запуска
      sleep 5

      # Проверяем, запустился ли Qdrant
      if curl -s http://localhost:6333 > /dev/null; then
        echo "Qdrant успешно запущен"
      else
        echo "ОШИБКА: Не удалось запустить Qdrant"
        exit 1
      fi
    else
      echo "ОШИБКА: Исполняемый файл Qdrant не найден в текущей директории"
      echo "Убедитесь, что вы загрузили и распаковали Qdrant"
      exit 1
    fi
  fi
}

# Функция проверки необходимых зависимостей
check_dependencies() {
  echo "Проверка Python-зависимостей..."

  # Список необходимых пакетов
  required_packages="sentence-transformers qdrant-client fastapi uvicorn torch transformers"

  for package in $required_packages; do
    if ! pip list | grep -i "$package" > /dev/null; then
      echo "Отсутствует пакет: $package. Устанавливаем..."
      pip install "$package"
    fi
  done

  echo "Все необходимые зависимости установлены"
}

# Главная логика скрипта
main() {
  echo "=== Запуск RAG-системы ==="

  # Проверяем зависимости
  check_dependencies

  # Проверяем и запускаем Qdrant
  check_and_start_qdrant

  # Запускаем RAG-систему
  echo "Запуск RAG-системы..."
  python3 rag_system.py "$@"
}

# Запускаем главную функцию с передачей всех аргументов командной строки
main "$@"
