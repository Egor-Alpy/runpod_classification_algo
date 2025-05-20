#!/bin/bash

# Скрипт для автоматизации запуска RAG-системы
# Проверяет необходимые компоненты и запускает их при необходимости

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция проверки и запуска Qdrant
check_and_start_qdrant() {
  echo -e "${BLUE}[INFO]${NC} Проверка статуса Qdrant..."

  # Проверяем доступность Qdrant
  if curl -s http://localhost:6333 > /dev/null; then
    echo -e "${GREEN}[OK]${NC} Qdrant уже запущен на порту 6333"
  else
    echo -e "${YELLOW}[WARN]${NC} Qdrant не запущен. Запускаем..."

    # Проверяем, что исполняемый файл qdrant существует
    if [ -f "./qdrant" ] && [ -x "./qdrant" ]; then
      # Запускаем Qdrant в фоновом режиме
      ./qdrant > ./qdrant.log 2>&1 &

      # Подождем для запуска
      echo -e "${BLUE}[INFO]${NC} Ожидание запуска Qdrant..."
      sleep 5

      # Проверяем, запустился ли Qdrant
      if curl -s http://localhost:6333 > /dev/null; then
        echo -e "${GREEN}[OK]${NC} Qdrant успешно запущен"
      else
        echo -e "${RED}[ERROR]${NC} Не удалось запустить Qdrant"
        echo -e "${YELLOW}[WARN]${NC} Пробуем запуск через Docker..."

        # Проверяем наличие Docker
        if command -v docker &> /dev/null; then
          echo -e "${BLUE}[INFO]${NC} Docker найден, запускаем Qdrant в контейнере..."
          docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

          # Ждем запуска
          sleep 10

          if curl -s http://localhost:6333 > /dev/null; then
            echo -e "${GREEN}[OK]${NC} Qdrant успешно запущен в Docker"
          else
            echo -e "${RED}[ERROR]${NC} Не удалось запустить Qdrant даже через Docker"
            exit 1
          fi
        else
          echo -e "${RED}[ERROR]${NC} Docker не установлен. Не удалось запустить Qdrant"
          exit 1
        fi
      fi
    else
      echo -e "${YELLOW}[WARN]${NC} Исполняемый файл Qdrant не найден в текущей директории"
      echo -e "${BLUE}[INFO]${NC} Пытаемся скачать и распаковать Qdrant..."

      # Скачиваем и распаковываем Qdrant
      curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz
      tar -xzf qdrant.tar.gz
      chmod +x ./qdrant

      # Запускаем Qdrant
      ./qdrant > ./qdrant.log 2>&1 &

      # Ждем запуска
      echo -e "${BLUE}[INFO]${NC} Ожидание запуска Qdrant..."
      sleep 5

      # Проверяем, запустился ли Qdrant
      if curl -s http://localhost:6333 > /dev/null; then
        echo -e "${GREEN}[OK]${NC} Qdrant успешно запущен после автоматической установки"
      else
        echo -e "${RED}[ERROR]${NC} Не удалось запустить Qdrant после автоматической установки"
        exit 1
      fi
    fi
  fi
}

# Функция поиска JSON-файлов для импорта
find_json_files() {
  echo -e "${BLUE}[INFO]${NC} Поиск JSON-файлов для импорта..."

  # Создаем директорию data, если она не существует
  mkdir -p data

  # Ищем все JSON-файлы в текущей директории и директории data
  # Исключаем файлы из node_modules, .git и других служебных директорий
  JSON_FILES=$(find . -type f -name "*.json" -not -path "*/\.*" -not -path "*/node_modules/*" | tr '\n' ' ')

  if [ -n "$JSON_FILES" ]; then
    echo -e "${GREEN}[OK]${NC} Найдены JSON-файлы для импорта"
    echo -e "${BLUE}[INFO]${NC} Файлы: $JSON_FILES"
    IMPORT_ARGS="--import-data $JSON_FILES"
  else
    echo -e "${YELLOW}[WARN]${NC} JSON-файлы для импорта не найдены"
    IMPORT_ARGS=""
  fi
}

# Функция проверки необходимых зависимостей
check_dependencies() {
  echo -e "${BLUE}[INFO]${NC} Проверка Python-зависимостей..."

  # Список необходимых пакетов
  required_packages=(
    "sentence-transformers>=2.2.2"
    "qdrant-client>=1.5.0"
    "fastapi>=0.100.0"
    "uvicorn[standard]>=0.23.0"
    "torch>=2.1.0"
    "transformers>=4.37.2"
    "psutil>=5.9.0"
    "tqdm>=4.64.0"
  )

  # Проверяем наличие pip
  if ! command -v pip &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} pip не установлен"

    # Пробуем установить pip
    echo -e "${BLUE}[INFO]${NC} Пытаемся установить pip..."
    apt-get update && apt-get install -y python3-pip || {
      echo -e "${RED}[ERROR]${NC} Не удалось установить pip. Установите его вручную."
      exit 1
    }
  fi

  # Устанавливаем отсутствующие пакеты
  for package in "${required_packages[@]}"; do
    package_name=$(echo "$package" | cut -d'>=' -f1)
    if ! pip list | grep -i "$package_name" > /dev/null; then
      echo -e "${YELLOW}[WARN]${NC} Отсутствует пакет: $package. Устанавливаем..."
      pip install "$package" || {
        echo -e "${RED}[ERROR]${NC} Не удалось установить $package"
        exit 1
      }
    fi
  done

  echo -e "${GREEN}[OK]${NC} Все необходимые зависимости установлены"
}

# Функция проверки GPU
check_gpu() {
  echo -e "${BLUE}[INFO]${NC} Проверка наличия и настройки GPU..."

  # Проверяем наличие GPU с поддержкой CUDA
  if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}[OK]${NC} GPU с поддержкой CUDA найден"
    GPU_ARGS="--use-gpu"

    # Получаем информацию о GPU
    GPU_INFO=$(python3 -c "import torch; print(f'CUDA Version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')")
    echo -e "${BLUE}[INFO]${NC} $GPU_INFO"
  else
    echo -e "${YELLOW}[WARN]${NC} GPU с поддержкой CUDA не найден. Будет использован CPU"
    GPU_ARGS=""
  fi
}

# Функция для запуска автоматического мониторинга
start_monitoring() {
  echo -e "${BLUE}[INFO]${NC} Запуск автоматического мониторинга (каждые 5 минут)..."

  # Создаем скрипт для мониторинга
  cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
  echo "$(date) - Проверка работоспособности системы..."
  HEALTH=$(curl -s http://localhost:8000/health)
  echo "$HEALTH" | jq .

  # Проверяем статистику документов
  STATS=$(curl -s http://localhost:8000/stats)
  echo "Статистика документов:"
  echo "$STATS" | jq .qdrant

  # Проверяем активные задачи
  TASKS=$(curl -s http://localhost:8000/stats/tasks)
  ACTIVE_TASKS=$(echo "$TASKS" | jq .active_tasks_count)
  echo "Активных задач: $ACTIVE_TASKS"

  # Очистка старых задач каждые 3 часа
  HOUR=$(date +%H)
  if [ $((HOUR % 3)) -eq 0 ]; then
    echo "Очистка старых задач..."
    curl -s -X POST http://localhost:8000/maintenance/cleanup
  fi

  echo "Следующая проверка через 5 минут..."
  sleep 300
done
EOF

  chmod +x monitor.sh

  # Запускаем мониторинг в фоновом режиме
  ./monitor.sh > monitoring.log 2>&1 &
  echo -e "${GREEN}[OK]${NC} Мониторинг запущен в фоновом режиме (логи в monitoring.log)"
}

# Главная функция скрипта
main() {
  echo -e "${BLUE}===========================================${NC}"
  echo -e "${GREEN}=== Запуск улучшенной RAG-системы v2.0 ===${NC}"
  echo -e "${BLUE}===========================================${NC}"

  # Проверяем зависимости
  check_dependencies

  # Проверяем и запускаем Qdrant
  check_and_start_qdrant

  # Проверяем GPU
  check_gpu

  # Ищем JSON-файлы для импорта
  find_json_files

  # Запускаем RAG-систему
  echo -e "${BLUE}[INFO]${NC} Запуск RAG-системы..."

  # Определяем количество рабочих потоков в зависимости от доступных ядер CPU
  CPU_CORES=$(python3 -c "import psutil; print(psutil.cpu_count(logical=False))")
  WORKER_THREADS=$((CPU_CORES > 0 ? CPU_CORES : 2))

  echo -e "${BLUE}[INFO]${NC} Запуск с $WORKER_THREADS рабочими потоками"

  # Запускаем RAG-систему с соответствующими аргументами
  python3 rag_system.py $IMPORT_ARGS $GPU_ARGS --worker-threads $WORKER_THREADS "$@"

  # Запускаем мониторинг
  start_monitoring
}

# Запускаем главную функцию с передачей всех аргументов командной строки
main "$@"