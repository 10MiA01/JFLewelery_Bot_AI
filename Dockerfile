# Базовый образ с Python 3.10
FROM python:3.10-slim

# Обновляем pip и system-зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --upgrade pip

# Копируем файл зависимостей
COPY requirements.txt /app/requirements.txt

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем все Python-зависимости
RUN pip install --no-cache-dir --timeout=100 --retries=10 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Копируем остальной код проекта
COPY . /app

# Открываем порт (если нужен)
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
