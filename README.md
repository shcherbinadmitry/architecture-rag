# RAG-бот 

## Структура проекта
```
architecture-rag/
├── knowledge_base/           # Источники
│   ├── originals/           # Оригинальный HTM
│   ├── clean/*.txt                # Очищенные файлы
│   ├── terms_map.json       # Маппинг замен
│   ├── prepare_data.py       # Подготовка текстовых файлов из HTML
│   └── build_terms_map.py   # Скрипт для замены названий
├── indexing/
│   ├── build_index.py       # Построение индекса ChromaDB
│   └── chroma/              # хранение файлов ChromaDB
├── bot/
│   ├── results               # скриншоты и результаты работы бота
│   ├── rag_bot.py           # RAG бот
│   ├── example_dialogues.md # Example conversations
│   └── requirements.txt
├── docker-compose.yml        
└── Dockerfile

```

## Запуск

### 1. Установка зависимостей

Использовать файлы requirements.txt для разных скриптом
```bash
pip install -r knowledge_base/requirements.txt
```

### 2. Подготовка данных

```bash
python knowledge_base/prepare_data.py                                                 

python knowledge_base/build_terms_map.py --rename-files                               
```

### 3. Создание индекса

```bash
python indexing/build_chroma.py --persist-dir indexing/chroma --collection knowledge_base --use-terms-map
```

### 4. Запуск бота

```bash
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_API_KEY=sk-local

python bot/rag_bot.py

# подробный вывод
python bot/rag_bot.py --verbose

# Запуск демо
python bot/rag_bot.py --demo
```