FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/
COPY indexing/ ./indexing/
COPY knowledge_base/ ./knowledge_base/

RUN mkdir -p /app/indexing/chroma /app/bot_logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV OPENAI_API_BASE=http://host.docker.internal:1234/v1
ENV OPENAI_API_KEY=sk-local

EXPOSE 8080

CMD ["python", "bot/rag_bot.py", "--persist-dir", "indexing/chroma", "--collection", "knowledge_base"]
