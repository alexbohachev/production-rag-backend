FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY corpus corpus
COPY eval eval
ENV PYTHONUNBUFFERED=1
ENV RERANK_BACKEND=feature
ENV EMBEDDING_BACKEND=hash
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
