FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements*.txt ./backend/
RUN pip install --no-cache-dir --only-binary=:all: -r backend/requirements-cloud.txt
COPY backend/corpus ./backend/corpus
COPY backend/app/retrieval/local_model.py backend/app/retrieval/local_model_setup.py ./backend/app/retrieval/
ENV PYTHONPATH=/app/backend
RUN python -m app.retrieval.local_model_setup --download
COPY backend/app ./backend/app
COPY backend/product_server.py ./backend/product_server.py
COPY backend/cloud ./backend/cloud
COPY --from=frontend /ui/dist ./frontend/dist
RUN mkdir -p /app/backend/data/reports && chown 10001:10001 /app/backend/data /app/backend/data/reports
USER 10001:10001
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV SEMATRACE_INFERENCE_BATCH_SIZE=8
ENV SEMATRACE_RELEASE_IDLE_MEMORY=true
ENV MALLOC_ARENA_MAX=2
HEALTHCHECK --interval=60s --timeout=45s --start-period=90s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/api/ready',timeout=40)"
CMD ["python", "backend/product_server.py", "--host", "0.0.0.0"]
