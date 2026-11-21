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
RUN pip install --no-cache-dir --only-binary=:all: -r backend/requirements-product.txt
COPY backend/app ./backend/app
COPY backend/corpus ./backend/corpus
COPY backend/product_server.py ./backend/product_server.py
COPY --from=frontend /ui/dist ./frontend/dist
RUN mkdir -p /app/backend/data/reports && chown -R 10001:10001 /app/backend/data
USER 10001:10001
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=45s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready',timeout=40)"
CMD ["python", "backend/product_server.py", "--host", "0.0.0.0", "--port", "8000"]
