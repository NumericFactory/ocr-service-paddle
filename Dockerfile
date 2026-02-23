# ─── Stage 1 : Node deps ──────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS node-deps

WORKDIR /app
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi \
    && npm cache clean --force

# ─── Stage 2 : Python deps (PaddleOCR + PyMuPDF) ──────────────────────────────
FROM python:3.11-slim-bookworm AS python-deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLAGS_call_stack_level=2

# Dépendances système requises par Paddle et PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgeos-dev \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# PaddleOCR (CPU) + PyMuPDF (rasterisation PDF)
#RUN pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#RUN pip install "paddleocr>=2.7.3" pymupdf
RUN pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install paddleocr pymupdf

COPY download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py

# ─── Stage 3 : runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    FLAGS_call_stack_level=2 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgeos-dev \
    libxrender1 \
    libxext6 \
    ca-certificates \
    curl \
    wget \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Les modèles PaddleOCR sont mis en cache dans ~/.paddleocr
COPY --from=python-deps /root/.paddlex /home/appuser/.paddlex

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser \
    && chown -R appuser:appgroup /home/appuser/.paddlex

# Variable d'environnement pour que PaddleOCR utilise le cache pré-téléchargé
ENV HOME=/home/appuser

WORKDIR /app
COPY --from=node-deps /app/node_modules ./node_modules
COPY server.js ocr_worker.py ./
RUN chown -R appuser:appgroup /app

USER appuser
STOPSIGNAL SIGTERM

# start-period élevé : les N workers chargent leur modèle en parallèle au boot
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=5 \
    CMD wget -qO- http://localhost:3000/health | grep -q '"ok":true' || exit 1

CMD ["node", "server.js"]
