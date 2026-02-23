# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 : node-deps
# ──────────────────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS node-deps

WORKDIR /app
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi \
    && npm cache clean --force


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 : python-deps (lib + modèles PP-OCRv5 + cache PaddleX)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS python-deps
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PADDLEOCR_HOME=/models \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FLAGS_call_stack_level=2 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tar \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Installe PaddlePaddle + PaddleOCR (PP-OCRv5) + PyMuPDF
RUN pip install --no-cache-dir "paddlepaddle==3.3.0" \
    && pip install --no-cache-dir "paddleocr==3.4.0" \
    && pip install --no-cache-dir "pymupdf==1.24.10"

# ─── Téléchargement des modèles PP-OCRv5 (server) + orientation ───────────────
RUN mkdir -p /models/ppocrv5/det /models/ppocrv5/rec /models/ppocrv5/cls \
    && curl -L --retry 3 --retry-delay 2 -o /tmp/det.tar \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar" \
    && tar -xf /tmp/det.tar -C /models/ppocrv5/det --strip-components=1 \
    && rm -f /tmp/det.tar \
    && curl -L --retry 3 --retry-delay 2 -o /tmp/rec.tar \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar" \
    && tar -xf /tmp/rec.tar -C /models/ppocrv5/rec --strip-components=1 \
    && rm -f /tmp/rec.tar \
    && curl -L --retry 3 --retry-delay 2 -o /tmp/cls.tar \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar" \
    && tar -xf /tmp/cls.tar -C /models/ppocrv5/cls --strip-components=1 \
    && rm -f /tmp/cls.tar

# ─── Pré-chauffe PaddleOCR au build → télécharge UVDoc, PP-LCNet_x1_0_doc_ori
# et tout autre modèle auxiliaire que paddleocr==3.4.0 charge au premier appel.
# PaddleX ignore PADDLEX_HOME et écrit dans /root/.paddlex (HOME du build).
COPY download_models.py /app/download_models.py
RUN python3 /app/download_models.py


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 : runtime (image finale)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLEOCR_HOME=/models \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FLAGS_call_stack_level=2 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_use_mkldnn=0

# Dépendances système + Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    gnupg \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libxrender1 \
    libxext6 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Libs Python depuis python-deps
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Modèles PP-OCRv5 baked
COPY --from=python-deps /models /models

# Utilisateur non-root (créé avant le COPY pour pouvoir chown en une passe)
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser

# Cache PaddleX baked au build — PaddleX écrit dans $HOME/.paddlex
# On le place donc dans le home de appuser : trouvé automatiquement à runtime.
COPY --from=python-deps /root/.paddlex /home/appuser/.paddlex

# Code app + dépendances Node
COPY --from=node-deps /app/node_modules ./node_modules
COPY server.js ocr_worker.py ./
RUN chown -R appuser:appgroup /home/appuser /app

USER appuser
STOPSIGNAL SIGTERM

# start-period élevé : les N workers PaddleOCR chargent leur modèle au boot
# (modèles en cache → plus rapide, mais loading reste lourd)
HEALTHCHECK --interval=20s --timeout=5s --start-period=180s --retries=5 \
    CMD wget -qO- http://localhost:3000/health | grep -q '"ok":true' || exit 1

CMD ["node", "server.js"]