# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 : node-deps (si tu as un front)
# ──────────────────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS node-deps
WORKDIR /app

# (Optionnel) si tu as package.json/package-lock.json
# COPY package*.json ./
# RUN npm ci

# COPY . .
# RUN npm run build


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 : python-deps (lib + modèles PP-OCRv5)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS python-deps
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLEOCR_HOME=/models \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Déps système minimales (certs + curl + libs souvent nécessaires)
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

# ⚠️ Important pour éviter des segfault/free() invalid pointer (souvent OpenMP)
# - force OpenBLAS en mono-thread au build
# - évite les sur-parallélisations dans les containers
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Installe PaddlePaddle + PaddleOCR (PP-OCRv5)
# NOTE: ajuste les versions si besoin, mais garde-les compatibles entre elles.
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
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar" \
    && tar -xf /tmp/cls.tar -C /models/ppocrv5/cls --strip-components=1 \
    && rm -f /tmp/cls.tar

# (Optionnel) si tu veux garder ton script (mais NE PAS l'exécuter au build si ça crash)
COPY download_models.py /app/download_models.py


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 : runtime (image finale)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLEOCR_HOME=/models \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Libs python + modèles depuis python-deps
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin
COPY --from=python-deps /models /models

# Ton code app
# COPY . /app

# Exemple : lance ton worker OCR (à adapter au nom de ton fichier)
# CMD ["python", "/app/ocr_worker.py"]

CMD ["python", "-c", "print('Container ready. Models in /models/ppocrv5')"]