# ocr-service-paddle

Microservice OCR HTTP — moteur **PaddleOCR** (CPU).

Identique à `ocr-service-p-eye` dans son architecture Node.js + Python worker pool, mais utilise **PaddleOCR** à la place de Doctr pour la reconnaissance de texte.

## Stack

| Couche | Technologie |
|--------|-------------|
| Serveur HTTP | Node.js 20 + Express |
| OCR engine | PaddleOCR ≥ 2.7.3 (Python 3.11, CPU) |
| Rasterisation PDF | PyMuPDF (fitz) |
| Communication | stdin/stdout JSON (worker pool) |

## API

### `GET /health`
```json
{ "ok": true, "engine": "paddleocr", "workers": [...], "queue_size": 0 }
```

### `POST /ocr`

| Paramètre | Type | Description |
|-----------|------|-------------|
| `file` | multipart | Fichier PDF (≤ 25 MB) |
| `lang` | query string | `fra` (défaut), `eng`, `deu`, … |

Réponse :
```json
{ "text": "...", "page_count": 3 }
```

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `PORT` | `3000` | Port d'écoute |
| `MAX_FILE_SIZE_MB` | `25` | Taille max upload |
| `OCR_TIMEOUT_MS` | `60000` | Timeout par requête OCR |
| `WORKER_COUNT` | `min(CPUs, 4)` | Nombre de workers Python |
| `QUEUE_MAX_SIZE` | `50` | Taille max de la file d'attente |

## Build Docker

```bash
docker build -t ocr-service-paddle .
docker run -p 3000:3000 --memory=4g ocr-service-paddle
```

## Test rapide

```bash
curl -F "file=@mon_document.pdf" http://localhost:3000/ocr
```
