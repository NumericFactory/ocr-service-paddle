#!/usr/bin/env python3
"""
OCR worker persistant — lit des requêtes JSON sur stdin, écrit des réponses JSON sur stdout.
Modèle PaddleOCR chargé UNE SEULE FOIS au démarrage du process.

Protocole :
  stdin  → {"id": "abc", "pdf_path": "/tmp/input.pdf"}
  stdout → {"id": "abc", "text": "...", "page_count": N}
         | {"id": "abc", "error": "message"}
  stdout → {"ready": true}  (au démarrage, une seule fois)
"""

import sys
import json
import os
import logging
import traceback
import builtins

# ── Tout vers stderr avant imports ────────────────────────────────────────────
os.environ["FLAGS_call_stack_level"] = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_orig_print = builtins.print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _orig_print(*args, **kwargs)


builtins.print = _stderr_print


def emit(obj):
    _orig_print(json.dumps(obj, ensure_ascii=False), file=sys.stdout, flush=True)


# ── Chargement modèle ─────────────────────────────────────────────────────────

def load_model(lang: str = "fr"):
    """
    Charge PaddleOCR en restant compatible avec plusieurs versions :
    - anciennes: use_angle_cls / use_gpu
    - nouvelles: use_textline_orientation / device
    """
    _orig_print(
        f"[worker pid={os.getpid()}] Loading PaddleOCR model (lang={lang})...",
        file=sys.stderr,
        flush=True,
    )

    from paddleocr import PaddleOCR

    # On essaie d'abord la nouvelle API, sinon on fallback vers l'ancienne.
    try:
        model = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,  # nouvelle API (remplace use_angle_cls)
            device="cpu",                   # nouvelle API (remplace use_gpu=False)
        )
    except TypeError:
        # Fallback ancienne API
        model = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=False,
        )

    _orig_print(f"[worker pid={os.getpid()}] Ready.", file=sys.stderr, flush=True)
    return model


# ── OCR call compatible ──────────────────────────────────────────────────────

def ocr_image(model, img):
    """
    Appel PaddleOCR compatible :
    - certaines versions acceptent cls=...
    - d'autres non (erreur: predict() got unexpected keyword argument 'cls')
    """
    try:
        return model.ocr(img, cls=True)
    except TypeError:
        return model.ocr(img)


# ── OCR PDF ──────────────────────────────────────────────────────────────────

def ocr_pdf(model, pdf_path: str, dpi: int = 300):
    """Convertit un PDF en texte via PaddleOCR (page par page)."""
    import fitz  # PyMuPDF
    import numpy as np
    import cv2

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages_text = []

    # DPI -> zoom (PDF est en 72 DPI de base)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_index in range(page_count):
        page = doc.load_page(page_index)

        # Render page -> pixmap (sans alpha)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # pix.samples = bytes
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Normalize en BGR (OpenCV)
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        result = ocr_image(model, img)

        lines = []
        # PaddleOCR renvoie typiquement une liste par image
        # Structure commune: [[ [box], (text, score) ], ...]
        if result:
            for res_page in result:
                if not res_page:
                    continue
                for line in res_page:
                    try:
                        text = line[1][0].strip()
                    except Exception:
                        continue
                    if text:
                        lines.append(text)

        pages_text.append("\n".join(lines))

    doc.close()
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text).strip()
    return full_text, page_count


# ── Boucle principale ─────────────────────────────────────────────────────────

def main():
    try:
        # Mets "fr" si tes docs sont FR ; sinon "en" ou autre
        model = load_model(lang=os.getenv("OCR_LANG", "fr"))
    except Exception as e:
        emit({"ready": False, "error": f"Model load failed: {e}"})
        sys.exit(1)

    emit({"ready": True})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        req_id = None
        try:
            req = json.loads(raw_line)
            req_id = req.get("id")
            pdf_path = req["pdf_path"]

            # Optionnel: dpi configurable depuis la requête (fallback 300)
            dpi = int(req.get("dpi", 300))

            text, page_count = ocr_pdf(model, pdf_path, dpi=dpi)
            emit({"id": req_id, "text": text, "page_count": page_count})

        except json.JSONDecodeError as e:
            emit({"id": req_id, "error": f"Invalid JSON: {e}"})
        except KeyError as e:
            emit({"id": req_id, "error": f"Missing field: {e}"})
        except Exception as e:
            _orig_print(traceback.format_exc(), file=sys.stderr, flush=True)
            emit({"id": req_id, "error": str(e)})


if __name__ == "__main__":
    main()