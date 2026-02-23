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

def load_model():
    _orig_print(f"[worker pid={os.getpid()}] Loading PaddleOCR model...", file=sys.stderr, flush=True)
    from paddleocr import PaddleOCR
    model = PaddleOCR(
        use_textline_orientation=True,  # remplace use_angle_cls
        device="cpu",                   # remplace use_gpu=False
    )
    _orig_print(f"[worker pid={os.getpid()}] Ready.", file=sys.stderr, flush=True)
    return model

# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr_pdf(model, pdf_path):
    """Convertit un PDF en texte via PaddleOCR (page par page)."""
    import fitz  # PyMuPDF — pour rasteriser chaque page du PDF

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages_text = []

    for page in doc:
        # Rasteriser la page en image (300 DPI)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")

        # PaddleOCR accepte directement des bytes d'image
        result = model.ocr(img_bytes)

        lines = []
        if result:
            for res_page in result:
                if res_page is None:
                    continue
                for line in res_page:
                    # line = [[coords], [text, confidence]]
                    text = line[1][0].strip()
                    if text:
                        lines.append(text)

        pages_text.append("\n".join(lines))

    doc.close()
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text).strip()
    return full_text, page_count

# ── Boucle principale ─────────────────────────────────────────────────────────

def main():
    try:
        model = load_model()
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
            text, page_count = ocr_pdf(model, req["pdf_path"])
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