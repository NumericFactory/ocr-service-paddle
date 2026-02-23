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
    import fitz  # PyMuPDF
    import numpy as np
    import cv2

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages_text = []

    # 300 DPI
    mat = fitz.Matrix(300 / 72, 300 / 72)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False => fond opaque
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # PyMuPDF -> souvent RGB ; Paddle/OpenCV manipulent souvent BGR
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        result = model.ocr(img, cls=True)

        lines = []
        if result:
            for res_page in result:
                if not res_page:
                    continue
                for line in res_page:
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