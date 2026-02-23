#!/usr/bin/env python3
"""
OCR worker persistant — lit des requêtes JSON sur stdin, écrit des réponses JSON sur stdout.
Modèle PaddleOCR chargé UNE SEULE FOIS au démarrage du process.

Protocole :
  stdin  → {"id": "abc", "pdf_path": "/tmp/input.pdf"}
  stdout → {"id": "abc", "text": "...", "page_count": N}
         | {"id": "abc", "error": "message"}
  stdout → {"ready": true}  (au démarrage, une seule fois)

Options (env) :
  OCR_LANG        : "fr" (défaut) / "en" / ...
  OCR_DPI         : 300 (défaut)
  PPOCR_DET_DIR   : /models/ppocrv5/det (défaut)
  PPOCR_REC_DIR   : /models/ppocrv5/rec (défaut)
  PPOCR_CLS_DIR   : /models/ppocrv5/cls (défaut)
  PADDLEX_HOME    : /root/.paddlex (défaut PaddleX — override si non-root)
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
# Désactive OneDNN/MKL-DNN : PP-OCRv5 server crash avec
# "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"
os.environ.setdefault("FLAGS_use_mkldnn", "0")
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
    - paddleocr >= 3.4.0 : text_detection_model_dir / text_recognition_model_dir /
                           textline_orientation_model_dir / use_textline_orientation / device
    - paddleocr < 3.4.0  : det_model_dir / rec_model_dir / cls_model_dir /
                           use_angle_cls / use_gpu

    Si des modèles custom (PP-OCRv5) sont présents dans /models/ppocrv5,
    on force les chemins pour éviter tout téléchargement à l'exécution.
    """
    _orig_print(
        f"[worker pid={os.getpid()}] Loading PaddleOCR model (lang={lang})...",
        file=sys.stderr,
        flush=True,
    )

    from paddleocr import PaddleOCR

    det_dir = os.getenv("PPOCR_DET_DIR", "/models/ppocrv5/det")
    rec_dir = os.getenv("PPOCR_REC_DIR", "/models/ppocrv5/rec")
    cls_dir = os.getenv("PPOCR_CLS_DIR", "/models/ppocrv5/cls")

    have_custom = all(os.path.isdir(p) for p in (det_dir, rec_dir, cls_dir))

    if have_custom:
        _orig_print(
            f"[worker pid={os.getpid()}] Using custom PP-OCR model dirs:\n"
            f"  det={det_dir}\n  rec={rec_dir}\n  cls={cls_dir}",
            file=sys.stderr,
            flush=True,
        )
    else:
        _orig_print(
            f"[worker pid={os.getpid()}] No custom model dirs found; using PaddleOCR defaults.",
            file=sys.stderr,
            flush=True,
        )

    # ── Essai prioritaire : nouvelle API paddleocr >= 3.4.0 ──────────────────
    try:
        kwargs = dict(lang=lang, use_textline_orientation=True, device="cpu")
        if have_custom:
            kwargs.update(
                text_detection_model_dir=det_dir,
                text_recognition_model_dir=rec_dir,
                textline_orientation_model_dir=cls_dir,
            )
        model = PaddleOCR(**kwargs)
        _orig_print(f"[worker pid={os.getpid()}] Ready (new API).", file=sys.stderr, flush=True)
        return model
    except TypeError:
        pass  # paramètres inconnus → on tente l'ancienne API

    # ── Fallback : ancienne API paddleocr < 3.4.0 ────────────────────────────
    kwargs = dict(lang=lang, use_angle_cls=True, use_gpu=False)
    if have_custom:
        kwargs.update(
            det_model_dir=det_dir,
            rec_model_dir=rec_dir,
            cls_model_dir=cls_dir,
        )
    model = PaddleOCR(**kwargs)
    _orig_print(f"[worker pid={os.getpid()}] Ready (legacy API).", file=sys.stderr, flush=True)
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

        # IMPORTANT:
        # - pix.get_pixmap(alpha=False) renvoie généralement du RGB (n==3)
        # - PaddleOCR accepte des numpy arrays.
        # Si tu veux absolument du BGR (sans OpenCV), tu peux inverser les canaux :
        # img = img[:, :, ::-1]
        #
        # Ici on garde RGB pour éviter OpenCV (souvent source de crash quand les wheels opencv se mélangent).
        if pix.n == 4:
            # Sécurité si jamais (devrait être rare avec alpha=False)
            img = img[:, :, :3]

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

            # DPI configurable depuis la requête (fallback env OCR_DPI puis 300)
            dpi = int(req.get("dpi", os.getenv("OCR_DPI", 300)))

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