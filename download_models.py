#!/usr/bin/env python3
"""
Pre-warms all PaddleOCR / PaddlePaddle models at Docker BUILD time.
Run via: python3 /app/download_models.py

This ensures that ALL models — including the extra PaddleX models that
paddleocr==3.4.0 downloads on first use (PP-LCNet_x1_0_doc_ori, UVDoc, etc.)
— are baked into the image and not fetched at container startup.
"""
import os
import sys
from pathlib import Path

# Désactive OneDNN : PP-OCRv5 server crash à l'inférence sinon
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ─── Model dirs ───────────────────────────────────────────────────────────────
DET_DIR = os.environ.get("PPOCR_DET_DIR", "/models/ppocrv5/det")
REC_DIR = os.environ.get("PPOCR_REC_DIR", "/models/ppocrv5/rec")
CLS_DIR = os.environ.get("PPOCR_CLS_DIR", "/models/ppocrv5/cls")

# ─── Sanity-check baked model dirs ────────────────────────────────────────────
def check_dir(p: str) -> None:
    path = Path(p)
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing: {p}")
    if not any(path.iterdir()):
        raise SystemExit(f"[ERROR] Empty directory: {p}")
    print(f"[OK] {p}", flush=True)

print("=== Checking baked PP-OCRv5 model dirs ===", flush=True)
check_dir(DET_DIR)
check_dir(REC_DIR)
check_dir(CLS_DIR)

# ─── Init PaddleOCR with the NEW 3.4.0 API ────────────────────────────────────
# This triggers ALL lazy model downloads (UVDoc, PP-LCNet_x1_0_doc_ori, etc.)
# so they end up in $PADDLEX_HOME and are copied into the runtime image.
print("=== Initializing PaddleOCR (will download auxiliary models) ===", flush=True)
try:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="fr",
        device="cpu",
        use_textline_orientation=True,
        text_detection_model_dir=DET_DIR,
        text_recognition_model_dir=REC_DIR,
        textline_orientation_model_dir=CLS_DIR,
    )
    print("[OK] PaddleOCR initialized with new API.", flush=True)

except TypeError as e:
    # Fallback: old API (should not happen with paddleocr==3.4.0)
    print(f"[WARN] New API failed ({e}), falling back to legacy API.", flush=True)
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang="fr",
        use_angle_cls=True,
        use_gpu=False,
        det_model_dir=DET_DIR,
        rec_model_dir=REC_DIR,
        cls_model_dir=CLS_DIR,
    )
    print("[OK] PaddleOCR initialized with legacy API.", flush=True)

except Exception as e:
    print(f"[ERROR] PaddleOCR init failed: {e}", flush=True)
    sys.exit(1)

print("=== All models ready. Build cache is warm. ===", flush=True)