#!/usr/bin/env python3
"""Pre-download ALL PaddleOCR models used at runtime."""
import os

os.environ["FLAGS_call_stack_level"] = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

print("Downloading PaddleOCR models...")
PaddleOCR(
    #use_textline_orientation=True,  # remplace use_angle_cls
    #device="cpu",                   # remplace use_gpu=False
    use_angle_cls=True,
    lang="fr",   # si tu veux du FR par défaut
    use_gpu=False,
)
print("All PaddleOCR models downloaded successfully.")