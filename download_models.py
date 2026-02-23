#!/usr/bin/env python3
"""Pre-download ALL PaddleOCR models used at runtime."""
import os

# Réduit la verbosité de Paddle au build
os.environ["FLAGS_call_stack_level"] = "2"

from paddleocr import PaddleOCR

print("Downloading PaddleOCR models (detection + recognition + angle classifier)...")
# Déclenche le téléchargement de tous les modèles nécessaires
PaddleOCR(
    use_angle_cls=True,
    use_gpu=False,
    show_log=False,
    lang="fr",          # modèle multilingue Europe occidentale (latin script)
)
print("All PaddleOCR models downloaded successfully.")
