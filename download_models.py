import os
from pathlib import Path

def check_dir(p: str) -> None:
    path = Path(p)
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing: {p}")
    if not any(path.iterdir()):
        raise SystemExit(f"[ERROR] Empty directory: {p}")
    print(f"[OK] {p}")

def main():
    det = os.environ.get("PPOCR_DET_DIR", "/models/ppocrv5/det")
    rec = os.environ.get("PPOCR_REC_DIR", "/models/ppocrv5/rec")
    cls = os.environ.get("PPOCR_CLS_DIR", "/models/ppocrv5/cls")

    check_dir(det)
    check_dir(rec)
    check_dir(cls)

    # Import optionnel (à éviter pendant le build si tu as des soucis OpenCV)
    # Ici c'est juste un exemple d'init PaddleOCR en pointant sur tes modèles.
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_angle_cls=True,
            det_model_dir=det,
            rec_model_dir=rec,
            cls_model_dir=cls,
            lang="en",     # adapte à ton cas
            use_gpu=False,
        )
        print("[OK] PaddleOCR init done (PP-OCRv5 paths).")
        # Tu peux faire un test:
        # res = ocr.ocr("test.jpg")
        # print(res)
    except Exception as e:
        print("[WARN] PaddleOCR import/init failed (this is OK if you only wanted model download).")
        print("Reason:", repr(e))

if __name__ == "__main__":
    main()