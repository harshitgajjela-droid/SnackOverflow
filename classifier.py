import os
import re

MODEL_PATH = "models/legal_vision/weights/best.pt"
vision_model = None

# Only attempt to load the YOLO model if the trained weights file exists
if os.path.exists(MODEL_PATH):
    try:
        from ultralytics import YOLO
        vision_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load vision model ({e}). Falling back to text heuristics.")

def get_product_classification(ocr_text: str, image_path: str = None) -> dict:
    """Classifies package attributes using vision detections (if available) and OCR text heuristics."""
    text = ocr_text.lower()
    detected_classes = []

    # Run vision detection only if weights exist and an image path is passed
    if vision_model and image_path and os.path.exists(image_path):
        try:
            results = vision_model(image_path, conf=0.5, verbose=False)
            detected_classes = [vision_model.names[int(box.cls)] for box in results[0].boxes]
        except Exception:
            detected_classes = []

    has_fssai = "fssai_logo" in detected_classes
    has_veg_mark = "veg_dot" in detected_classes or "non_veg_dot" in detected_classes

    # Classify as food via vision symbols or OCR text markers
    is_food = has_fssai or has_veg_mark or bool(re.search(r'\bfssai\b|nutritional|ingredients|protein|kcal', text))

    return {
        "package_scope": "retail_prepackaged",
        "is_food": is_food,
        "is_imported": bool(re.search(r'\bimported\s+by\b|country\s+of\s+origin', text)),
        "manufacturer_is_packer": bool(re.search(r'manufactured\s+(?:and|&)\s+packed', text)),
        "may_become_unfit_for_human_consumption": is_food,
        "dimensions_relevant": bool(re.search(r'\b(?:size|dimensions)\b|\d+\s*(?:cm|mm|inch)\b', text)),
        "declarations_legible_and_prominent": True
    }
    