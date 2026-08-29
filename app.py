import cv2
import os
from pipeline import analyze_ocr
from modules.classifier import get_product_classification

from modules.image_processing import (
    load_image,
    check_image_quality,
    assess_image_quality,
    resize_image,
    deskew_image,
    generate_ocr_versions,
    detect_text_regions
)

from modules.ocr import (
    run_ocr,
    draw_ocr_boxes
)

# ==========================================
# 21. RUN LEGAL METROLOGY COMPLIANCE ENGINE
# ==========================================

print("\n========================================")
print("       RUNNING COMPLIANCE ENGINE")
import os
import cv2

from modules.image_processing import (
    load_image,
    check_image_quality,
    assess_image_quality,
    resize_image,
    deskew_image,
    generate_ocr_versions,
    detect_text_regions
)

from modules.ocr import (
    run_ocr,
    draw_ocr_boxes
)

from modules.classifier import get_product_classification
from pipeline import analyze_ocr


# ==========================================
# 1. GET PRODUCT IMAGE FROM USER
# ==========================================

print("\n========================================")
print(" PRODUCT LABEL OCR + COMPLIANCE SYSTEM")
print("========================================\n")

image_path = input("Enter product image path: ").strip()


# ==========================================
# 2. CHECK IMAGE EXISTS
# ==========================================

if not os.path.exists(image_path):
    print("\nERROR: Image file not found!")
    print("\nExample: sample_images/p2.jpg")
    exit()


# ==========================================
# 3. LOAD IMAGE
# ==========================================

image = load_image(image_path)
print("\n========== PRODUCT IMAGE LOADED ==========\n")
print(f"Image Path: {image_path}")


# ==========================================
# 4. IMAGE QUALITY CHECK
# ==========================================

quality = check_image_quality(image)
print("\n========== ORIGINAL IMAGE DETAILS ==========\n")
for key, value in quality.items():
    print(f"{key}: {value}")


# ==========================================
# 5. IMAGE QUALITY ASSESSMENT
# ==========================================

assessment = assess_image_quality(quality)
print("\n========== IMAGE QUALITY ASSESSMENT ==========\n")
print(f"Status: {assessment['status']}")
print(f"Suitable for OCR: {assessment['suitable_for_ocr']}")

if assessment["issues"]:
    print("\nIssues Found:")
    for issue in assessment["issues"]:
        print(f"- {issue}")

if assessment["recommendations"]:
    print("\nRecommendations:")
    for recommendation in assessment["recommendations"]:
        print(f"- {recommendation}")


# ==========================================
# 6. RESIZE / UPSCALE IMAGE
# ==========================================

resized_image = resize_image(image)
print("\n========== IMAGE RESIZE ==========\n")
print(f"Original Size: {image.shape[1]} x {image.shape[0]}")
print(f"Resized Size: {resized_image.shape[1]} x {resized_image.shape[0]}")


# ==========================================
# 7. DESKEW IMAGE
# ==========================================

deskewed_image, angle = deskew_image(resized_image)
print("\n========== DESKEW ==========\n")
print(f"Detected Rotation Angle: {angle:.2f} degrees")


# ==========================================
# 8. GENERATE OCR IMAGE VERSIONS
# ==========================================

ocr_versions = generate_ocr_versions(deskewed_image)
print("\n========== OCR IMAGE VERSIONS ==========\n")
for method in ocr_versions:
    print(f"Generated: {method}")


# ==========================================
# 9. DETECT TEXT REGIONS
# ==========================================

text_regions = detect_text_regions(ocr_versions["enhanced"])
print("\n========== TEXT REGION DETECTION ==========\n")
print(f"Possible Text Regions Found: {len(text_regions)}")
print("\nText Region Bounding Boxes:")
for index, region in enumerate(text_regions, start=1):
    x, y, w, h = region
    print(f"Region {index}: x={x}, y={y}, width={w}, height={h}")


# ==========================================
# 10. RUN OCR ON ALL IMAGE VERSIONS
# ==========================================

ocr_results = {}
print("\n========== RUNNING OCR ==========")

for method, processed_image in ocr_versions.items():
    print(f"\nRunning OCR on: {method}")
    result = run_ocr(processed_image)
    ocr_results[method] = result
    print(f"Best PSM: {result['psm']}")
    print(f"Average Confidence: {result['average_confidence']}%")
    print(f"Words Detected: {result['word_count']}")
    print(f"Total Characters: {result['total_characters']}")
    print(f"OCR Score: {result['score']}")


# ==========================================
# 11. SELECT BEST OCR RESULT
# ==========================================

best_method = max(
    ocr_results,
    key=lambda method: ocr_results[method]["score"]
)
best_result = ocr_results[best_method]


# ==========================================
# 12. DRAW OCR BOUNDING BOXES
# ==========================================

ocr_visualization = draw_ocr_boxes(
    ocr_versions[best_method],
    best_result["detections"]
)


# ==========================================
# 13. PRINT BEST OCR RESULT
# ==========================================

print("\n========================================")
print("          BEST OCR RESULT")
print("========================================\n")
print(f"Best Preprocessing Method: {best_method}")
print(f"Best PSM Mode: {best_result['psm']}")
print(f"Average Confidence: {best_result['average_confidence']}%")
print(f"Words Detected: {best_result['word_count']}")
print(f"Total Characters: {best_result['total_characters']}")
print(f"Final OCR Score: {best_result['score']}")


# ==========================================
# 14. PRINT EXTRACTED TEXT
# ==========================================

print("\n========================================")
print("          EXTRACTED TEXT")
print("========================================\n")
print(best_result["text"])


# ==========================================
# 15. PRINT ALL OCR DETECTIONS
# ==========================================

print("\n========================================")
print("    TEXT + CONFIDENCE + BOUNDING BOX")
print("========================================\n")

for index, detection in enumerate(best_result["detections"], start=1):
    print(f"Detection {index}")
    print(f"  Text: {detection['text']}")
    print(f"  Confidence: {detection['confidence']}%")
    print(f"  Bounding Box: {detection['bbox']}")
    print("-" * 40)


# ==========================================
# 16. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("output/processed", exist_ok=True)


# ==========================================
# 17. SAVE DESKEWED IMAGE
# ==========================================

deskewed_path = "output/processed/deskewed.jpg"
cv2.imwrite(deskewed_path, deskewed_image)
print(f"\nSaved: {deskewed_path}")


# ==========================================
# 18. SAVE ALL OCR VERSIONS
# ==========================================

for method, processed_image in ocr_versions.items():
    output_path = f"output/processed/{method}.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Saved: {output_path}")


# ==========================================
# 19. SAVE OCR VISUALIZATION
# ==========================================

ocr_result_path = "output/processed/ocr_result.jpg"
cv2.imwrite(ocr_result_path, ocr_visualization)
print(f"Saved OCR visualization: {ocr_result_path}")


# ==========================================
# 20. PIPELINE COMPLETE
# ==========================================

print("\n========================================")
print(" FULL OPENCV + OCR PIPELINE COMPLETE")
print("========================================\n")


# ==========================================
# 21. RUN LEGAL METROLOGY COMPLIANCE ENGINE
# ==========================================

print("\n========================================")
print("       RUNNING COMPLIANCE ENGINE")
print("========================================\n")

normalized_ocr = {
    "lines": best_result["detections"],
    "generic_name": None,
    "dimensions": None
}

classification_flags = get_product_classification(best_result["text"])

report = analyze_ocr(
    normalized_ocr,
    **classification_flags
)

print(f"Outcome: {report['outcome']}")
print(f"Rules Passed: {report['counts']['pass']}")
print(f"Needs Review: {report['counts']['needs_review']}")
print(f"Failed: {report['counts']['fail']}\n")

if report["findings"]:
    print("----- DETAILED FINDINGS -----")
    for finding in report["findings"]:
        if finding["outcome"] != "PASS":
            print(f"Rule: {finding['rule_name']} ({finding['rule_id']})")
            print(f"Status: {finding['outcome']}")
            print(f"Message: {finding['message']}")
            print(f"Fix: {finding['fix']}\n")