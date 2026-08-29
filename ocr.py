import pytesseract

# Point to your local Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def run_ocr_with_psm(image, psm):

    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=config
    )

    detections = []
    extracted_words = []
    confidence_values = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if text != "" and confidence >= 0:

            detection = {
                "text": text,
                "confidence": round(confidence, 2),
                "bbox": {
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i])
                }
            }

            detections.append(detection)
            extracted_words.append(text)
            confidence_values.append(confidence)

    if confidence_values:
        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        average_confidence = 0

    return {
        "psm": psm,
        "text": " ".join(extracted_words),
        "detections": detections,
        "average_confidence": round(
            average_confidence,
            2
        )
    }


def run_ocr(image):
    """
    Try multiple PSM modes and select the most useful
    OCR result using confidence + text quantity.
    """

    psm_modes = [3, 6, 11, 12]

    all_results = []

    for psm in psm_modes:

        result = run_ocr_with_psm(image, psm)

        word_count = len(result["detections"])

        total_characters = sum(
            len(detection["text"])
            for detection in result["detections"]
        )

        average_confidence = result["average_confidence"]

        # Combined OCR quality score
        #
        # Confidence is important, but results with more
        # useful text should also receive a better score.

        score = (
            average_confidence * 0.6
            + min(word_count, 50) * 0.5
            + min(total_characters, 300) * 0.1
        )

        # Strong penalty if almost no text was detected
        if word_count < 5:
            score -= 30

        result["word_count"] = word_count
        result["total_characters"] = total_characters
        result["score"] = round(score, 2)

        all_results.append(result)

    best_result = max(
        all_results,
        key=lambda result: result["score"]
    )

    return best_result
import cv2


def draw_ocr_boxes(image, detections):
    """
    Draw OCR bounding boxes and confidence
    on the image using OpenCV.
    """

    # Make a copy so original image is unchanged
    output_image = image.copy()

    for detection in detections:

        bbox = detection["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        text = detection["text"]
        confidence = detection["confidence"]

        # Ignore very low confidence detections
        if confidence < 30:
            continue

        # Draw bounding box
        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Create label
        label = f"{text} ({confidence:.0f}%)"

        # Draw label
        cv2.putText(
            output_image,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return output_image
