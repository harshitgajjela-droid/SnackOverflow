import cv2
import numpy as np


# ==========================================
# 1. LOAD IMAGE
# ==========================================

def load_image(image_path):
    """
    Load a product image using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    return image


# ==========================================
# 2. IMAGE QUALITY CHECK
# ==========================================

def check_image_quality(image):
    """
    Check basic image quality for OCR.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Get dimensions
    height, width = image.shape[:2]

    # Blur / sharpness score
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Average brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = np.std(gray)

    return {
        "width": width,
        "height": height,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2)
    }


# ==========================================
# 3. IMAGE QUALITY ASSESSMENT
# ==========================================

def assess_image_quality(quality):
    """
    Interpret image quality values and decide
    whether the image needs preprocessing.
    """

    issues = []
    recommendations = []

    # Resolution check
    width = quality["width"]
    height = quality["height"]

    if width < 600 or height < 300:
        issues.append("Low resolution")
        recommendations.append("Upscale image before OCR")

    # Blur check
    blur_score = quality["blur_score"]

    if blur_score < 100:
        issues.append("Image is blurry")
        recommendations.append("Apply sharpening")

    # Brightness check
    brightness = quality["brightness"]

    if brightness < 60:
        issues.append("Image is too dark")
        recommendations.append(
            "Increase brightness and contrast"
        )

    elif brightness > 210:
        issues.append("Image is too bright")
        recommendations.append(
            "Reduce brightness"
        )

    # Contrast check
    contrast = quality["contrast"]

    if contrast < 25:
        issues.append("Low contrast")
        recommendations.append(
            "Apply CLAHE contrast enhancement"
        )

    # Final status
    if len(issues) == 0:
        status = "GOOD"
        suitable_for_ocr = True
    else:
        status = "NEEDS PROCESSING"
        suitable_for_ocr = False

    return {
        "status": status,
        "suitable_for_ocr": suitable_for_ocr,
        "issues": issues,
        "recommendations": recommendations
    }


# ==========================================
# 4. RESIZE / UPSCALE IMAGE
# ==========================================

def resize_image(image, target_width=2400):
    """
    Upscale product label images for better OCR
    while maintaining the original aspect ratio.
    """

    height, width = image.shape[:2]

    # Do not enlarge if already large enough
    if width >= target_width:
        return image

    scale = target_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    return resized


# ==========================================
# 5. DESKEW IMAGE
# ==========================================

def deskew_image(image):
    """
    Detect and correct dominant text rotation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Detect foreground text regions
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Get foreground coordinates
    coords = np.column_stack(
        np.where(binary > 0)
    )

    # Safety check
    if len(coords) < 100:
        return image, 0

    # Find dominant angle
    angle = cv2.minAreaRect(coords)[-1]

    # Convert OpenCV angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore tiny rotations
    if abs(angle) < 0.5:
        return image, 0

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    corrected = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected, angle


# ==========================================
# 6. TEXT ENHANCEMENT
# ==========================================

def enhance_text(image):
    """
    Basic OpenCV enhancement for OCR.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Light denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast_enhanced = clahe.apply(
        denoised
    )

    # Sharpen text
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        contrast_enhanced,
        -1,
        kernel
    )

    return sharpened


# ==========================================
# 7. GENERATE MULTIPLE OCR VERSIONS
# ==========================================

def generate_ocr_versions(image):
    """
    Generate multiple OpenCV preprocessing versions.

    Different product labels may work better
    with different preprocessing techniques.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Light denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    versions = {}

    # --------------------------------------
    # VERSION 1: CLAHE + SHARPENING
    # --------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    versions["enhanced"] = sharpened


    # --------------------------------------
    # VERSION 2: OTSU THRESHOLD
    # --------------------------------------

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    versions["otsu"] = otsu


    # --------------------------------------
    # VERSION 3: ADAPTIVE THRESHOLD
    # --------------------------------------

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        9
    )

    versions["adaptive"] = adaptive


    # --------------------------------------
    # VERSION 4: INVERTED ADAPTIVE
    # Useful for some difficult label images
    # --------------------------------------

    inverted = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9
    )

    versions["inverted"] = inverted

    return versions
def detect_text_regions(image):
    """
    Detect possible text regions using OpenCV.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 5)
    )

    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < 40 or h < 15:
            continue

        if w > image.shape[1] * 0.95:
            continue

        regions.append((x, y, w, h))

    regions = sorted(
        regions,
        key=lambda box: (box[1], box[0])
    )

    return regions