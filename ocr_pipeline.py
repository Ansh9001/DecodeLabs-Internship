"""
Project 4 - Image or Text Recognition (Basic)
Path 1: Optical Character Recognition (OCR)
DecodeLabs Industrial Training Kit

Pipeline (per the training deck):
    INPUT   -> raw image (possibly noisy, tilted, poorly lit)
    PROCESS -> Grayscale -> Gaussian Blur -> Deskew -> Adaptive Threshold (Otsu)
               -> pytesseract (Tesseract OCR engine)
    OUTPUT  -> recognized text + per-word confidence scores,
               filtered by the 80% confidence gate (Milestone Validation Rule)

Engine used: pytesseract (Python wrapper for Google's Tesseract, which
internally runs a CNN + bidirectional LSTM pipeline to read character
sequences).
"""

import argparse
import os

import cv2
import numpy as np
import pytesseract

CONFIDENCE_THRESHOLD = 80.0  # The "80% Gate" — Project 4's minimum standard


# ---------------------------------------------------------------------------
# STEP 1: PRE-PROCESSING ("The Logic Skeleton")
# ---------------------------------------------------------------------------

def to_grayscale(image):
    """
    Collapses the 3D RGB matrix (H x W x 3) into a 1D intensity matrix
    (H x W). Removes distracting color data the OCR engine doesn't need.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray_image, kernel_size=(5, 5)):
    """
    Smooths the image to eliminate micro-imperfections and sensor/artifact
    noise before we make hard black/white decisions on each pixel.
    """
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def deskew(gray_image):
    """
    Calculates the rotation angle of the text blob and rotates the image
    back to a perfect horizontal baseline. Tesseract's accuracy drops
    sharply on tilted text lines.
    """
    # Invert + threshold so text pixels are white (255) on a black background,
    # which is what minAreaRect needs to find the text's bounding orientation.
    inverted = cv2.bitwise_not(gray_image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return gray_image, 0.0  # nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in a quirky range; normalize to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def adaptive_threshold(gray_image):
    """
    Forces every pixel to choose a side: pure black or pure white.
    Uses Otsu's method to automatically calculate the optimal cutoff
    intensity, rather than a fixed guess:

        IF pixel_intensity >= cutoff THEN pixel = 255 (White)
        IF pixel_intensity <  cutoff THEN pixel = 0   (Black)
    """
    cutoff, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, cutoff


def preprocess_image(image, save_steps_dir=None):
    """
    Runs the full pre-processing chain and (optionally) saves each
    intermediate step for visual confirmation / debugging.
    """
    steps = {}

    gray = to_grayscale(image)
    steps["1_grayscale"] = gray

    blurred = gaussian_blur(gray)
    steps["2_gaussian_blur"] = blurred

    deskewed, angle = deskew(blurred)
    steps["3_deskewed"] = deskewed

    binary, cutoff = adaptive_threshold(deskewed)
    steps["4_adaptive_threshold"] = binary

    if save_steps_dir:
        os.makedirs(save_steps_dir, exist_ok=True)
        for name, step_img in steps.items():
            cv2.imwrite(os.path.join(save_steps_dir, f"{name}.png"), step_img)

    return binary, {"deskew_angle": round(float(angle), 2), "otsu_cutoff": round(float(cutoff), 1)}


# ---------------------------------------------------------------------------
# STEP 2: RECOGNITION + CONFIDENCE FILTERING
# ---------------------------------------------------------------------------

def run_ocr(binary_image, psm=6):
    """
    Runs pytesseract with a configurable Page Segmentation Mode (PSM):

        --psm 3  : Fully automatic (default, varied layouts)
        --psm 6  : Single uniform block of text (book pages / signage)
        --psm 7  : Single text line (number plates / headers)
        --psm 11 : Sparse, scattered text (invoices)

    Returns word-level text + confidence via image_to_data, which is
    what lets us apply the 80% confidence gate per word.
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        binary_image, config=config, output_type=pytesseract.Output.DICT
    )
    return data


def apply_confidence_gate(data, threshold=CONFIDENCE_THRESHOLD):
    """
    The 80% Gate: without a filter, the model treats every guess with
    equal certainty, leading to confident hallucinations. Any word
    below the threshold is dropped from the accepted output.
    """
    accepted_words = []
    rejected_words = []

    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:  # pytesseract uses -1 for non-text regions
            continue
        if conf >= threshold:
            accepted_words.append((word, conf))
        else:
            rejected_words.append((word, conf))

    return accepted_words, rejected_words


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def recognize_text(image_path, psm=6, output_dir="output", save_steps=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    steps_dir = os.path.join(output_dir, "preprocessing_steps") if save_steps else None
    binary_image, meta = preprocess_image(image, save_steps_dir=steps_dir)

    data = run_ocr(binary_image, psm=psm)
    accepted, rejected = apply_confidence_gate(data)

    full_text = " ".join(word for word, _ in accepted)
    avg_conf = round(sum(c for _, c in accepted) / len(accepted), 1) if accepted else 0.0

    result = {
        "recognized_text": full_text,
        "average_confidence": avg_conf,
        "accepted_words": accepted,
        "rejected_words": rejected,
        "preprocessing_meta": meta,
        "gate_passed": avg_conf >= CONFIDENCE_THRESHOLD,
    }

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "ocr_final_binary.png"), binary_image)

    return result


def print_report(result):
    print("\n=== OCR Recognition Report ===")
    print(f"Deskew angle applied : {result['preprocessing_meta']['deskew_angle']}°")
    print(f"Otsu threshold cutoff: {result['preprocessing_meta']['otsu_cutoff']}")
    print(f"\nRecognized text: \"{result['recognized_text']}\"")
    print(f"Average confidence: {result['average_confidence']}%")
    gate = "PASSED" if result["gate_passed"] else "FAILED"
    print(f"80% Confidence Gate: {gate}")

    if result["rejected_words"]:
        print("\nWords dropped by the confidence gate (< 80%):")
        for word, conf in result["rejected_words"]:
            print(f"  - '{word}' ({conf}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 4 - OCR Recognition Pipeline")
    parser.add_argument(
        "--image", default="sample_images/sample_noisy_tilted.png",
        help="Path to the input image"
    )
    parser.add_argument("--psm", type=int, default=6, help="Tesseract Page Segmentation Mode")
    parser.add_argument("--output", default="output", help="Directory to save results")
    args = parser.parse_args()

    result = recognize_text(args.image, psm=args.psm, output_dir=args.output)
    print_report(result)
    print(f"\nPreprocessing step images + final binary saved to: {args.output}/")
