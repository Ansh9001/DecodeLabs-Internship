"""
Project 4 - Image or Text Recognition (Basic)
Path 2: Object Detection with MobileNet-SSD
DecodeLabs Industrial Training Kit

Pipeline (per the training deck):
    INPUT   -> raw image
    PROCESS -> Blob construction (cv2.dnn.blobFromImage: mean subtraction +
               resize to 300x300) -> Single Shot Detector forward pass
               (MobileNet v3 backbone, via Transfer Learning off ImageNet)
    OUTPUT  -> (X, Y, W, H) bounding boxes + class labels + confidence
               scores, filtered by the 80% confidence gate

This is a "Single Shot Detector" (SSD): unlike older multi-pass detectors,
it locates and classifies every object in a single forward pass through
the network — trading a little accuracy for real-time speed, which is
why it's the standard choice for edge devices.
"""

import argparse
import os

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.80  # The "80% Gate" — Project 4's minimum standard

# The 20 object classes MobileNet-SSD was trained on (VOC dataset),
# plus background as class 0.
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Reproducible distinct colors per class for the bounding-box overlay
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")


# ---------------------------------------------------------------------------
# STEP 1: MODEL LOADING (Transfer Learning — "inheriting the machine's knowledge")
# ---------------------------------------------------------------------------

def load_model(prototxt_path="models/MobileNetSSD_deploy.prototxt",
                weights_path="models/MobileNetSSD_deploy.caffemodel"):
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Model files not found. Download the pre-trained MobileNet-SSD "
            f"weights into '{os.path.dirname(prototxt_path)}/'. See README.md "
            "for the exact source URLs."
        )
    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    return net


# ---------------------------------------------------------------------------
# STEP 2: BLOB CONSTRUCTION (Pre-Processing)
# ---------------------------------------------------------------------------

def build_blob(image):
    """
    Converts the raw image into the 4D "blob" the network expects:
      - Resizes to the network's required 300x300 input dimensions
      - Applies mean subtraction (127.5) for normalization
      - Scales pixel values by 1/127.5
    """
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 127.5, size=(300, 300),
        mean=(127.5, 127.5, 127.5), swapRB=True
    )
    return blob


# ---------------------------------------------------------------------------
# STEP 3: FORWARD PASS + DECODING BOUNDING BOXES
# ---------------------------------------------------------------------------

def detect_objects(net, image):
    """
    Runs a single forward pass and decodes the network's normalized
    coordinate output into a list of detections. The model doesn't
    output an image — it outputs normalized (0-1) spatial coordinates
    that must be scaled back to the image's actual pixel dimensions.
    """
    (h, w) = image.shape[:2]
    blob = build_blob(image)
    net.setInput(blob)
    raw_detections = net.forward()

    detections = []
    for i in range(raw_detections.shape[2]):
        confidence = float(raw_detections[0, 0, i, 2])
        class_id = int(raw_detections[0, 0, i, 1])

        box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        detections.append({
            "class_id": class_id,
            "label": CLASSES[class_id] if class_id < len(CLASSES) else "unknown",
            "confidence": confidence,
            "box": (int(start_x), int(start_y), int(end_x), int(end_y)),
        })

    return detections


# ---------------------------------------------------------------------------
# STEP 4: CONFIDENCE GATE + VISUAL OUTPUT
# ---------------------------------------------------------------------------

def apply_confidence_gate(detections, threshold=CONFIDENCE_THRESHOLD):
    """
    The 80% Gate: high thresholds minimize false positives (confident
    hallucinations) at the cost of some false negatives. Project 4 sets
    80% as the absolute minimum standard.
    """
    accepted = [d for d in detections if d["confidence"] >= threshold]
    rejected = [d for d in detections if d["confidence"] < threshold]
    return accepted, rejected


def draw_detections(image, detections):
    """
    Draws bounding boxes + class label + confidence for every detection
    that passed the confidence gate — the "Visual Confirmation" required
    by the milestone rubric.
    """
    output = image.copy()
    for det in detections:
        (start_x, start_y, end_x, end_y) = det["box"]
        color = [int(c) for c in COLORS[det["class_id"]]]

        label = f"{det['label']}: {det['confidence'] * 100:.1f}%"
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), color, 2)

        text_y = start_y - 10 if start_y - 10 > 10 else start_y + 20
        cv2.putText(
            output, label, (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return output


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def run_pipeline(image_path, output_dir="output",
                  prototxt_path="models/MobileNetSSD_deploy.prototxt",
                  weights_path="models/MobileNetSSD_deploy.caffemodel"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    net = load_model(prototxt_path, weights_path)
    detections = detect_objects(net, image)
    accepted, rejected = apply_confidence_gate(detections)

    annotated = draw_detections(image, accepted)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "detection_result.png")
    cv2.imwrite(out_path, annotated)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "output_image": out_path,
        "gate_passed": len(accepted) > 0,
    }


def print_report(result):
    print("\n=== Object Detection Report ===")
    if result["accepted"]:
        print(f"Objects passing the 80% confidence gate: {len(result['accepted'])}")
        for det in result["accepted"]:
            print(f"  - {det['label']}: {det['confidence'] * 100:.1f}%  box={det['box']}")
    else:
        print("No objects passed the 80% confidence gate.")

    if result["rejected"]:
        print(f"\nDetections dropped by the confidence gate (< 80%): {len(result['rejected'])}")
        for det in result["rejected"][:5]:
            print(f"  - {det['label']}: {det['confidence'] * 100:.1f}%")

    print(f"\nAnnotated image saved to: {result['output_image']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 4 - Object Detection Pipeline")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", default="output", help="Directory to save results")
    parser.add_argument("--prototxt", default="models/MobileNetSSD_deploy.prototxt")
    parser.add_argument("--weights", default="models/MobileNetSSD_deploy.caffemodel")
    args = parser.parse_args()

    result = run_pipeline(args.image, args.output, args.prototxt, args.weights)
    print_report(result)
