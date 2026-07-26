# Image or Text Recognition (Basic)

**Project 4 — Optional Mastery Phase**
DecodeLabs Industrial Training Kit | Batch 2026

> This project is **optional**. Per the deck: completing Project 3 already
> qualifies you for certification. Project 4 exists to prove you can
> integrate pre-trained AI libraries into a working perception pipeline —
> "building the machine's optic nerve."

This kit implements **both paths** described in the training deck, so you
can pick whichever matches your interest (or submit both):

- **Path 1 — OCR** (`ocr_pipeline.py`): reads text out of an image
- **Path 2 — Object Detection** (`object_detection.py`): locates and labels
  physical objects in an image using MobileNet-SSD

---

## Why this matters (from the deck)

Structured data (spreadsheets, clean CSVs) accounts for under 20% of
enterprise data. The other 80%+ is **unstructured** — scanned documents,
photos, video. Project 4 is about building the bridge between raw pixels
and machine-readable intelligence: teaching a script to *see*.

To a machine, an image isn't a picture — it's a 3D array: **Height ×
Width × Depth (3 color channels)**, with every pixel holding an intensity
value from 0–255. A single 512×512 image is 786,432 individual numbers.
Both pipelines below exist to turn that raw number grid into something
meaningful.

---

## Files

| File                    | Purpose                                            |
|-------------------------|-----------------------------------------------------|
| `ocr_pipeline.py`       | Path 1 — text recognition via pytesseract           |
| `object_detection.py`   | Path 2 — object detection via OpenCV + MobileNet-SSD|
| `sample_images/`        | Test images (a noisy/tilted text image + a photo)   |
| `models/`               | Pre-trained MobileNet-SSD weights (auto-downloaded) |
| `output/`               | Where results and preprocessing snapshots land      |
| `requirements.txt`      | Python dependencies                                 |

---

## Path 1: OCR (`ocr_pipeline.py`)

**Engine:** `pytesseract`, a Python wrapper around Google's Tesseract
engine (a CNN + bi-directional LSTM pipeline for reading character
sequences).

### The pre-processing chain ("Logic Skeleton")

Raw images are cluttered with shadows, noise, and tilt — all of which
tank OCR accuracy. Four steps clean that up before Tesseract ever sees
the image:

1. **Grayscale conversion** — collapses the 3-channel RGB array into a
   single intensity channel; OCR doesn't need color.
2. **Gaussian blur** — smooths out micro-noise and artifacts.
3. **Deskewing** — detects the rotation angle of the text blob (via
   `cv2.minAreaRect`) and rotates it back to a perfect horizontal
   baseline.
4. **Adaptive thresholding (Otsu's method)** — forces every pixel to
   pure black or white:
   ```
   IF pixel_intensity >= cutoff THEN pixel = 255 (white)
   IF pixel_intensity <  cutoff THEN pixel = 0   (black)
   ```
   Otsu automatically calculates the optimal cutoff rather than using a
   fixed guess.

### Page Segmentation Modes (PSM)

Tesseract needs to know the shape of the text layout:

| PSM | Use case                          |
|-----|-------------------------------------|
| 3   | Fully automatic (default, varied layouts) |
| 6   | Single uniform block of text (book pages) |
| 7   | Single text line (number plates, headers) |
| 11  | Sparse, scattered text (invoices)  |

### The 80% Confidence Gate

Every recognized word carries a per-word confidence score. Words below
**80%** are dropped from the accepted output rather than silently
included — this is the "Milestone Validation" standard from the deck.

### Run it

```bash
pip install -r requirements.txt
python3 ocr_pipeline.py --image sample_images/sample_noisy_tilted.png --psm 6
```

Example output:
```
=== OCR Recognition Report ===
Deskew angle applied : 4.03°
Otsu threshold cutoff: 146.0

Recognized text: "DecodeLabs Al Project 4"
Average confidence: 93.0%
80% Confidence Gate: PASSED
```

All four preprocessing stages are saved as separate images in
`output/preprocessing_steps/` so you can visually confirm each step
worked (this satisfies the "Visual Confirmation" requirement).

Use `--image path/to/your/photo.png` to run it on your own scanned
document, sign, or screenshot.

---

## Path 2: Object Detection (`object_detection.py`)

**Engine:** OpenCV's `cv2.dnn` module running a pre-trained
**MobileNet-SSD** (Single Shot Detector) model.

### Transfer Learning

Rather than training a neural network from scratch, this pipeline
downloads a model that has already learned universal visual concepts
(edges, shapes, gradients) from millions of ImageNet images, then reuses
that knowledge for detection — "inheriting the machine's knowledge."

The model files are already included in `models/`:
- `MobileNetSSD_deploy.prototxt` (network architecture)
- `MobileNetSSD_deploy.caffemodel` (trained weights, ~23MB)

It recognizes 20 object classes (from the Pascal VOC dataset): person,
car, dog, cat, bicycle, bus, chair, and more.

### The pipeline

1. **Blob construction** (`cv2.dnn.blobFromImage`) — resizes the image
   to the required 300×300 input, applies mean subtraction, and scales
   pixel values. This is the "pre-processing" step for the network.
2. **Single forward pass** — unlike older multi-pass detectors, an SSD
   locates *and* classifies every object in one shot, trading a sliver
   of accuracy for real-time speed (why it's the standard for edge
   devices).
3. **Decoding the output** — the network outputs *normalized*
   coordinates (0–1), which get multiplied by the image's actual pixel
   width/height to produce a real bounding box.
4. **The 80% confidence gate** — same rule as Path 1. High thresholds
   minimize false positives (confident hallucinations) at the cost of
   some missed detections; 80% is the deck's minimum standard.

### Run it

```bash
python3 object_detection.py --image sample_images/astronaut.png
```

Example output:
```
=== Object Detection Report ===
Objects passing the 80% confidence gate: 1
  - person: 100.0%  box=(5, 7, 357, 509)

Annotated image saved to: output/detection_result.png
```

The annotated image shows a labeled bounding box drawn directly on the
photo. Use `--image path/to/your/photo.png` to try it on your own image
(works best with people, vehicles, or animals — see the VOC class list
in the script).

---

## The Gatekeeper Rule: Milestone Validation

Per the deck, completing Project 4 requires passing four checks. Here's
how this kit satisfies each one:

| # | Requirement | How it's satisfied |
|---|---|---|
| 1 | **Library Integration** — seamless use of `pytesseract` or `cv2.dnn` | Both are used, in `ocr_pipeline.py` and `object_detection.py` respectively |
| 2 | **Pre-Processing Integrity** — grayscale + adaptive thresholding | `ocr_pipeline.py` implements both explicitly (`to_grayscale`, `adaptive_threshold`), plus blur and deskew |
| 3 | **Accuracy Benchmarking** — ≥80% confidence on the final output | Both scripts hard-enforce `CONFIDENCE_THRESHOLD = 80%` and report pass/fail |
| 4 | **Visual Confirmation** — legible OCR text or accurate labeled bounding boxes | OCR prints readable text + saves each preprocessing stage; detection draws labeled boxes on the output image |

---

## Choosing your path

| | Path 1: OCR | Path 2: Object Detection |
|---|---|---|
| **Best for** | Documents, signs, screenshots, printed/typed text | Photos containing people, vehicles, animals, everyday objects |
| **Core library** | `pytesseract` | `cv2.dnn` + MobileNet-SSD |
| **Output** | A text string | Bounding boxes + labels |

You can run either one independently, or both, against the same image —
try feeding a photo with a street sign into both scripts to see the
contrast in what each is built to detect.
