# Install if needed
!pip install pillow pytesseract

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Patient data ----------
cbc_data = {
    "Age": 33,
    "Sex": "Male",
    "Hb": 12.4,
    "Hct": 40.7,
    "MCV": 75.4,
    "MCH": 23.0,
    "MCHC": 30.5,
    "RDW": 14.7,
    "RBC count": 5.4
}

# ---------- Step 2: Generate CBC report image ----------
def generate_cbc_image(data, filename="cbc_report.png"):
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("LiberationMono-Regular.ttf", 32)
    except:
        font = ImageFont.load_default()

    y = 30
    draw.text((20, y), "CBC Blood Test Report", fill="black", font=font)
    y += 50
    draw.text((20, y), f"Patient Information:", fill="black", font=font)
    y += 40
    draw.text((20, y), f"Age: {data['Age']}   Sex: {data['Sex']}", fill="black", font=font)
    y += 60
    draw.text((20, y), "Test Results:", fill="black", font=font)
    y += 40

    for key, value in data.items():
        if key not in ["Age", "Sex"]:
            draw.text((20, y), f"{key}: {value}", fill="black", font=font)
            y += 40

    image.save(filename)
    print(f"Report image saved as {filename}")
    return image

report_img = generate_cbc_image(cbc_data)

# ---------- Step 3: Preprocess image for OCR ----------
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.bitwise_not(img_bin)
    preprocessed_path = "cbc_report_preprocessed.png"
    cv2.imwrite(preprocessed_path, img_inv)
    print(f"Preprocessed image saved as {preprocessed_path}")
    return preprocessed_path

preprocessed_img_path = preprocess_image_for_ocr("cbc_report.png")

# ---------- Step 4: Extract text with OCR ----------
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print("\n--- OCR Extracted Text ---\n")
    print(text)
    return text

ocr_text = extract_text_from_image(preprocessed_img_path)

# ---------- Step 5: Parse features from OCR text ----------
def parse_cbc_features(text):
    features = {
        "Age": None, "Sex": None, "Hb": None, "Hct": None,
        "MCV": None, "MCH": None, "MCHC": None,
        "RDW": None, "RBC count": None
    }

    # Age and Sex
    age_sex = re.search(r"Age[:\s]+(\d+)\s+Sex[:\s]+(\w+)", text, re.IGNORECASE)
    if age_sex:
        features["Age"] = int(age_sex.group(1))
        features["Sex"] = age_sex.group(2)

    # Extract values for remaining features
    for key in ["Hb", "Hct", "MCV", "MCH", "MCHC", "RDW", "RBC count"]:
        pattern = re.compile(rf"{key}[:\s]+([\d.]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            features[key] = float(match.group(1))

    return features

parsed_features = parse_cbc_features(ocr_text)

print("\n--- Parsed CBC Features ---\n")
for k, v in parsed_features.items():
    print(f"{k}: {v}")
