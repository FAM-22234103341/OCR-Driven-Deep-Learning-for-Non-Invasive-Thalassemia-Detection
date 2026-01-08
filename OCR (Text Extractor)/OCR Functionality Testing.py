# Example usage with external report
external_img_path = "/content/cbc_report.png"  # path to your uploaded image

# Optional: Preprocess for better OCR (you can skip if image is already clean)
preprocessed_img_path = preprocess_image_for_ocr(external_img_path)

# OCR
ocr_text = extract_text_from_image(preprocessed_img_path)

# Feature Parsing
parsed_features = parse_cbc_features(ocr_text)

print("\n--- Parsed CBC Features from Uploaded Report ---\n")
for k, v in parsed_features.items():
    print(f"{k}: {v}")
