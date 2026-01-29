from PIL import Image
import pytesseract
import os

def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError("Bill file not found")

    # Image-based OCR
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()

    # PDF handling (basic version: first page image)
    if file_path.lower().endswith(".pdf"):
        raise NotImplementedError("PDF OCR will be added next step")

    raise ValueError("Unsupported file format")
