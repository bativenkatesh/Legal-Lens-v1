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

    # PDF handling (converts each page to an image and runs OCR)
    if file_path.lower().endswith(".pdf"):
        from pdf2image import convert_from_path
        
        try:
            # Increase DPI for better OCR quality
            images = convert_from_path(file_path, dpi=300)
            full_text = ""
            for img in images:
                # Convert to grayscale for better Tesseract performance
                img = img.convert('L')
                # Use PSM 6: Assume a single uniform block of text. Often better for invoices.
                page_text = pytesseract.image_to_string(img, config='--psm 6')
                full_text += page_text + "\n"
            return full_text.strip()
        except Exception as e:
            print(f"PDF OCR Error: {e}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    raise ValueError("Unsupported file format")
