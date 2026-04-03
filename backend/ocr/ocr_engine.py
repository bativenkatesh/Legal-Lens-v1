import os
import easyocr
from pdf2image import convert_from_path

# Initialize once (global)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError("Bill file not found")

    # ---------------- IMAGE ----------------
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        result = reader.readtext(file_path)

        text_lines = []
        for res in result:
            try:
                text_lines.append(res[1])
            except Exception:
                continue

        return "\n".join(text_lines)

    # ---------------- PDF ----------------
    if file_path.lower().endswith(".pdf"):
        try:
            images = convert_from_path(file_path, dpi=300)
            full_text = ""

            for i, img in enumerate(images):
                temp_path = f"temp_page_{i}.png"
                img.save(temp_path)

                result = reader.readtext(temp_path)

                text_lines = []
                for res in result:
                    try:
                        text_lines.append(res[1])
                    except Exception:
                        continue

                full_text += " ".join(text_lines) + "\n"

                os.remove(temp_path)

            return full_text.strip()

        except Exception as e:
            print("🔥 OCR ERROR:", str(e))
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    raise ValueError("Unsupported file format")