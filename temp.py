import pandas as pd
import requests
from tqdm import tqdm
import json
import time


# ======================================================
# CONFIG
# ======================================================

INPUT_CSV = "sam_before.csv"              # your existing CSV
OUTPUT_CSV = "acts_enhanced.csv"        # new CSV
OLLAMA_MODEL = "llama3.1:8b"                 # change if needed
OLLAMA_URL = "http://localhost:11434/api/generate"

SLEEP_BETWEEN_CALLS = 0.5               # be nice to your CPU


# ======================================================
# LLM CALL
# ======================================================

def call_llm(section, title, content):
    prompt = f"""
You are assisting in structuring Indian Income-tax law.

Section: {section}
Title: {title}

Content:
{content[:3000]}

Tasks:
1. Identify who this section applies to.
   Choose ONE from:
   Individual, Company, EOU, SEZ, Handicraft, Infrastructure, Banking, General, Unknown

2. Identify status:
   Active, Withdrawn, Amended, Unknown

3. If withdrawn, mention last applicable assessment year (else null).

Rules:
- Do NOT guess
- If unclear, say "Unknown"
- Output JSON ONLY
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        text = response.json()["response"]

        data = json.loads(text)

        return {
            "applies_to": data.get("applies_to", "Unknown"),
            "status": data.get("status", "Unknown"),
            "effective_to": data.get("effective_to"),
            "llm_confidence": "LOW"   # always low until human review
        }

    except Exception as e:
        return {
            "applies_to": "ERROR",
            "status": "ERROR",
            "effective_to": None,
            "llm_confidence": "ERROR"
        }


# ======================================================
# MAIN PROCESS
# ======================================================

def main():
    print("📂 Loading existing CSV...")
    df = pd.read_csv(INPUT_CSV)

    # Add new columns (if not already present)
    for col in ["applies_to", "status", "effective_to", "llm_confidence"]:
        if col not in df.columns:
            df[col] = None

    print(f"🔍 Processing {len(df)} sections...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Skip if already processed
        if pd.notna(row["applies_to"]):
            continue

        result = call_llm(
            section=row["section"],
            title=row["title"],
            content=row["full_content"]
        )

        df.at[idx, "applies_to"] = result["applies_to"]
        df.at[idx, "status"] = result["status"]
        df.at[idx, "effective_to"] = result["effective_to"]
        df.at[idx, "llm_confidence"] = result["llm_confidence"]

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("💾 Writing enhanced CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Done! New file written: {OUTPUT_CSV}")
    print("⚠️ IMPORTANT: Review and verify before using in RAG.")


if __name__ == "__main__":
    main()
