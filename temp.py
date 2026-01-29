import pandas as pd
import requests
from tqdm import tqdm
import time


# ======================================================
# CONFIGURATION
# ======================================================

INPUT_CSV = "sam_before.csv"          # your existing CSV
OUTPUT_CSV = "acts_enhanced.csv"      # new CSV with extra columns

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SLEEP_BETWEEN_CALLS = 0.5


# ======================================================
# LLM CALL (STRICT FORMAT)
# ======================================================

def call_llm(section, title, summary):
    """
    Returns:
    entity_class | special_category | status | effective_to
    """

    prompt = f"""
You are classifying sections of the Indian Income-tax Act.

Section: {section}
Title: {title}

Summary:
{summary}

Respond in EXACTLY this format (ONE LINE ONLY):

<ENTITY_CLASS>|<SPECIAL_CATEGORY>|<STATUS>|<LAST_AY>

Where:

ENTITY_CLASS = Individual, Company, Firm, HUF, General
SPECIAL_CATEGORY = EOU, SEZ, Banking, Infrastructure, Handicraft, None
STATUS = Active, Withdrawn, Amended
LAST_AY = Assessment year like AY 2011-12 or None

Rules:
- ONE LINE ONLY
- NO explanations
- NO markdown
- NO extra text
- If unclear, use General|None
- If not withdrawn, LAST_AY = None
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        response.raise_for_status()

        raw_output = response.json()["response"].strip()

        # Print raw LLM output for debugging
        print("\n==============================")
        print(f"SECTION: {section}")
        print("🧠 LLM OUTPUT:", raw_output)
        print("==============================\n")

        parts = [p.strip() for p in raw_output.split("|")]

        if len(parts) != 4:
            raise ValueError("Invalid output format")

        return {
            "entity_class": parts[0],
            "special_category": parts[1],
            "status": parts[2],
            "effective_to": parts[3],
            "llm_confidence": "LOW"
        }

    except Exception as e:
        print("❌ LLM FORMAT ERROR")
        print("Section:", section)
        print("Error:", e)

        return {
            "entity_class": "General",
            "special_category": "None",
            "status": "Unknown",
            "effective_to": None,
            "llm_confidence": "ERROR"
        }


# ======================================================
# MAIN PROCESS
# ======================================================

def main():
    print("📂 Loading existing CSV...")
    df = pd.read_csv(INPUT_CSV)

    # Ensure new columns exist
    new_columns = [
        "entity_class",
        "special_category",
        "status",
        "effective_to",
        "llm_confidence"
    ]

    for col in new_columns:
        if col not in df.columns:
            df[col] = None

    print(f"🔍 Processing {len(df)} sections...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # Skip rows already processed
        if pd.notna(row["entity_class"]):
            continue

        result = call_llm(
            section=row["section"],
            title=row["title"],
            summary=row["ai_generated_summary"]
        )

        df.at[idx, "entity_class"] = result["entity_class"]
        df.at[idx, "special_category"] = result["special_category"]
        df.at[idx, "status"] = result["status"]
        df.at[idx, "effective_to"] = result["effective_to"]
        df.at[idx, "llm_confidence"] = result["llm_confidence"]

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("💾 Writing enhanced CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ DONE! Enhanced CSV written to: {OUTPUT_CSV}")
    print("⚠️ IMPORTANT: Manually review before using in RAG.")


if __name__ == "__main__":
    main()
