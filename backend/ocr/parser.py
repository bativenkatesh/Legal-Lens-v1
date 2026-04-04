from ocr.schemas import ParsedBill, ParsedGSTInvoice
from agent_core import orchestrator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json
ocr_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)
import re

def get_fallback_amounts(text):
    """
    Extract potential currency amounts from OCR text,
    filtering out long strings (Bank A/C, GSTIN, Phone).
    """
    matches = re.findall(r'\d[\d,]*\.?\d*', text)

    numbers = []
    for m in matches:
        clean_num = m.replace(",", "")
        try:
            val = float(clean_num)
            # 🔹 Filter: ignore numbers > 10 digits (Bank A/C / GSTIN / Phone)
            # Most invoices aren't 1000 Cr+ (10 digits)
            if len(clean_num.split('.')[0]) > 10:
                continue
            numbers.append(val)
        except:
            continue

    # Sort descending → biggest reasonable number = likely invoice value
    numbers.sort(reverse=True)

    return numbers
def fix_magnitude(value):
    """
    Fix common OCR/LLM magnitude errors:
    - Extra zero (11800000 instead of 1180000)
    - Missing zero (118000 instead of 1180000)
    """

    if not value:
        return value

    # If number is suspiciously large (> 50 lakh)
    if value > 5_000_000:
        # Try removing one zero
        reduced = value / 10

        # If reduced looks reasonable → use it
        if reduced < 5_000_000:
            return reduced

    return value
def parse_indian_number(num_str):
    if not num_str:
        return 0.0
    
    # Remove commas
    num_str = num_str.replace(",", "").strip()

    try:
        return float(num_str)
    except:
        return 0.0

def parse_bill_text(text: str) -> ParsedBill:
    prompt = f"""
You are a tax document extraction assistant.

From the OCR text below, extract bill details and return ONLY valid JSON.

Required fields:
- vendor
- bill_date (YYYY-MM-DD if possible)
- total_amount (number only)
- currency
- category (Medical, Rent, Insurance, Donation, Education, Electronics, Other)
- tax_section (Indian Income Tax Act, if applicable, else null)
- tax_eligible (true/false)
- confidence (0 to 1)

OCR TEXT:
{text}
"""

    messages = [
        SystemMessage(
            content=(
                "You are an information extraction engine. "
                "Do NOT use tools. Do NOT explain. "
                "Return ONLY valid JSON."
            )
        ),
        HumanMessage(content=prompt)
    ]

    response = ocr_llm.invoke(messages)
    output = response.content


    try:
        print("\n--- [OCR] LLM RAW OUTPUT ---")
        print(output)
        print("----------------------------\n")

        parsed = ParsedBill.model_validate_json(output)

        # 🔹 Amount normalization (OCR safety)
        if parsed.total_amount and parsed.total_amount > 100000:
            parsed.total_amount = parsed.total_amount / 100

        print("\n--- [OCR] FINAL PARSED DATA ---")
        print(parsed.model_dump_json(indent=2))
        print("-------------------------------\n")

        return parsed
    except Exception:
        # fallback if LLM messes up
        return ParsedBill(
            vendor=None,
            bill_date=None,
            total_amount=None,
            currency=None,
            category="Other",
            tax_section=None,
            tax_eligible=False,
            confidence=0.2
        )

def parse_gst_invoice_text(text: str) -> ParsedGSTInvoice:
    prompt = f"""
You are an expert in Indian GST invoice data extraction.
Extract the following fields from the OCR text into a valid JSON object.

OUTPUT FORMAT (STRICT — DO NOT CHANGE KEYS):

{{
  "b2b": [
    {{
      "Invoice Date": "",
      "Invoice Value": 0,
      "Place Of Supply": "",
      "Reverse Charge": "",
      "Applicable % of Tax Rate": 0,
      "Invoice Type": "",
      "E-Commerce GSTIN": "",
      "Rate": 0,
      "Taxable Value": 0,
      "Cess Amount": 0
    }}
  ],
  "hsn": [
    {{
      "HSN": "",
      "Description": "",
      "UQC": "",
      "Total Quantity": 0,
      "Total Value": 0,
      "Rate": 0,
      "Taxable Value": 0,
      "Integrated Tax Amount": 0,
      "Central Tax Amount": 0,
      "State/UT Tax Amount": 0,
      "Cess Amount": 0
    }}
  ],
  "documents": [
    {{
      "Nature of Document": "",
      "Sr. No. From": "",
      "Sr. No. To": "",
      "Total Number": 0,
      "Cancelled": 0
    }}
  ]
}}

RULES:
- Return ONLY JSON (no explanation)
- Dates → YYYY-MM-DD
- Remove commas from numbers
- Reverse Charge → "Y" or "N"
- Place Of Supply → state code (KA, MH, etc.)
- Rate = total GST rate (e.g., 18)
- If missing → use null or 0

NUMERIC FORMATS:
- The text uses Indian numbering (e.g., 20,00,000 is 2 million).
- Be EXTREMELY careful not to add or remove zeros.
- Return ONLY the number without commas or currency symbols.

CRITICAL EXTRACTION RULES:
1. IGNORE "Bank Details", "A/C No", "Account Number", "IFSC", and "GSTIN" values. 
2. The "Invoice Value" is the Grand Total (incl. tax), usually found near "Total" or "Total Invoice value".
3. The "Taxable Value" is the total before GST.
4. If you see a very long number (12+ digits), it is LIKELY a bank account. DO NOT extract it as Invoice Value.

OCR TEXT:
{text}
"""

    print("\n--- [GST OCR] RAW TEXT FROM FILE ---")
    print(text)
    print("------------------------------------\n")

    messages = [
        SystemMessage(content="Return ONLY raw JSON. No markdown, no explanation."),
        HumanMessage(content=prompt)
    ]

    response = ocr_llm.invoke(messages)
    output = response.content.strip()

    # 🔹 Robust JSON extraction (handles markdown blocks)
    if "```" in output:
        output = output.split("```")[1].replace("json", "").strip()

    try:
        print("\n--- [GST OCR] LLM RAW OUTPUT ---")
        print(output)
        print("--------------------------------\n")

        data = json.loads(output)
        fallback_nums = get_fallback_amounts(text)
        print("🔢 FALLBACK NUMBERS (Sanitized):", fallback_nums)

        # 🔥 NORMALIZATION
        for inv in data.get("b2b", []):
            # 1. Preferred approach: Use LLM output but fix magnitude if needed
            val = parse_indian_number(str(inv.get("Invoice Value", 0)))
            
            # 2. Logic Check: If LLM picked a bank account (too many digits)
            if len(str(int(val))) > 10:
                val = 0 # invalidate it
            
            # 3. Fallback: If LLM returned 0, use largest sanitized regex number
            if not val and fallback_nums:
                val = fallback_nums[0]
            
            # 4. Final Magnitude Fix
            inv["Invoice Value"] = fix_magnitude(val)

            # Similar for Taxable Value
            tax_val = parse_indian_number(str(inv.get("Taxable Value", 0)))
            if not tax_val and len(fallback_nums) >= 2:
                tax_val = fallback_nums[1]
            inv["Taxable Value"] = fix_magnitude(tax_val)

            inv["Cess Amount"] = parse_indian_number(str(inv.get("Cess Amount", 0)))

        print("\n--- [GST OCR] FINAL PARSED DATA (NORMALIZED) ---")
        print(json.dumps(data, indent=2))
        print("------------------------------------------------\n")

        return data

    except Exception as e:
        print("❌ LLM PARSE ERROR:", e)

        # SAFE FALLBACK
        return {
            "b2b": [],
            "hsn": [],
            "documents": []
        }

