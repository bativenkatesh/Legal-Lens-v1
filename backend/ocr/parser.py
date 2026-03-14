from ocr.schemas import ParsedBill, ParsedGSTInvoice
from agent_core import orchestrator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

ocr_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

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
        parsed = ParsedBill.model_validate_json(output)

        # 🔹 Amount normalization (OCR safety)
        if parsed.total_amount and parsed.total_amount > 100000:
            parsed.total_amount = parsed.total_amount / 100

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

Fields:
1. "type": "E" if E-commerce, "OE" otherwise.
2. "place_of_supply": State name/code (e.g. "Karnataka" or "29-Karnataka").
3. "applicable_tax_rate_percent": Look for "Applicable % of Tax Rate" column. If empty or dashed, return "". If it says "65%", return "65%".
4. "rate": Combined GST rate (e.g. if CGST=9% and SGST=9%, return "18%").
5. "taxable_value": The BASE amount before tax.
6. "total_amount": The final amount including tax (Grand Total).
7. "cess_amount": Cess amount as a number, else 0.
8. "gstin": The primary GSTIN mentioned in the invoice (E-commerce or Vendor GSTIN).

NUMERIC FORMATS:
- The text uses Indian numbering (e.g., 20,00,000 is 2 million).
- Be EXTREMELY careful not to add or remove zeros.
- Return ONLY the number without commas or currency symbols.

OCR TEXT:
{text}
"""

    messages = [
        SystemMessage(content="Return ONLY raw JSON. No markdown, no explanation."),
        HumanMessage(content=prompt)
    ]

    response = ocr_llm.invoke(messages)
    output = response.content.strip()

    # 🔹 Robust JSON extraction (handles markdown blocks)
    if "```json" in output:
        output = output.split("```json")[1].split("```")[0].strip()
    elif "```" in output:
        output = output.split("```")[1].split("```")[0].strip()

    try:
        return ParsedGSTInvoice.model_validate_json(output)
    except Exception:
        # Fallback
        return ParsedGSTInvoice(
            type="OE",
            place_of_supply=None,
            applicable_tax_rate_percent="",
            rate=None,
            taxable_value=0.0,
            cess_amount=0.0,
            ecommerce_gstin=None
        )

