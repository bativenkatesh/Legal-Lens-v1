from ocr.schemas import ParsedBill
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
