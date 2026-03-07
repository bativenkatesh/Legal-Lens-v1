from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

router_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

async def route_query(query: str) -> str:

    prompt = f"""
You are a query router for a tax AI assistant.

Classify the user query into one of three categories:

FINANCIAL → question about user's own deductions, tax savings, bills, expenses.
LEGAL → question about tax law sections, rules, or regulations.
GENERAL → general conversation or unrelated question.

Return ONLY one word:
FINANCIAL
LEGAL
GENERAL

Query:
{query}
"""

    messages = [
        SystemMessage(content="You classify user queries."),
        HumanMessage(content=prompt)
    ]

    response = router_llm.invoke(messages)

    return response.content.strip()