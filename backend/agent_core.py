from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agent_tools import search_legal_rules, search_practical_articles
import sys

# 1. Initialize Ollama
# Ensure you have run `ollama pull mistral` in your terminal
print("--- [AGENT] Initializing Local LLM (Ollama: Mistral)... ---")
try:
    llm = ChatOllama(model="mistral:7b", temperature=0)
except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    sys.exit(1)

# 2. Register Tools
tools = [search_legal_rules, search_practical_articles]

# 3. System Prompt
system_prompt = """You are 'LegalLens', an elite Indian Tax Law Expert.
Your goal is to provide comprehensive answers by combining LEGAL STATUTES (Rules) with PRACTICAL INTERPRETATION (Articles).

### YOUR PROTOCOL:
1.  **Search the Law**: ALWAYS start by using `search_legal_rules` to find the exact Section numbers (e.g., "Section 80C", "Section 10(13A)").
2.  **Search Practice**: AFTER finding the law, use `search_practical_articles` to find real-world case laws or examples.
3.  **Synthesize**: Combine the "Letter of the Law" with "Real World Practice" to answer the user.

If you cannot find specific sections, state that clearly. Do not hallucinate laws.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Create Agent & Executor
agent = create_tool_calling_agent(llm, tools, prompt)

orchestrator = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Shows the "Thinking" process in console
    handle_parsing_errors=True
)