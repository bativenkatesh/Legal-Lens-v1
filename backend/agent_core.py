from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agent_tools import search_legal_rules, search_practical_articles
import sys

# 1. Initialize Ollama (Using Llama 3.1 for Tool Support)
print("--- [AGENT] Initializing Local LLM (Llama 3.1)... ---")
try:
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    sys.exit(1)

# 2. Register Tools
tools = [search_legal_rules, search_practical_articles]

# 3. System Prompt (ANTI-HALLUCINATION VERSION)
system_prompt = """You are a precise Legal Retrieval Assistant.
Your ONLY source of truth is the text provided by your tools (`search_legal_rules`).

### OPERATING RULES:
1. **Analyze the Tool Outputs**: Look at the text returned by `search_legal_rules`.
2. **Handle Missing Data**: 
   - If `search_practical_articles` returns "No articles found", IGNORE the articles completely.
   - **DO NOT** make up information to fill the gap.
   - **DO NOT** use your internal training data to answer.
   - Answer solely based on the Legal Rules text you found in step 1.
3. **Strict Grounding**:
   - If the tools return Section 10A/10B, discuss Section 10A/10B. 
   - Do NOT bring up unrelated sections like 10(13A) unless they are in the tool output.

### FAILURE HANDLING:
If you have the Legal Rules but no Articles, your answer should be:
"Based on the legal text found in [Section Number]..." and then summarize the legal text.
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
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=3 # Prevent loops
)