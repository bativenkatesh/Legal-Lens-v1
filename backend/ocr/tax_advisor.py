import json
from langchain_core.messages import HumanMessage, SystemMessage
from agent_core import llm
from agent_tools import search_legal_rules, search_practical_articles, search_news_articles

def run_tax_saving_analysis(data: dict) -> str:
    """
    Multi-stage Agentic RAG Pipeline for Tax Saving Advice.
    """
    
    # ---------------------------------------------------------
    # STAGE 2: ANALYSIS AGENT (Financial Reasoning + Query Gen)
    # ---------------------------------------------------------
    analysis_prompt = f"""
You are a Senior Tax Analyst. Analyze the following structured GST invoice data:
{json.dumps(data, indent=2)}

### TASK:
1. Identify potential tax-saving opportunities based on the HSN codes and totals.
2. Generate 3-5 high-impact search queries to find legal rules or real-world examples for these opportunities.

### OUTPUT FORMAT (JSON ONLY):
{{
  "reasoning": "Brief overview of financial status",
  "queries": ["query 1", "query 2", ...]
}}
"""
    print("\n--- [STAGE 2] ANALYZING DATA & GENERATING QUERIES ---")
    analysis_res = llm.invoke([SystemMessage(content="Return ONLY raw JSON."), HumanMessage(content=analysis_prompt)])
    
    try:
        # Clean potential markdown from LLM
        content = analysis_res.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        
        analysis_data = json.loads(content)
        reasoning = analysis_data.get("reasoning", "Analyzing GST layout...")
        queries = analysis_data.get("queries", [])
    except Exception as e:
        print(f"Analysis Parse Error: {e}")
        reasoning = "Initial financial analysis completed."
        queries = ["GST input tax credit loopholes", "Tax exemptions for business expenses"]

    # ---------------------------------------------------------
    # STAGE 3: RAG LAYER (Multi-Source Retrieval)
    # ---------------------------------------------------------
    print(f"\n--- [STAGE 3] EXECUTING RAG FOR {len(queries)} QUERIES ---")
    rag_context = []
    
    for q in queries[:3]: # Limit to top 3 for speed
        print(f"🔍 RAG Query: {q}")
        
        # Search Legal Rules (ChromaDB)
        legal = search_legal_rules(q)
        if legal and "No relevant sections" not in legal:
            rag_context.append(f"LEGAL RULE FOR '{q}':\n{legal}")
        
        # Search News/Articles (MongoDB)
        news = search_news_articles(q)
        if news and "No specific real-world examples" not in news:
            rag_context.append(f"REAL-WORLD EXAMPLE FOR '{q}':\n{news}")

    full_context = "\n\n".join(rag_context) if rag_context else "No specific legal exemptions found for these items."

    # ---------------------------------------------------------
    # STAGE 4: SYNTHESIS AGENT (Final Actionable Report)
    # ---------------------------------------------------------
    print("\n--- [STAGE 4] SYNTHESIZING FINAL ADVICE ---")
    
    synthesis_prompt = f"""
You are a high-end GST Tax Strategy Consultant. 
Based on the initial analysis and the retrieved legal/real-world context below, generate a FINAL TAX SAVING REPORT.

### INITIAL ANALYSIS:
{reasoning}

### RETRIEVED LEGAL & REAL-WORLD CONTEXT:
{full_context}

### INSTRUCTIONS:
- Be extremely specific (mention Section numbers and HSN codes if found).
- Provide actionable 'loopholes' or optimizations.
- Format the output in clean Markdown with professional headers.
- Include a "Real World Examples" section.
- Use a tone that is premium and confident.
"""
    
    final_res = llm.invoke([
        SystemMessage(content="You are a professional tax strategist. Provide specific, actionable advice based on the provided context."),
        HumanMessage(content=synthesis_prompt)
    ])
    
    return final_res.content
