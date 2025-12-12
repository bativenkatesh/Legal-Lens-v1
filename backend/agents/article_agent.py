from typing import List, Dict, Any
from openai import OpenAI
import json

# Import tools
try:
    from agent_tools import search_articles_local
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent_tools import search_articles_local

class ArticleAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are an expert Tax Consultant who applies legal theory to practical reality.
        
        Your Input:
        1. User Query
        2. Legal Context (provided by the Rule Agent)
        
        Your Goal:
        Find practical examples, interpreting articles, or case laws that support or clarify the legal rules.
        Then, answer the user's query comprehensively.
        
        Process:
        1. Read the Legal Context.
        2. Create a specific keyword search query for the Articles Database to find relevant interpretative content.
        3. Search using 'search_articles'.
        4. Synthesize the final answer:
           - Start with the Direct Legal Answer (based on Rules).
           - Support it with Practical Insights/Case Laws (based on Articles).
           - Cite sources clearly.
        """

    def run(self, query: str, rule_context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"--- Article Agent: Processing for context ---")
        
        legal_summary = rule_context.get("legal_summary", "")
        
        # Step 1: Formulate search query for articles
        # We can ask LLM to generate keywords or just use the user query + key sections
        
        # Quick prompt to get keywords
        keyword_prompt = f"""
        User Query: {query}
        Legal Context Summary: {legal_summary}
        
        Generate a concise keyword search string to find relevant articles/case laws. 
        Focus on key terms and section numbers. 
        Output ONLY the search string.
        """
        
        kw_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", # faster/cheaper for this simple task
            messages=[{"role": "user", "content": keyword_prompt}]
        )
        
        search_query = kw_response.choices[0].message.content.strip()
        print(f"--- Article Agent: Searching articles with '{search_query}' ---")
        
        # Step 2: Search local articles
        articles = search_articles_local(search_query, limit=5)
        
        # Step 3: Final Synthesis
        synthesis_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            User Query: {query}
            
            LEGAL RULE ANALYSIS (Theory):
            {legal_summary}
            
            RELEVANT ARTICLES/CASES (Practice):
            {json.dumps(articles, indent=2)}
            
            Please provide the Final Response to the user.
            Structure:
            1. **Legal Principle**: What the Act says (cite sections).
            2. **Practical Application/Analysis**: Insights from articles, case laws, or interpretations.
            3. **Conclusion**: Direct answer to the query.
            """}
        ]
        
        final_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=synthesis_messages
        )
        
        return {
            "response": final_response.choices[0].message.content,
            "articles_used": articles,
            "search_query_used": search_query
        }
