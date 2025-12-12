from typing import List, Dict, Any
from openai import OpenAI
import json
import re

# Import tools (adjust path as needed depending on where this is run, 
# assuming run from backend root or python path set correctly)
try:
    from agent_tools import search_rules_atlas, get_related_rules, get_section_by_id
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent_tools import search_rules_atlas, get_related_rules, get_section_by_id

class RuleAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are an expert Legal Rule Analyst for Indian Tax Law.
        Your goal is to identify ALL relevant legal sections for a user's query and explain how they interact.
        
        Process:
        1. Analyze the user Query.
        2. use 'search_rules' to find the most relevant sections.
        3. READ the content of those sections carefully. 
        4. If a section mentions other sections (e.g., "Subject to section 45", "As per section 10(1)"), you MUST identify them as DEPENDENCIES.
        5. Use 'get_related_rules' to fetch those dependent sections if they are critical to the answer.
        6. Once you have all necessary rules, generate a "Legal Context Summary".
        
        Output Format:
        Return a structured object (JSON) containing:
        - "primary_sections": List of main sections found.
        - "dependencies": List of related sections fetched.
        - "summary": A clear, technical summary of the rules and how they apply.
        """

    def run(self, query: str) -> Dict[str, Any]:
        print(f"--- Rule Agent: Processing '{query}' ---")
        
        # Step 1: Initial Search
        # We can let the LLM decide to search, or just force a search. 
        # For efficiency in this architecture, we'll do an initial retrieval first, then let LLM refine.
        
        initial_results = search_rules_atlas(query, limit=5)
        
        # Step 2: Analyze for dependencies
        # We pass the initial results to the LLM and ask if it needs more info (dependencies).
        
        context_str = json.dumps(initial_results, indent=2)
        
        analysis_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nInitial Search Results:\n{context_str}\n\nIdentify if we need to fetch any referenced sections (dependencies) that are missing from the context but crucial for a complete answer. If yes, list the Section IDs. If no, just proceed to summarize."}
        ]
        
        # define a tool for the LLM to request more sections
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_dependencies",
                    "description": "Fetch content of referenced sections",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of Section IDs to fetch (e.g. ['45', '10(1)'])"
                            }
                        },
                        "required": ["section_ids"]
                    }
                }
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4", # or gpt-4-turbo
            messages=analysis_messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        tool_calls = msg.tool_calls
        
        fetched_dependencies = []
        
        if tool_calls:
            analysis_messages.append(msg)
            
            for tool_call in tool_calls:
                if tool_call.function.name == "fetch_dependencies":
                    args = json.loads(tool_call.function.arguments)
                    ids = args.get("section_ids", [])
                    print(f"--- Rule Agent: Fetching dependencies: {ids} ---")
                    
                    deps = get_related_rules(ids)
                    fetched_dependencies.extend(deps)
                    
                    analysis_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(deps, indent=2)
                    })
        
        # Final Step: Summarize everything
        final_instructions = """
        Based on all the rules found (primary and dependencies), provide a comprehensive 'Legal Context Summary'. 
        This summary should explain the legal position clearly.
        Also list the section numbers explicitly.
        """
        
        analysis_messages.append({"role": "user", "content": final_instructions})
        
        final_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=analysis_messages
        )
        
        summary = final_response.choices[0].message.content
        
        return {
            "primary_sections": initial_results,
            "dependencies": fetched_dependencies,
            "legal_summary": summary,
            "full_context_str": f"Query: {query}\nRules Found: {len(initial_results) + len(fetched_dependencies)}\nSummary: {summary}"
        }
