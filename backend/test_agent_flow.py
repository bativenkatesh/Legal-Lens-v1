import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_core import orchestrator
    print("✅ Successfully imported orchestrator from agent_core")
except ImportError as e:
    print(f"❌ Failed to import orchestrator: {e}")
    sys.exit(1)

def test_flow():
    query = "What are the deductions available under Section 80C?"
    print(f"\nTesting Query: {query}")
    
    # Check if keys are present (warn if not)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Test might fail or return error response.")
    if not os.getenv("ATLAS_URI"):
        print("⚠️  ATLAS_URI not set. Vector search will fail.")
        
    try:
        result = orchestrator.run(query)
        
        print("\n--- Result Summary ---")
        print(f"Response Length: {len(result.get('response', ''))}")
        print(f"Primary Sections Found: {len(result.get('primary_sections', []))}")
        print(f"Dependencies Found: {len(result.get('dependencies', []))}")
        print(f"Articles Found: {len(result.get('relevant_articles', []))}")
        print("\n✅ Test Run Completed (Check output for correctness)")
        
    except Exception as e:
        print(f"❌ Test Failed with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flow()
