import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools.rag_tool import rag_query
import config

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    temperature=0.2
)

@tool
def suggest_fix(issue_json: str) -> str:
    """
    Suggest a concrete Python fix for a linter-detected issue.
    
    Args:
        issue_json: a single issue object from the run_linter output (JSON format)
    """
    print("[Tool: Suggest Fix] Generating code fix...")
    try:
        issue = json.loads(issue_json)
        filename = issue.get("filename", "unknown_file.py")
        row = issue.get("location", {}).get("row", "unknown line")
        message = issue.get("message", "unknown issue")
        code_rule = issue.get("code", "unknown rule")
        
        # Ask RAG for the specific file and line number
        context = rag_query.invoke({"query": f"code at {filename} line {row}"})
        
        prompt = f"""
        Linter Issue: {message} (Rule: {code_rule})
        File: {filename} at line {row}
        
        Code Context:
        {context}
        
        Write the corrected Python code to fix this linter issue. Include a very brief explanation of what you changed.
        """
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error parsing issue or generating fix: {str(e)}"

# --- Execution Test Block ---
if __name__ == "__main__":
    # Mock output from your linter tool from earlier
    mock_linter_issue = json.dumps({
        "filename": "/tmp/tmp_lzielru.py",
        "location": {"row": 5},
        "message": "Undefined name `c`",
        "code": "F821"
    })
    
    print("\nExecuting Fix Suggester Tool...")
    fix_output = suggest_fix.invoke({"issue_json": mock_linter_issue})
    print("\n" + "="*50)
    print(fix_output)
    print("="*50)