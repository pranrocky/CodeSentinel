import os
import re
from langchain_core.tools import tool

SUPPORTED_EXTS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}

@tool
def universal_grep(search_term: str, repo_path: str) -> str:
    """
    Language-agnostic search tool. Searches the entire repository for an exact string,
    function name, or variable name. Returns the file paths and surrounding code snippets.
    Use this for JavaScript, Java, Go, etc., where the Python AST slicer won't work.
    
    Args:
        search_term: The exact text, function, or variable to search for.
        repo_path: The absolute path to the repository.
    """
    print(f"[Tool: Grep] Searching repo for '{search_term}'...")
    results = []
    
    for root, _, files in os.walk(repo_path):
        if '.git' in root:
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
                
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines):
                    if search_term in line:
                        # Grab 3 lines above and 5 lines below for context
                        start = max(0, i - 3)
                        end = min(len(lines), i + 6)
                        snippet = "".join(lines[start:end])
                        
                        match_info = f"### File: {filepath} (Line {i+1}) ###\n{snippet}\n"
                        results.append(match_info)
                        break # Only grab the first major context block per file to save tokens
            except Exception:
                pass
                
    if not results:
        return f"No matches found for '{search_term}' in the supported files."
        
    return "\n---\n".join(results[:5]) # Return top 5 matches to avoid blowing up the LLM context   