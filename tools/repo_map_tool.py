import os
from langchain_core.tools import tool

# Only read text/code files, ignore binaries and heavy assets
SUPPORTED_EXTS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.html', '.css', '.md'}

@tool
def read_entire_codebase(repo_path: str) -> str:
    """
    Reads the entire repository into a single string wrapped in XML tags.
    ONLY use this if the user asks for a broad, global summary of the whole project.
    """
    print("[Tool: Repo Map] Reading entire codebase into context...")
    all_code = []
    total_chars = 0
    
    for root, _, files in os.walk(repo_path):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root:
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTS and file.lower() != 'readme.md':
                continue
                
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_chars += len(content)
                    
                    # Emergency Brake: ~100k chars is a safe max for a 14B local model
                    if total_chars > 100000:
                        return "Error: Repository is too large to read entirely. Use `list_repo_files` and `read_file` instead."
                        
                    # Use relative paths so the LLM doesn't get confused by your local /tmp/ paths
                    rel_path = os.path.relpath(filepath, repo_path)
                    all_code.append(f"<file name=\"{rel_path}\">\n{content}\n</file>")
            except Exception:
                pass
                
    if not all_code:
        return "Repository is empty or no supported files found."
        
    return "\n\n".join(all_code)