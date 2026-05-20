import os
from langchain_core.tools import tool

@tool
def list_repo_files(repo_path: str) -> str:
    """
    Returns a list of all code and documentation files in the repository.
    Use this FIRST when asked for a broad summary or to explain how multiple files work.
    
    Args:
        repo_path: The absolute path to the repository.
    """
    print("[Tool: Explorer] Listing repository files...")
    file_list = []
    
    for root, _, files in os.walk(repo_path):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root:
            continue
        for file in files:
            # Get the relative path to make it easier for the LLM to read
            rel_path = os.path.relpath(os.path.join(root, file), repo_path)
            file_list.append(rel_path)
            
    if not file_list:
        return "Repository is empty or files could not be read."
        
    return "REPOSITORY CONTENTS:\n" + "\n".join(file_list)

@tool
def read_file(file_path: str, repo_path: str) -> str:
    """
    Reads the exact, complete contents of a single file. 
    Use this after list_repo_files to inspect specific files of interest.
    
    Args:
        file_path: The relative path to the file (e.g., 'src/main.js').
        repo_path: The absolute path to the repository.
    """
    print(f"[Tool: Explorer] Reading file: {file_path}...")
    full_path = os.path.join(repo_path, file_path)
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Optional safeguard: truncate massive files (e.g., > 1000 lines) to save context
        lines = content.split('\n')
        if len(lines) > 1000:
            return "\n".join(lines[:1000]) + "\n\n... [TRUNCATED DUE TO LENGTH] ..."
            
        return content
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"