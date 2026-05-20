import ast
import os
from langchain_core.tools import tool

@tool
def search_symbol(name: str, repo_path: str) -> str:
    """
    Find exactly where a function, class, or variable is defined in the repository.
    Use this when you need the exact definition location (file path and line number) 
    of a specific symbol.
    
    Args:
        name: The exact function or class name to search for (e.g., 'url_for').
        repo_path: The absolute path to the cloned repository workspace.
    """
    print(f"[Tool: Symbol Search] Scanning for symbol '{name}'...")
    results = []
    
    # Walk the entire directory tree
    for root, _, files in os.walk(repo_path):
        for f in files:
            if not f.endswith('.py'): 
                continue
                
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8') as file:
                    tree = ast.parse(file.read())
                    
                    # Walk the AST looking for Class or Function definitions
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            if node.name == name:
                                results.append(f"Found '{name}' at {fpath}: Line {node.lineno}")
            except Exception:
                # Silently pass over files with syntax errors (like Python 2 code)
                pass
                
    if results:
        return '\n'.join(results)
    else:
        return f"Symbol '{name}' not found in the repository."

# --- Execution Test Block ---
if __name__ == "__main__":
    from ingestion.loader import load_repo
    
    test_url = "https://github.com/pallets/flask"
    
    print("Loading repo for Symbol Search test...")
    _, repo_path = load_repo(test_url)
    
    # Let's search for Flask's core 'Flask' class
    print("\nExecuting Symbol Search Tool for 'Flask'...")
    symbol_output = search_symbol.invoke({
        "name": "Flask", 
        "repo_path": repo_path
    })
    
    print("\n" + "="*50)
    print(symbol_output)
    print("="*50)