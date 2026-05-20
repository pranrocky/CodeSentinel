import subprocess
import json
import os
from langchain_core.tools import tool

@tool
def run_linter(file_path: str) -> str:
    """
    Run the ruff linter on a Python file. Returns a list of issues
    with code, message, and line number. 
    Use this to find real, verifiable bugs in a file before guessing or 
    asking for explanations.
    
    Args:
        file_path: The absolute path to the .py file to analyze.
    """
    print(f"[Tool: Linter] Analyzing {file_path}...")
    if not file_path.endswith('.py'):
        return "Error: The linter tool currently only supports Python (.py) files. Please use manual RAG analysis for this language."
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."

    try:
        # We run ruff as a subprocess and force it to output clean JSON
        result = subprocess.run(
            ['ruff', 'check', '--output-format=json', file_path],
            capture_output=True, 
            text=True
        )
        
        # Ruff returns exit code 1 if it finds issues, which is normal.
        # We parse the stdout into a Python dictionary.
        issues = json.loads(result.stdout or '[]')
        
        if not issues:
            return 'No issues found. The code is clean.'
            
        # Return a nicely formatted JSON string to the LLM
        return json.dumps(issues, indent=2)
        
    except FileNotFoundError:
        return "Error: 'ruff' is not installed or not found in the system PATH."
    except Exception as e:
        return f"Error executing linter: {e}"

# --- Execution Test Block ---
if __name__ == "__main__":
    import tempfile
    
    # Let's create a temporarily "buggy" Python file to test the linter
    buggy_code = """
import os
import sys  # Unused import
def add(a, b):
    return a + c  # 'c' is undefined
"""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, 'w') as f:
        f.write(buggy_code)
        
    print("\nExecuting Linter Tool on buggy file...")
    linter_output = run_linter.invoke({"file_path": path})
    
    print("\n" + "="*50)
    print(linter_output)
    print("="*50)
    
    # Cleanup
    os.remove(path)