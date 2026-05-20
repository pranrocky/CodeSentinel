import ast
import os
from langchain_core.tools import tool

# We wrap this in LangChain's @tool decorator so the Agent can natively understand 
# its inputs, outputs, and purpose via the docstring.
@tool
def get_call_slice(function_name: str, repo_path: str) -> str:
    """
    Finds a function by name in the repository, extracts its source code, 
    and attempts to find the source code of any internal functions it calls.
    Use this to understand the complete execution path of an entry point.
    
    Args:
        function_name: The exact name of the Python function to trace.
        repo_path: The absolute path to the cloned repository.
    """
    print(f"[Tool: Slicer] Tracing execution path for '{function_name}'...")
    
    target_node = None
    target_filepath = None
    target_source = None
    
    # STEP 1: Walk the repo to find where the requested function is defined
    for root, _, files in os.walk(repo_path):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse the file into an Abstract Syntax Tree
                tree = ast.parse(source_code)
                
                # Walk every node in the tree looking for a 'def' (FunctionDef)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                        target_node = node
                        target_filepath = filepath
                        target_source = source_code
                        break
            except Exception:
                # Silently skip files that have syntax errors or encoding issues
                continue
                
        if target_node:
            break
            
    # If we searched the whole repo and found nothing, report back to the LLM
    if not target_node:
        return f"Error: Could not find definition for function '{function_name}' in the repository."

    # STEP 2: We found the main function. Now, find what it calls.
    # We look for 'Call' nodes (e.g., helper_function()) inside the main function's body.
    called_function_names = set()
    for node in ast.walk(target_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_function_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # Handles method calls like 'self.process_data()' -> grabs 'process_data'
                called_function_names.add(node.func.attr)

    # STEP 3: Format the main function's source code using ast.get_source_segment
    main_func_code = ast.get_source_segment(target_source, target_node)
    
    # We format the string that will be returned to the LLM
    result = f"### EXECUTION SLICE FOR: {function_name} ###\n\n"
    result += f"# File: {target_filepath}\n"
    result += f"{main_func_code}\n\n"
    
    if called_function_names:
        result += f"### INTERNAL CALLS DETECTED: {', '.join(called_function_names)} ###\n"
        result += "(Agent Note: If you need to see the logic of these internal calls to find a bug, use the search_symbol or rag_query tools to fetch them.)\n"

    return result

# --- Execution Test Block ---
if __name__ == "__main__":
    from ingestion.loader import load_repo
    
    test_url = "https://github.com/pallets/flask"
    
    print("Loading repo for Slicer test...")
    # We just need the path to the raw files for the Slicer tool
    _, repo_path = load_repo(test_url)
    
    # Let's test it by asking it to slice Flask's built-in 'url_for' function
    print("\nExecuting Slicer Tool on 'url_for'...")
    slice_output = get_call_slice.invoke({
        "function_name": "url_for", 
        "repo_path": repo_path
    })
    
    print("\n" + "="*50)
    print(slice_output)
    print("="*50)