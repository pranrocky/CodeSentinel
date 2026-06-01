import os
import sys
import json
import random
import argparse
import re

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def extract_unique_keyphrases(func_name: str, code: str, lang: str) -> list[str]:
    """
    Extracts highly distinctive, technical code tokens directly from the implementation.
    Filters out common language keywords, stop words, and generic types, ensuring that
    phrase coverage correlates directly with high-fidelity technical precision.
    """
    phrases = [func_name]
    
    # If dot-notation, add the base function/method name too
    if "." in func_name:
        phrases.append(func_name.split(".")[-1])
        
    # Extract structural words from code (length >= 5)
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{4,}', code)
    
    # Stop words/generic terms to ignore
    stop_words = {
        "public", "private", "protected", "class", "void", "return", "import", "package",
        "function", "const", "let", "var", "async", "await", "func", "struct", "interface",
        "string", "String", "int", "float", "double", "boolean", "Boolean", "Object", "object",
        "null", "true", "false", "throw", "new", "except", "error", "nil", "type", "chan",
        "case", "select", "default", "defer", "go", "for", "if", "else", "switch", "break",
        "continue", "java", "python", "javascript", "golang", "ruby", "php", "this", "self",
        "require", "console", "export", "module", "undefined", "map", "list", "array", "Array"
    }
    
    candidates = []
    for w in words:
        if w not in stop_words and len(w) >= 6:
            # Prioritize camelCase, snake_case, or uppercase-starting custom classes/methods
            if "_" in w or any(c.isupper() for c in w[1:]):
                candidates.append(w)
                
    # Select top 3 unique technical candidates
    unique_candidates = []
    for c in candidates:
        if c not in phrases and c not in unique_candidates:
            unique_candidates.append(c)
            if len(unique_candidates) >= 3:
                break
                
    phrases.extend(unique_candidates)
    return phrases

def main():
    parser = argparse.ArgumentParser(description="📦 CodeSentinel - CodeSearchNet Evaluation Dataset Generator")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to collect per language (default: 5)"
    )
    args = parser.parse_args()
    
    print("==================================================================")
    print("📦  CodeSentinel - CodeSearchNet Evaluation Dataset Generator  📦")
    print("==================================================================")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Error: 'datasets' package is not installed.")
        print("[*] Please run: pip install datasets")
        sys.exit(1)
        
    languages = ["python", "javascript", "go", "java"]
    samples_per_lang = args.samples
    dataset = []
    
    print(f"[*] Loading test splits from Hugging Face for: {languages}...")
    print(f"[*] Configured samples per language: {samples_per_lang} (Total targeted: {samples_per_lang * len(languages)})")
    print("-" * 66)
    
    for lang in languages:
        print(f"    -> Loading '{lang}' test split...")
        try:
            # Load test split (much smaller than train)
            ds = load_dataset("code_search_net", lang, split="test")
            
            # Shuffle with reproducible seed
            ds_shuffled = ds.shuffle(seed=42)
            
            print(f"    -> Selecting and filtering {samples_per_lang} samples...")
            count = 0
            for item in ds_shuffled:
                if count >= samples_per_lang:
                    break
                    
                func_name = item.get("func_name", "").strip()
                docstring = item.get("func_documentation_string", "").strip()
                code = item.get("func_code_string", "").strip()
                repo = item.get("repo", "").strip()
                
                # Ensure we have a valid docstring and function name
                if not func_name or not docstring or len(docstring) < 15:
                    continue
                    
                # Set expectations based on language
                if lang == "python":
                    expected_tools = ["search_symbol", "rag_query", "get_call_slice"]
                else:
                    # Non-python languages only support grep and RAG
                    expected_tools = ["rag_query", "universal_grep"]
                    
                # Clean up query
                query_text = f"Explain and locate the implementation for the {lang} function that performs: {docstring}"
                
                # Programmatically extract unique, high-fidelity code-token keyphrases
                key_phrases = extract_unique_keyphrases(func_name, code, lang)
                
                dataset.append({
                    "id": f"CSN_{lang.upper()}_{func_name}",
                    "language": lang,
                    "query": query_text,
                    "expected_tools": expected_tools,
                    "key_phrases": key_phrases,
                    "ground_truth_code": code,
                    "repo": repo
                })
                count += 1
                
        except Exception as e:
            print(f"    [!] Failed to load language '{lang}': {e}")
            
    if not dataset:
        print("[!] Error: No samples were collected.")
        sys.exit(1)
        
    # Shuffle entire combined dataset so language tests interleave
    random.seed(42)
    random.shuffle(dataset)
    
    out_file = os.path.join(config.BASE_DIR, "evaluation", "dataset.json")
    with open(out_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print("-" * 66)
    print(f"[+] Success! Generated {len(dataset)} cross-language CodeSearchNet samples.")
    print(f"[+] Saved dataset to: {out_file}")

if __name__ == "__main__":
    main()
