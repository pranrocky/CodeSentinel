import os
import time
import json
import urllib.request
import urllib.error
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import agent and prompts
import config
from agent.graph import agent_app, set_agent_model
from agent.prompts import get_system_prompt, PROMPT_VARIANTS

OLLAMA_API_URL = "http://localhost:11434/api"

def check_available_ollama_models() -> list[str]:
    """Queries local Ollama tags API to find currently installed models."""
    print("[*] Contacting local Ollama server to fetch installed models...")
    try:
        req = urllib.request.Request(f"{OLLAMA_API_URL}/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            print(f"[*] Found {len(models)} installed models: {models}")
            return models
    except Exception as e:
        print(f"[!] Warning: Could not contact Ollama API ({e}). Falling back to hardcoded check.")
        return []

def run_auto_judge(query: str, answer: str, expected_phrases: list[str], judge_model: str) -> float:
    """Uses the judge LLM to score the response from 1 to 5 based on accuracy."""
    phrase_list = ", ".join([f"'{p}'" for p in expected_phrases])
    
    prompt = f"""You are an expert AI evaluator and code auditor.
Evaluate the following Assistant's response to the User's Query.

User Query: "{query}"

Assistant's Response:
\"\"\"
{answer}
\"\"\"

Ground Truth Key Reference Phrases: [{phrase_list}]

=== CALIBRATION SCORING GUIDELINES & FEW-SHOT EXAMPLES ===
Your scoring must be strictly calibrated against these anchor cases:

- SCORE 1.0/5.0: Completely wrong, irrelevant, or hallucinated answer. 
  Example: "I don't see any load_repo function in this codebase, perhaps it is a third party package." (when load_repo is defined in ingestion/loader.py).

- SCORE 3.0/5.0: Answer is factually correct and somewhat helpful, but vague and lacks key technical detail/concreteness.
  Example: "The load_repo function is defined in ingestion/loader.py. It loads files from local directories or clones a git repository, and scrapes the files." (Correct, but lacks deep analysis of git clones, directory scraping, supported extensions, workspace storage, or dependencies).

- SCORE 5.0/5.0: Perfect, flawless, precise answer citing exact files, code locations, design details, and accurately mentioning expected reference phrases.
  Example: "The load_repo function is defined in ingestion/loader.py (lines 50-73). It accepts a local folder path or git clone URL. For local directories, it calls scrape_directory to parse code files. For remote URLs, it prevents disk leaks by cloning to a reusable workspace at /tmp/codesentinel_workspace using Repo.clone_from. It parses extensions [.py, .js, .ts, .java, .cpp, .go, .rs] and maps them to LlamaIndex Document nodes, skipping cached artifacts and cache dirs."

Score the response strictly on a scale from 1.0 to 5.0.

Your output must be a single JSON object containing:
{{
  "score": <float, 1.0 to 5.0>,
  "reason": "<1 sentence explanation of your score>"
}}

CRITICAL: Return ONLY raw JSON, nothing else. Do not include markdown codeblocks.
"""
    try:
        # Construct raw Ollama generate request
        req_data = json.dumps({
            "model": judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{OLLAMA_API_URL}/generate",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_bytes = response.read()
            resp_data = json.loads(resp_bytes.decode())
            content = resp_data.get("response", "").strip()
            
            # Clean JSON markdown if model emitted it
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            parsed = json.loads(content)
            score = float(parsed.get("score", 3.0))
            return max(1.0, min(5.0, score))
    except Exception as e:
        # Fallback to simple keyword/keyphrase match density if judge fails or timeouts
        print(f"    [!] Auto-judge call failed ({e}). Falling back to rule-based keyword score.")
        matches = sum(1 for phrase in expected_phrases if phrase.lower() in answer.lower())
        pct = matches / len(expected_phrases) if expected_phrases else 0.0
        # scale percentage (0.0 - 1.0) to (1.0 - 5.0) range
        return 1.0 + (pct * 4.0)

def get_ollama_embedding(text: str, model: str = "nomic-embed-text:latest") -> list[float]:
    """Fetches a 768-dimensional embedding from local Ollama model."""
    try:
        req_data = json.dumps({
            "model": model,
            "prompt": text
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{OLLAMA_API_URL}/embeddings",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            resp_bytes = response.read()
            resp_data = json.loads(resp_bytes.decode())
            return resp_data.get("embedding", [])
    except Exception as e:
        return []

def calculate_semantic_similarity(answer: str, ground_truth: str) -> float:
    """Calculates cosine similarity between two texts using nomic-embed-text."""
    v1 = get_ollama_embedding(answer)
    v2 = get_ollama_embedding(ground_truth)
    if not v1 or not v2:
        return 0.0
        
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (norm_v1 * norm_v2)))

def evaluate_agent(
    repo_path: str,
    models: list[str],
    prompt_versions: list[str],
    dataset_path: str = "evaluation/dataset.json",
    quick_mode: bool = False,
    judge_model: str = "qwen2.5-coder:14b"
) -> str:
    """Runs the full comparative benchmark grid and saves results."""
    # 1. Load Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    with open(dataset_path, 'r') as f:
        queries = json.load(f)
        
    if quick_mode:
        # Dynamically select the first query for each programming language to ensure a balanced quick run
        by_lang = {}
        for q in queries:
            lang = q.get("language", "unknown")
            if lang not in by_lang:
                by_lang[lang] = q
        queries = list(by_lang.values())
        print(f"[!] Quick Mode enabled. Using a balanced representative set of {len(queries)} queries (1 per language): {[q['id'] for q in queries]}")
        
    # 2. Results Directory
    results_dir = os.path.join(config.BASE_DIR, "data", "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    # 3. Grid Search Run
    for model in models:
        print(f"\n==========================================")
        print(f"[*] BENCHMARKING MODEL: {model}")
        print(f"==========================================")
        
        try:
            set_agent_model(model)
        except Exception as e:
            print(f"[!] Skipped model {model}: Could not set model ({e})")
            continue
            
        for prompt_ver in prompt_versions:
            print(f"\n---> Prompt Version: {prompt_ver}")
            
            for q_idx, q_item in enumerate(queries):
                q_id = q_item["id"]
                query_text = q_item["query"]
                expected_tools = q_item.get("expected_tools", [])
                expected_phrases = q_item.get("key_phrases", [])
                
                print(f"  [{q_idx+1}/{len(queries)}] Query {q_id}: '{query_text[:50]}...'")
                
                # Assemble system + human state
                sys_content = get_system_prompt(repo_path, prompt_ver, query_text)
                messages = [
                    SystemMessage(content=sys_content),
                    HumanMessage(content=query_text)
                ]
                
                attempts = 0
                max_retries = 2
                while attempts <= max_retries:
                    start_time = time.time()
                    tools_called = []
                    final_answer = ""
                    success = True
                    error_msg = ""
                    
                    # Stream the state machine and monitor node transitions
                    try:
                        for event in agent_app.stream({"messages": messages}):
                            for node_name, node_state in event.items():
                                last_msg = node_state['messages'][-1]
                                
                                # Log tool executions
                                if node_name == "agent" and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        t_name = tc.get('name')
                                        if t_name and t_name not in tools_called:
                                            tools_called.append(t_name)
                                            print(f"    [Tool Call] Agent requested: {t_name}")
                                            
                                if last_msg.content:
                                    final_answer = last_msg.content
                        break # Break loop on success
                    except Exception as e:
                        attempts += 1
                        success = False
                        error_msg = str(e)
                        if attempts <= max_retries:
                            print(f"    [!] Graph run crashed ({e}). Retrying ({attempts}/{max_retries})...")
                            time.sleep(2)
                        else:
                            print(f"    [!] Graph run crashed permanently after {attempts} attempts.")
                    
                latency = time.time() - start_time
                print(f"    Latency: {latency:.2f}s | Success: {success}")
                
                # Metrics scoring
                # 1. Tool Call Reliability Score: Routing Accuracy (binary 1.0 or 0.0)
                called_set = set(tools_called)
                expected_set = set(expected_tools)
                matched_tools = called_set.intersection(expected_set)
                
                tool_reliability = 0.0
                if expected_set:
                    # 1.0 if the agent called at least one acceptable routing tool, else 0.0
                    tool_reliability = 1.0 if matched_tools else 0.0
                    
                # 2. Semantic Similarity Score via local neural embeddings
                phrase_coverage = calculate_semantic_similarity(final_answer, query_text) if success and final_answer else 0.0
                
                # 3. LLM Auto-judge score (1.0 to 5.0)
                accuracy_score = 1.0
                if success and final_answer:
                    # Use Qwen to judge answer quality
                    accuracy_score = run_auto_judge(
                        query=query_text,
                        answer=final_answer,
                        expected_phrases=expected_phrases,
                        judge_model=judge_model
                    )
                
                print(f"    Tool Match: {len(matched_tools)}/{len(expected_tools)} | Coverage: {phrase_coverage*100:.1f}% | Judge: {accuracy_score}/5.0")
                
                # Save result
                results.append({
                    "model": model,
                    "prompt_version": prompt_ver,
                    "query_id": q_id,
                    "language": q_item.get("language", "unknown"),
                    "query": query_text,
                    "success": success,
                    "error": error_msg,
                    "latency_seconds": latency,
                    "tools_called": tools_called,
                    "expected_tools": expected_tools,
                    "tool_reliability_score": tool_reliability,
                    "key_phrases": expected_phrases,
                    "key_phrase_coverage": phrase_coverage,
                    "accuracy_score": accuracy_score,
                    "response": final_answer,
                    "timestamp": datetime.now().isoformat()
                })
                
    # 4. Save results to disk
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(results_dir, f"eval_run_{run_timestamp}.json")
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[*] Evaluation complete! Results saved to: {out_file}")
    return out_file
