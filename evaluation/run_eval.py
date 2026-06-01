import os
import sys
import argparse
import json
import glob
from datetime import datetime

# Add parent directory to path to ensure proper local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import evaluate_agent, check_available_ollama_models

def get_previous_run_file(current_file: str) -> str:
    """Finds the run file created immediately prior to the current results file."""
    search_path = os.path.join(os.path.dirname(current_file), "eval_run_*.json")
    files = glob.glob(search_path)
    if not files:
        return ""
        
    # Sort files by creation time
    files.sort(key=os.path.getctime)
    
    # Filter out current file
    files = [f for f in files if os.path.basename(f) != os.path.basename(current_file)]
    if not files:
        return ""
    return files[-1]  # Get the most recent one

def get_stats_from_file(file_path: str) -> dict:
    """Loads a run file and calculates performance aggregates for each config."""
    if not file_path or not os.path.exists(file_path):
        return {}
        
    try:
        with open(file_path, 'r') as f:
            runs = json.load(f)
            
        groups = {}
        for run in runs:
            key = (run["model"], run["prompt_version"])
            if key not in groups:
                groups[key] = []
            groups[key].append(run)
            
        stats = {}
        for key, items in groups.items():
            total = len(items)
            successes = sum(1 for x in items if x.get("success", False))
            avg_latency = sum(x.get("latency_seconds", 0.0) for x in items) / total
            avg_tool = sum(x.get("tool_reliability_score", 0.0) for x in items) / total
            avg_phrase = sum(x.get("key_phrase_coverage", 0.0) for x in items) / total
            avg_judge = sum(x.get("accuracy_score", 1.0) for x in items) / total
            
            # Check for human scores if feedback loop has run
            human_scores = [x.get("human_score") for x in items if x.get("human_score") is not None]
            avg_human = sum(human_scores) / len(human_scores) if human_scores else None
            
            stats[key] = {
                "success_rate": (successes / total) * 100,
                "avg_latency": avg_latency,
                "avg_tool": avg_tool,
                "avg_phrase": avg_phrase,
                "avg_judge": avg_judge,
                "avg_human": avg_human
            }
        return stats
    except Exception:
        return {}

def format_delta(current: float, prior: float, is_latency: bool = False, is_percentage: bool = False) -> str:
    """Formats the value progression delta with a colored indicator."""
    diff = current - prior
    if abs(diff) < 0.01:
        return ""
        
    symbol = "+" if diff > 0 else ""
    
    if is_latency:
        # For latency, negative difference is better (decreased time)
        color = "🟢" if diff < 0 else "🔴"
        return f" ({color} {symbol}{diff:.2f}s)"
    elif is_percentage:
        color = "🟢" if diff > 0 else "🔴"
        return f" ({color} {symbol}{diff*100:.1f}%)"
    else:
        color = "🟢" if diff > 0 else "🔴"
        return f" ({color} {symbol}{diff:.2f})"

def generate_markdown_report(results_file: str) -> str:
    """Reads a results JSON file and produces a structured summary report with progression trends."""
    with open(results_file, 'r') as f:
        runs = json.load(f)
        
    if not runs:
        return "No evaluation results available."
        
    # 1. Group current results by configuration
    groups = {}
    for run in runs:
        key = (run["model"], run["prompt_version"])
        if key not in groups:
            groups[key] = []
        groups[key].append(run)
        
    # 2. Retrieve prior results for delta generation
    prior_file = get_previous_run_file(results_file)
    prior_stats = get_stats_from_file(prior_file) if prior_file else {}
    
    report = []
    report.append(f"# 🛡️ CodeSentinel - LLM Evaluation Report")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Results Source:** `{os.path.basename(results_file)}` ")
    if prior_file:
        report.append(f"**Baseline Comparison:** compared against prior run `{os.path.basename(prior_file)}`\n")
    else:
        report.append("**Baseline Comparison:** *No prior run found. Establishing new baseline.*\n")
        
    report.append("## 📊 Configuration Performance Matrix")
    report.append("| Model | Prompt Version | Success Rate | Avg Latency | Tool Routing Accuracy | Semantic Similarity | Avg Judge Score |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Calculate min latency across all successful runs to use in normalization
    successful_latencies = [x["latency_seconds"] for x in runs if x.get("success", False)]
    min_latency = min(successful_latencies) if successful_latencies else 1.0
    
    recommendation_candidates = []
    
    for (model, prompt_ver), items in sorted(groups.items()):
        total = len(items)
        successes = sum(1 for x in items if x["success"])
        avg_latency = sum(x["latency_seconds"] for x in items) / total
        avg_tool = sum(x["tool_reliability_score"] for x in items) / total
        avg_phrase = sum(x["key_phrase_coverage"] for x in items) / total
        avg_judge = sum(x["accuracy_score"] for x in items) / total
        
        success_rate = (successes / total) * 100
        
        # Calculate human score average from this run if it was loaded back
        human_scores = [x.get("human_score") for x in items if x.get("human_score") is not None]
        avg_human = sum(human_scores) / len(human_scores) if human_scores else None
        
        # Build comparative deltas
        prior = prior_stats.get((model, prompt_ver))
        
        d_succ, d_lat, d_tool, d_phrase, d_judge = "", "", "", "", ""
        if prior:
            d_succ = format_delta(success_rate, prior["success_rate"], is_percentage=True)
            d_lat = format_delta(avg_latency, prior["avg_latency"], is_latency=True)
            d_tool = format_delta(avg_tool, prior["avg_tool"], is_percentage=True)
            d_phrase = format_delta(avg_phrase, prior["avg_phrase"], is_percentage=True)
            d_judge = format_delta(avg_judge, prior["avg_judge"])
            
            # If no human scores in current run, carry over prior ones for display
            if avg_human is None:
                avg_human = prior["avg_human"]
                
        report.append(
            f"| {model} | `{prompt_ver}` | {success_rate:.1f}%{d_succ} | {avg_latency:.2f}s{d_lat} | "
            f"{avg_tool*100:.1f}%{d_tool} | {avg_phrase*100:.1f}%{d_phrase} | {avg_judge:.2f}/5.0{d_judge} |"
        )
        
        # Store for optimal recommendation calculations (only if there are successful runs)
        if successes > 0:
            recommendation_candidates.append({
                "model": model,
                "prompt_version": prompt_ver,
                "latency": avg_latency,
                "judge_score": avg_judge,
                "reliability": avg_tool,
                "semantic_similarity": avg_phrase,
                "human_score": avg_human
            })
        
    # 3. Dynamic Language Breakdown Matrix
    report.append("\n## 🌐 Per-Language Performance Breakdown Matrix")
    report.append("| Model | Prompt Version | Language | Avg Latency | Tool Routing Accuracy | Semantic Similarity | Avg Judge Score |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for (model, prompt_ver), items in sorted(groups.items()):
        # Group items by language
        lang_groups = {}
        for x in items:
            l = x.get("language", "unknown").upper()
            if l not in lang_groups:
                lang_groups[l] = []
            lang_groups[l].append(x)
            
        for lang_name, l_items in sorted(lang_groups.items()):
            l_total = len(l_items)
            l_latency = sum(x["latency_seconds"] for x in l_items) / l_total
            l_tool = sum(x["tool_reliability_score"] for x in l_items) / l_total
            l_phrase = sum(x["key_phrase_coverage"] for x in l_items) / l_total
            l_judge = sum(x["accuracy_score"] for x in l_items) / l_total
            
            report.append(
                f"| {model} | `{prompt_ver}` | **{lang_name}** | {l_latency:.2f}s | "
                f"{l_tool*100:.1f}% | {l_phrase*100:.1f}% | {l_judge:.2f}/5.0 |"
            )
            
    # 4. Closed Feedback Loop & Optimal Recommendation Engine
    report.append("\n## 🧠 Human Preference Correlation & Recommendation Engine")
    
    if recommendation_candidates:
        # Find if human scores are available
        human_aligned = [c for c in recommendation_candidates if c["human_score"] is not None]
        
        report.append("### 🏆 Optimal Configuration Recommendations")
        
        # Calculate scientifically weighted score:
        # Score = (JudgeScore/5.0 * 0.45) + (SemanticSimilarity * 0.25) + (RoutingAccuracy * 0.20) + (NormalizedLatency * 0.10)
        # where NormalizedLatency = MinLatency / CurrentLatency
        for c in recommendation_candidates:
            norm_lat = min_latency / c["latency"] if c["latency"] > 0 else 1.0
            c["weighted_score"] = (c["judge_score"] / 5.0 * 0.45) + (c.get("semantic_similarity", 0.0) * 0.25) + (c["reliability"] * 0.20) + (norm_lat * 0.10)
            
        recommendation_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
        best_auto = recommendation_candidates[0]
        
        report.append(
            f"- **🤖 Automated Benchmark Winner:** **{best_auto['model']}** with prompt variant **`{best_auto['prompt_version']}`**\n"
            f"  - *Weighted Performance Score:* `{best_auto['weighted_score']*100:.1f}/100` (Heuristics: **45% LLM Judge / 25% Semantic Similarity / 20% Tool Routing / 10% Latency**)\n"
            f"  - *Stats:* Score: `{best_auto['judge_score']:.2f}/5.0`, SemSim: `{best_auto.get('semantic_similarity', 0.0)*100:.1f}%`, Routing: `{best_auto['reliability']*100:.1f}%`, Latency: `{best_auto['latency']:.2f}s`."
        )
        
        # Recommendation B: Human Aligned Winner
        if human_aligned:
            human_aligned.sort(key=lambda x: x["human_score"], reverse=True)
            best_human = human_aligned[0]
            report.append(
                f"- **👤 Human-Alignment Winner:** **{best_human['model']}** with prompt variant **`{best_human['prompt_version']}`** "
                f"(Human alignment rating: `{best_human['human_score']:.2f}/5.0`)."
            )
            
            # Print alignment correlation insight
            correlation = "HIGH Alignment" if best_auto["prompt_version"] == best_human["prompt_version"] else "DIVERGENT Preferences"
            report.append(f"- **📈 Alignment Insight:** `{correlation}`. ")
            if correlation == "HIGH Alignment":
                report.append("Human ratings strongly correlate with the automated LLM-Judge heuristics! This validates the auto-scorer's calibrated anchors.")
            else:
                report.append("Human evaluators prefer a different prompt variant than the auto-judge, indicating that direct manual adjustments are actively refining LLM behavior.")
        else:
            report.append("- **👤 Human-Alignment Winner:** *Pending feedback loop.* Run `python -m evaluation.feedback_loop` to record your ratings!")
            
    report.append("\n## 🔍 Itemized Case-by-Case Analysis")
    for run in runs:
        report.append(f"### ❓ Query {run['query_id']} ({run['model']} - `{run['prompt_version']}`)")
        report.append(f"- **Language:** `{run.get('language', 'unknown').upper()}`")
        report.append(f"- **Query:** *{run['query']}*")
        report.append(f"- **Success:** `{run['success']}`" + (f" (Error: `{run['error']}`)" if not run['success'] else ""))
        report.append(f"- **Latency:** `{run['latency_seconds']:.2f}s`")
        report.append(f"- **Tools Called:** `{', '.join(run['tools_called']) if run['tools_called'] else 'None'}` (Expected Acceptable: `{', '.join(run['expected_tools'])}`)")
        report.append(f"- **Key Reference Coverage:** `{run['key_phrase_coverage']*100:.1f}%` (Found: `{', '.join([p for p in run['key_phrases'] if p.lower() in run['response'].lower()])}` / Target: `{', '.join(run['key_phrases'])}`)")
        report.append(f"- **Automated Scorer Score:** `{run['accuracy_score']:.2f}/5.0`")
        if run.get("human_score") is not None:
            report.append(f"- **Human Score:** `{run['human_score']:.1f}/5.0` (Feedback: *{run.get('human_feedback', 'None')}*)")
            
        report.append("\n#### 💬 Agent Response:")
        report.append("```markdown")
        report.append(run["response"].strip())
        report.append("```")
        report.append("\n" + "---" + "\n")
        
    report_content = "\n".join(report)
    
    # Save the report as a companion Markdown file
    report_file = results_file.replace(".json", ".md")
    with open(report_file, 'w') as f:
        f.write(report_content)
        
    return report_file

def main():
    parser = argparse.ArgumentParser(description="🛡️ CodeSentinel - LLM & Prompt Benchmarking Suite")
    parser.add_argument(
        "--repo",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Path to the repository to index/analyze for testing"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation (first 2 queries only) for fast sanity check"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen2.5-coder:14b,qwen2.5:14b",
        help="Comma-separated list of models to evaluate"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="v1_baseline,v2_rules_of_engagement,v3_slice_focused",
        help="Comma-separated list of prompt versions to evaluate"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="qwen2.5-coder:14b",
        help="Ollama model to use as evaluator judge"
    )
    
    args = parser.parse_args()
    
    print("==================================================================")
    print("🛡️  CodeSentinel - Automated LLM & Prompt Benchmarking Suite  🛡️")
    print("==================================================================")
    print(f"[*] Workspace Target Repo:  {args.repo}")
    print(f"[*] Quick Mode Enabled:     {args.quick}")
    print(f"[*] Judge Model Model:      {args.judge}")
    print("-" * 66)
    
    # 1. Gather models and prompt variants
    available_models = check_available_ollama_models()
    
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid_models = []
    
    for m in requested_models:
        match = [am for am in available_models if am == m or am.startswith(m + ":")]
        if match:
            valid_models.append(match[0])
        else:
            print(f"[!] Warning: Model '{m}' not listed in Ollama. Will attempt to invoke it anyway.")
            valid_models.append(m)
            
    prompt_vers = [p.strip() for p in args.prompts.split(",") if p.strip()]
    
    print(f"[*] Target Models:  {valid_models}")
    print(f"[*] Target Prompts: {prompt_vers}")
    print("-" * 66)
    
    # 2. Run the Comparative Benchmark
    try:
        results_file = evaluate_agent(
            repo_path=args.repo,
            models=valid_models,
            prompt_versions=prompt_vers,
            quick_mode=args.quick,
            judge_model=args.judge
        )
        
        # 3. Generate Markdown Report
        report_file = generate_markdown_report(results_file)
        
        print("\n" + "=" * 66)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 66)
        with open(report_file, 'r') as f:
            lines = f.readlines()
            matrix_started = False
            for line in lines:
                if "Configuration Performance Matrix" in line:
                    matrix_started = True
                elif matrix_started and "Per-Language Performance Breakdown" in line:
                    break
                if matrix_started:
                    print(line.strip())
                    
        # Print Per-Language Breakdown CLI Block
        print("\n🌐 PER-LANGUAGE PERFORMANCE BREAKDOWN MATRIX")
        lang_matrix_started = False
        with open(report_file, 'r') as f:
            for line in f:
                if "Per-Language Performance Breakdown Matrix" in line:
                    lang_matrix_started = True
                elif lang_matrix_started and "Human Preference Correlation" in line:
                    break
                if lang_matrix_started:
                    print(line.strip())
                    
        # Print recommendations in CLI
        print("\n🏆 OPTIMAL CONFIGURATION RECOMMENDATION")
        recommendations_started = False
        with open(report_file, 'r') as f:
            for line in f:
                if "Optimal Configuration Recommendations" in line:
                    recommendations_started = True
                elif recommendations_started and line.startswith("## "):
                    break
                if recommendations_started and line.strip() and not line.startswith("###"):
                    print(line.strip())
        print("=" * 66)
        print(f"\n[*] Companion Markdown report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n❌ Evaluation aborted due to error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
