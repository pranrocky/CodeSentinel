import os
import json
import glob
import sys

def get_latest_eval_file() -> str:
    """Finds the absolute path to the latest evaluation JSON file."""
    search_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_results", "eval_run_*.json")
    files = glob.glob(search_path)
    if not files:
        return ""
    return max(files, key=os.path.getctime)

def main():
    print("==================================================================")
    print("🛡️  CodeSentinel - Closed Human-Judgment & Scorer Feedback Loop  🛡️")
    print("==================================================================")
    
    eval_file = get_latest_eval_file()
    if not eval_file:
        print("[!] No evaluation files found in data/eval_results/.")
        print("[*] Please run the evaluation suite first:")
        print("    python -m evaluation.run_eval --repo <path>")
        sys.exit(1)
        
    print(f"[*] Loading latest evaluation run: {os.path.basename(eval_file)}")
    
    with open(eval_file, 'r') as f:
        results = json.load(f)
        
    print(f"[*] Loaded {len(results)} evaluated items.")
    print("-" * 66)
    
    updated = False
    
    for i, item in enumerate(results):
        print(f"\n[Item {i+1}/{len(results)}] Comparative Case:")
        print(f"  🤖 Model:          {item['model']}")
        print(f"  📝 Prompt Version: {item['prompt_version']}")
        print(f"  ❓ User Query:     {item['query']}")
        print(f"  🛠️  Tools Called:    {', '.join(item['tools_called']) if item['tools_called'] else 'None'} (Expected: {', '.join(item['expected_tools'])})")
        print(f"  📈 Auto-Judge Score: {item['accuracy_score']}/5.0 (Coverage: {item['key_phrase_coverage']*100:.1f}%)")
        print(f"  💬 Response:")
        print("-" * 50)
        
        # Trim response for readability if it's very long
        resp = item['response']
        if len(resp) > 600:
            print(resp[:600] + "\n... [TRUNCATED] ...")
        else:
            print(resp)
        print("-" * 50)
        
        # Current human values if already rated
        cur_human_score = item.get("human_score", "None")
        cur_human_feedback = item.get("human_feedback", "None")
        print(f"  Current Human Score:    {cur_human_score}")
        print(f"  Current Human Feedback: {cur_human_feedback}")
        
        # Prompt for human rating
        try:
            score_input = input("\n👉 Enter Human Rating (1-5, or press Enter to skip/keep): ").strip()
            if score_input:
                try:
                    h_score = float(score_input)
                    if 1.0 <= h_score <= 5.0:
                        item["human_score"] = h_score
                        updated = True
                    else:
                        print("[!] Invalid score. Must be between 1.0 and 5.0.")
                        continue
                except ValueError:
                    print("[!] Invalid input. Must be a number.")
                    continue
            
            feedback_input = input("👉 Enter qualitative feedback/guidance (optional, press Enter to skip): ").strip()
            if feedback_input:
                item["human_feedback"] = feedback_input
                updated = True
                
        except KeyboardInterrupt:
            print("\n[!] Exiting human-judgment loop early.")
            break
            
    if updated:
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[*] Success! Human scores and alignment logs updated inside: {eval_file}")
    else:
        print("\n[*] No human updates made. Exiting.")

if __name__ == "__main__":
    main()
