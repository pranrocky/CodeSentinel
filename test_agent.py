from langchain_core.messages import HumanMessage, SystemMessage
from agent.graph import agent_app
from agent.prompts import get_system_prompt


# We will use the exact temp path from your screenshot to test it realistically
repo_path = "/tmp/tmpc7sy8rdf" 

print(f"[*] Starting CLI E2E Test for repo: {repo_path}")

# 1. Setup the exact same state the UI uses
sys_prompt = get_system_prompt(repo_path)
messages = [
    SystemMessage(content=sys_prompt),
    HumanMessage(content="Trace the execution of the url_for function. What internal functions does it call to actually build the URL?")
]

print("\n[*] Streaming LangGraph State Machine...\n")

# 2. Run the graph and print the raw internal state
try:
    for event in agent_app.stream({"messages": messages}):
        for node_name, node_state in event.items():
            print(f"=== TRANSITION TO NODE: {node_name.upper()} ===")
            
            last_msg = node_state['messages'][-1]
            
            # Print the raw text content
            print(f"CONTENT:\n{last_msg.content}")
            
            # Critically: check if LangChain successfully parsed the tool call
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"\nTOOL_CALLS DETECTED: {last_msg.tool_calls}")
            else:
                print("\nTOOL_CALLS DETECTED: NONE (This will cause routing to END)")
            print("-" * 50)
            
except Exception as e:
    print(f"Graph crashed: {e}")