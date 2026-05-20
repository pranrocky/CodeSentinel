import json
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from agent.state import AgentState
import config

# Import all tools
from tools.kdg_tool import query_kdg
from tools.rag_tool import rag_query
from tools.slicer_tool import get_call_slice
from tools.linter_tool import run_linter
from tools.symbol_tool import search_symbol
from tools.trace_tool import explain_trace
from tools.fix_tool import suggest_fix
from tools.grep_tool import universal_grep
from tools.explorer_tool import list_repo_files, read_file
from tools.repo_map_tool import read_entire_codebase

print("[*] Initializing LangGraph Agent...")

# 1. Initialize the LLM
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    temperature=0.1
)

# 2. Bind the tools to the LLM
tools_list = [
    query_kdg,          # always try KDG first — cheapest path
    rag_query, get_call_slice, run_linter,
    search_symbol, explain_trace, suggest_fix,
    universal_grep, list_repo_files, read_file,
    read_entire_codebase
]
llm_with_tools = llm.bind_tools(tools_list)

# 3. Define the Agent Node (With Ollama Fallback Parser)
def run_agent(state: AgentState):
    """The node where the LLM evaluates the state and decides what to do."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    
    # --- OLLAMA FALLBACK PARSER ---
    if not response.tool_calls and response.content:
        try:
            clean_text = response.content.strip().strip('`').removeprefix('json')
            parsed = json.loads(clean_text)
            
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                print(f"[!] Intercepted raw JSON tool call for: {parsed['name']}")
                response.tool_calls = [{
                    "name": parsed["name"],
                    "args": parsed["arguments"],
                    "id": "call_ollama_fallback", 
                    "type": "tool_call"
                }]
                response.content = ""
        except json.JSONDecodeError:
            pass
    # ------------------------------
            
    return {"messages": [response]}

# 4. Define the Tool Node
tool_node = ToolNode(tools_list, handle_tool_errors=True)

# 5. Define the Routing Logic
def should_continue(state: AgentState):
    """Determines if we need to execute a tool or return to the user."""
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "continue"
    return "end"

# 6. Build and Compile the Graph
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", run_agent)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Export the compiled agent
agent_app = create_graph()