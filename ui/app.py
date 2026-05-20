import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import our pipeline pieces
from ingestion.loader import load_repo
from ingestion.chunker import chunk_code
from ingestion.indexer import build_and_persist_index, build_knowledge_graph
from agent.graph import agent_app
from agent.prompts import get_system_prompt

def ingest_repository(source):
    """Handles the downloading, chunking, and embedding of the codebase."""
    yield f"[*] Creating workspace and loading repository from {source}...", None
    try:
        docs, repo_path = load_repo(source)
        yield f"[*] Loaded {len(docs)} Python files. Chunking AST boundaries...", repo_path
        
        nodes = chunk_code(docs)
        yield f"[*] Chunked into {len(nodes)} distinct nodes. Building FAISS vector database...", repo_path
        
        build_and_persist_index(nodes)
        yield f"[*] FAISS index built. Generating knowledge graph...", repo_path

        build_knowledge_graph(repo_path)
        yield f"[*] Success! Repository indexed + knowledge graph ready.\n(Workspace: {repo_path})", repo_path
    except Exception as e:
        yield f"❌ Error during ingestion: {str(e)}", None

def chat_with_agent(user_message, history, repo_path):
    """Routes the chat history and the current user message through LangGraph."""
    if history is None:
        history = []
        
    if not repo_path:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ **Error:** Please ingest a repository first using the panel on the left!"})
        yield history
        return

    # 1. Build the system context
    system_instruction = get_system_prompt(repo_path)
    messages = [SystemMessage(content=system_instruction)]
    
    # 2. Append the chat history
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    # 3. Append the new user question
    messages.append(HumanMessage(content=user_message))
    history.append({"role": "user", "content": user_message})
    
    bot_response = ""
    history.append({"role": "assistant", "content": bot_response})
    yield history
    
    # 4. Stream the LangGraph execution
    try:
        for event in agent_app.stream({"messages": messages}):
            for node_name, node_state in event.items():
                last_msg = node_state['messages'][-1]
                
                if node_name == "agent":
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_name = last_msg.tool_calls[0]['name']
                        bot_response += f"\n> 🧠 **Agent Thought:** I need to use the `{tool_name}` tool...\n"
                        history[-1]["content"] = bot_response
                        yield history
                    elif last_msg.content:
                        bot_response += f"\n\n{last_msg.content}"
                        history[-1]["content"] = bot_response
                        yield history
                        
                elif node_name == "tools":
                    bot_response += f"> 🛠️ **Tool Output Received.** Analyzing results...\n"
                    history[-1]["content"] = bot_response
                    yield history
                    
    except Exception as e:
        history[-1]["content"] += f"\n❌ Pipeline crashed: {str(e)}"
        yield history

# --- The Gradio Interface ---
# FIX 1: Removed `theme` from the Blocks constructor
with gr.Blocks(title="CodeSentinel") as demo:
    gr.Markdown("# 🛡️ CodeSentinel: Agentic Code Review")
    gr.Markdown("RAG + Tool-augmented reasoning for local Python repositories.")
    
    repo_path_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ingest Codebase")
            repo_input = gr.Textbox(label="GitHub URL or Local .zip Path", placeholder="https://github.com/pallets/flask")
            ingest_btn = gr.Button("Build Knowledge Base", variant="primary")
            ingest_status = gr.Textbox(label="Ingestion Status", lines=5, interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Analysis & Review")
            # FIX 2: Removed `type="messages"` because it's now the automatic default in Gradio 6
            chatbot = gr.Chatbot(height=500)
            msg_input = gr.Textbox(label="Ask CodeSentinel to find bugs, explain features, or trace logic...", placeholder="e.g., Can you find where the Flask class is defined?")
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.ClearButton([msg_input, chatbot])

    ingest_btn.click(
        fn=ingest_repository,
        inputs=[repo_input],
        outputs=[ingest_status, repo_path_state]
    )
    
    msg_input.submit(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, repo_path_state],
        outputs=[chatbot]
    ).then(lambda: "", None, msg_input) 
    
    submit_btn.click(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, repo_path_state],
        outputs=[chatbot]
    ).then(lambda: "", None, msg_input)

if __name__ == "__main__":
    # FIX 3: Moved `theme` down into the launch method
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())