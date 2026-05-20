from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools.rag_tool import rag_query
import config

# Connect LangChain to your local Ollama instance
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # Required by the SDK, but Ollama ignores it
    temperature=0.2    # Low temperature for analytical accuracy
)

@tool
def explain_trace(stacktrace: str) -> str:
    """
    Explain the root cause of a Python stack trace.
    Retrieves relevant code and explains why the error occurred.
    
    Args:
        stacktrace: the full Python traceback string
    """
    print("[Tool: Explain Trace] Analyzing stack trace...")
    
    # We use the RAG tool internally to fetch the code that crashed
    # We slice the stacktrace to 300 chars to avoid overwhelming the embedding model
    context = rag_query.invoke({"query": f"code related to: {stacktrace[:300]}"})
    
    # Build the prompt for the internal LLM call
    prompt = f"""
    Stack trace:
    {stacktrace}
    
    Relevant codebase context:
    {context}
    
    Analyze the stack trace and the codebase context. 
    Explain the root cause of the error in 3-4 sentences.
    """
    
    # Call the LLM and return its string response
    response = llm.invoke(prompt)
    return response.content

# --- Execution Test Block ---
if __name__ == "__main__":
    # A fake stack trace for a common error
    fake_trace = """
    Traceback (most recent call last):
      File "/app/main.py", line 14, in <module>
        process_data(None)
      File "/app/utils.py", line 45, in process_data
        return data.split(",")
    AttributeError: 'NoneType' object has no attribute 'split'
    """
    
    print("\nExecuting Trace Explainer Tool...")
    # This will trigger RAG, then hit your Qwen model
    explanation = explain_trace.invoke({"stacktrace": fake_trace})
    print("\n" + "="*50)
    print(explanation)
    print("="*50)