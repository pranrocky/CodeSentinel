import os
import subprocess
from langchain_core.tools import tool

_GRAPHIFY_CMD = ["conda", "run", "-n", "codesen", "graphify"]


@tool
def query_kdg(query: str, repo_path: str) -> str:
    """
    Query the knowledge dependency graph of the repository using graphify.
    This is 70x cheaper than reading raw files and should be your FIRST choice
    for any architecture, dependency, call-chain, or structural question.

    Args:
        query: Natural language question about the codebase structure or relationships.
        repo_path: The absolute path to the cloned repository workspace.
    """
    print(f"[Tool: KDG Query] '{query}'")

    graph_path = os.path.join(repo_path, "graphify-out", "graph.json")
    if not os.path.exists(graph_path):
        return (
            "Knowledge graph not found for this repository. "
            "It may still be building, or ingestion did not complete successfully."
        )

    try:
        result = subprocess.run(
            _GRAPHIFY_CMD + ["query", query, "--graph", graph_path, "--budget", "2000"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        return output if output else "Knowledge graph returned no results for this query."
    except subprocess.TimeoutExpired:
        return "KDG query timed out. Try a more specific query."
    except Exception as e:
        return f"KDG query failed: {e}"
