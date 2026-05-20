from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The `add_messages` reducer is critical. It tells LangGraph to APPEND 
    # new messages to the list rather than overwriting the whole state every turn.
    messages: Annotated[list[AnyMessage], add_messages]