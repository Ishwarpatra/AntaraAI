"""
State definitions for the LTM application conversation graph.
"""

from typing import List
from langgraph.graph import MessagesState

class State(MessagesState):
    """State definition for the conversation graph with memory capabilities."""
    recall_memories: List[str]