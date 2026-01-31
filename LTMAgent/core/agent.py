"""
Agent processing logic for the LTM application.
"""

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from core.state import State
from config.prompt_templates import prompt

def agent(state: State, config: RunnableConfig, model_with_tools) -> dict:
    """Process the current state and generate a response using the LLM.

    Args:
        state (State): The current state of the conversation.
        model_with_tools: The model with bound tools.

    Returns:
        dict: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }

def load_memories(state: State, config: RunnableConfig) -> dict:
    """Load memories for the current conversation.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        dict: The updated state with loaded memories.
    """
    # LangMem tools will handle memory retrieval when the agent needs it
    # No automatic memory loading - agent uses tools on-demand for all memory operations
    return {
        "recall_memories": [],
    }

def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END