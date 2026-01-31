"""
Graph construction for the LTM application.
Uses MongoDBSaver for persistent conversation history when available,
falls back to MemorySaver for testing.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from core.state import State
from core.agent import agent, load_memories, route_tools, crisis_node
from core.tools import all_tools
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("✅ Graph Builder initializing...")


def _get_checkpointer():
    """Get the best available checkpointer for conversation persistence."""
    
    # Try MongoDB first
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient
        
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Test connection
        client.server_info()
        
        checkpointer = MongoDBSaver(client=client)
        print("✅ Using MongoDB for conversation persistence")
        return checkpointer
    except ImportError:
        print("⚠️ langgraph.checkpoint.mongodb not available")
    except Exception as e:
        print(f"⚠️ MongoDB checkpointer failed: {e}")
    
    # Fallback to in-memory saver
    try:
        from langgraph.checkpoint.memory import MemorySaver
        print("⚠️ Using in-memory checkpointer (conversations won't persist across restarts)")
        return MemorySaver()
    except ImportError:
        print("⚠️ MemorySaver not available")
    
    # No checkpointer available
    print("⚠️ No checkpointer available - conversations will not be saved")
    return None


def build_graph(model_with_tools):
    """Build the conversation graph with memory capabilities.

    Args:
        model_with_tools: The language model with bound tools

    Returns:
        Compiled graph ready for execution
    """
    print("Loading Graph...")
    
    # Create the graph and add nodes
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("load_memories", load_memories)
    builder.add_node("agent", lambda state, config: agent(state, config, model_with_tools))
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("crisis_node", crisis_node)

    # Add edges to the graph
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    builder.add_conditional_edges("agent", route_tools, ["tools", "crisis_node", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("crisis_node", END)

    # Get checkpointer and compile
    checkpointer = _get_checkpointer()
    
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    else:
        return builder.compile()


def pretty_print_stream_chunk(chunk):
    """Format and print stream chunks from the graph execution.
    
    Args:
        chunk: A chunk of data from the graph's stream method
    """
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)
        print("\n")