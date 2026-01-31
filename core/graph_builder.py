"""
Graph construction for the LTM application.
Uses MongoDBSaver for persistent conversation history.
"""

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from core.state import State
from core.agent import agent, load_memories, route_tools, crisis_node
from core.tools import all_tools
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("✅ Graph Builder with MongoDB persistence")

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

    # Compile the graph with MongoDB checkpointer for persistent conversation history
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    memory = MongoDBSaver(client=client)
    return builder.compile(checkpointer=memory)

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