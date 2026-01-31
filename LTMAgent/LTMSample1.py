"""
Main entry point for the LTM application with advanced memory capabilities.
"""

import uuid
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama

from config.app_config import get_config
from core.tools import all_tools
from core.graph_builder import build_graph, pretty_print_stream_chunk

def main():
    """Main entry point for the LTM application."""
    try:
        model_name = get_config("model_name")
        print(f"Using model: {model_name}")
        
        model = ChatGroq(model=model_name)
        model_with_tools = model.bind_tools(all_tools)
        graph = build_graph(model_with_tools)
        
        user_id = input("Enter user ID (press Enter for auto-generated ID): ").strip()
        if not user_id:
            user_id = str(uuid.uuid4())[:8]
            print(f"Using auto-generated user ID: {user_id}")
            
        thread_id = str(uuid.uuid4())[:8]
        print(f"New conversation thread ID: {thread_id}")
        
        config = {"configurable": {
            "user_id": user_id, 
            "thread_id": thread_id
        }}
        
        print("LTM Agent ready for conversation!")
        try:
            while True:
                user_prompt = str(input("User>>> "))
                
                for chunk in graph.stream({"messages": [("user", user_prompt)]}, config=config):
                    pretty_print_stream_chunk(chunk)
                    
        except KeyboardInterrupt:
            print("\nExiting LTM Agent. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()