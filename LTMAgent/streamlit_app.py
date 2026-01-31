"""
Streamlit web UI for the LTM application.
Uses the LTMService to provide a web interface to the core functionality.
"""

import streamlit as st
from core.service import LTMService
import time

def handle_response(response_chunks):
    """Process response chunks from the service."""
    # Create a placeholder for the ongoing response
    response_placeholder = st.empty()
    
    for chunk in response_chunks:
        for node, updates in chunk.items():
            if node == "agent" and "messages" in updates:
                # Extract and display the assistant's message
                message = updates["messages"][-1].content
                response_placeholder.markdown(message)
                
            # Display other node updates in debug mode
            if st.session_state.get('debug_mode', False):
                with st.expander(f"Debug: Update from {node}"):
                    st.write(updates)

def user_management():
    """Handle user management in the sidebar."""
    st.sidebar.header("User Management")
    
    # Initialize service if not already done
    if 'service' not in st.session_state:
        st.session_state.service = LTMService()
        st.session_state.model_info = st.session_state.service.get_model_info()
    
    # User ID selection
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Option to use existing or create new user ID
    user_option = st.sidebar.radio(
        "Select user option:",
        ["Create new user", "Use existing user"],
        index=0 if st.session_state.user_id is None else 1
    )
    
    if user_option == "Create new user":
        if st.sidebar.button("Generate new user ID"):
            st.session_state.user_id = st.session_state.service.create_user_id()
            st.session_state.thread_id = None  # Reset thread when user changes
    else:
        # This is a placeholder - in a real app, you would fetch actual users
        available_users = st.session_state.service.get_available_users()
        
        # Add the current user to the list if it's not there
        if st.session_state.user_id and st.session_state.user_id not in available_users:
            available_users.append(st.session_state.user_id)
            
        selected_user = st.sidebar.selectbox(
            "Select user ID:",
            available_users
        )
        
        if selected_user != st.session_state.user_id:
            st.session_state.user_id = selected_user
            st.session_state.thread_id = None  # Reset thread when user changes
    
    # Display current user ID
    if st.session_state.user_id:
        st.sidebar.success(f"Current User ID: {st.session_state.user_id}")
    else:
        st.sidebar.warning("No user selected. Please create or select a user.")
        
    return st.session_state.user_id

def thread_management(user_id):
    """Handle thread management in the sidebar."""
    st.sidebar.header("Thread Management")
    
    # Initialize thread ID if needed
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = None
    
    # Option to use existing or create new thread
    thread_option = st.sidebar.radio(
        "Select thread option:",
        ["Create new conversation", "Use existing conversation"],
        index=0 if st.session_state.thread_id is None else 1
    )
    
    if thread_option == "Create new conversation":
        if st.sidebar.button("Start new conversation"):
            st.session_state.thread_id = st.session_state.service.create_thread_id()
            st.session_state.messages = []  # Reset messages for new thread
    else:
        # This is a placeholder - in a real app, you would fetch actual threads
        available_threads = st.session_state.service.get_threads_for_user(user_id)
        
        if st.session_state.thread_id:
            # Add current thread to the list if it's not there
            if not any(thread['id'] == st.session_state.thread_id for thread in available_threads):
                available_threads.append({
                    "id": st.session_state.thread_id, 
                    "created": "Just now", 
                    "title": "Current conversation"
                })
                
        thread_labels = [f"{t['title']} ({t['id']}) - {t['created']}" for t in available_threads]
        selected_thread_label = st.sidebar.selectbox(
            "Select conversation:",
            thread_labels
        ) if available_threads else None
        
        if selected_thread_label:
            selected_thread_id = selected_thread_label.split('(')[1].split(')')[0]
            if selected_thread_id != st.session_state.thread_id:
                st.session_state.thread_id = selected_thread_id
                # In a real app, you would load previous messages for this thread
                st.session_state.messages = []  
    
    # Display current thread ID
    if st.session_state.thread_id:
        st.sidebar.success(f"Current Thread ID: {st.session_state.thread_id}")
    else:
        st.sidebar.warning("No conversation thread selected.")
        
    return st.session_state.thread_id

def model_info():
    """Display model information in the sidebar."""
    st.sidebar.header("Model Information")
    
    if 'model_info' in st.session_state:
        info = st.session_state.model_info
        st.sidebar.info(f"Provider: {info['provider']}\nModel: {info['model_info']}")

def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="LTM Assistant",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 Long-Term Memory Assistant")
    
    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Debug mode toggle
    with st.sidebar.expander("Advanced Settings"):
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
    
    # User and thread management in sidebar
    user_id = user_management()
    thread_id = thread_management(user_id) if user_id else None
    
    # Display model info
    if 'model_info' in st.session_state:
        with st.sidebar.expander("Model Information"):
            info = st.session_state.model_info
            st.write(f"**Provider:** {info['provider']}")
            st.write(f"**Model:** {info['model_name']}")
    
    # Main chat interface
    if user_id and thread_id:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get response from the service
            with st.chat_message("assistant"):
                response_chunks = st.session_state.service.process_message(
                    user_input, user_id, thread_id
                )
                handle_response(response_chunks)
                
                # Wait for the final response to be fully displayed
                time.sleep(0.5)
                
                # Add assistant response to chat history (assuming last chunk contains it)
                # This is a simplified approach - in a real app, you'd aggregate the entire response
                # This is just a placeholder - in a real implementation, you would extract the final
                # message from the response chunks
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Response from the assistant"  # Placeholder
                })
    else:
        # Show instructions if user or thread is not selected
        st.info("Please select a user and conversation thread in the sidebar to start chatting.")

if __name__ == "__main__":
    main()