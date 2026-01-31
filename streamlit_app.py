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

    # Live Session Controls
    try:
        if st.session_state.get('service') and st.session_state.service.live_session_manager:
            with st.sidebar.expander("Live Session", expanded=False):
                st.write("### Live Video/Audio Session")

                if st.button("Start Live Session"):
                    if user_id:
                        session_id = st.session_state.service.start_live_session(user_id)
                        if session_id:
                            st.session_state['live_session_id'] = session_id
                            st.success(f"Live session started: {session_id}")
                        else:
                            st.error("Failed to start live session")
                    else:
                        st.error("Please select a user first!")

                if st.session_state.get('live_session_id'):
                    if st.button("Stop Live Session"):
                        success = st.session_state.service.stop_live_session(st.session_state['live_session_id'])
                        if success:
                            del st.session_state['live_session_id']
                            st.success("Live session stopped")
                        else:
                            st.error("Failed to stop live session")

                    # Show active session info
                    active_sessions = st.session_state.service.get_active_sessions()
                    if active_sessions:
                        st.write("Active sessions:")
                        for sid, info in active_sessions.items():
                            st.write(f"- {sid}: {'Running' if info['active'] else 'Stopped'}")

    except Exception as e:
        st.sidebar.error(f"Could not manage live sessions: {e}")

    # Integration Status Panel
    try:
        from core.integrations import get_integration_manager
        integration_manager = get_integration_manager()

        with st.sidebar.expander("Integration Status", expanded=False):
            st.write("### Connected Services")
            st.write(f"📞 WhatsApp: {'✅' if integration_manager.whatsapp.is_available() else '❌'}")
            st.write(f"💬 Telegram: {'✅' if integration_manager.telegram.is_available() else '❌'}")
            st.write(f"🏥 EHR: {'✅' if integration_manager.ehr.is_available() else '❌'}")

            if st.button("Test Integrations"):
                test_results = []
                if integration_manager.whatsapp.is_available():
                    test_results.append("WhatsApp connection OK")
                if integration_manager.telegram.is_available():
                    test_results.append("Telegram connection OK")
                if integration_manager.ehr.is_available():
                    test_results.append("EHR connection OK")

                if test_results:
                    st.success("Connections working: " + ", ".join(test_results))
                else:
                    st.info("Configure environment variables to enable integrations")

    except Exception as e:
        st.sidebar.error(f"Could not check integration status: {e}")

    # Mood Tracker Visualization
    try:
        from core.memory_manager import db
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        with st.sidebar.expander("Mood Tracker Dashboard", expanded=True):
            if st.button("Refresh Mood Data"):
                st.rerun()

            # Fetch mood data
            mood_data = list(db["mood_logs"].find(
                {"user_id": user_id} if user_id else {},
                {"_id": 0, "timestamp": 1, "intensity": 1, "mood": 1, "notes": 1}
            ).sort("timestamp", -1))

            if mood_data:
                df = pd.DataFrame(mood_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')  # Sort chronologically for charts

                # Mood trend chart
                st.subheader("Mood Trend Over Time")
                fig_trend = px.line(df, x='timestamp', y='intensity', color='mood',
                                   title="Emotional Intensity Over Time",
                                   labels={'intensity': 'Intensity (1-10)', 'mood': 'Mood'})
                st.plotly_chart(fig_trend, use_container_width=True)

                # Mood distribution pie chart
                st.subheader("Mood Distribution")
                mood_counts = df['mood'].value_counts()
                fig_pie = px.pie(values=mood_counts.values, names=mood_counts.index,
                                title="Distribution of Mood Types")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Weekly average mood
                st.subheader("Weekly Average Mood")
                df['week'] = df['timestamp'].dt.isocalendar().week
                weekly_avg = df.groupby('week')['intensity'].mean().reset_index()
                fig_weekly = px.bar(weekly_avg, x='week', y='intensity',
                                   title="Average Intensity by Week",
                                   labels={'intensity': 'Avg Intensity', 'week': 'Week Number'})
                st.plotly_chart(fig_weekly, use_container_width=True)

                # Recent mood logs table
                st.subheader("Recent Mood Entries")
                recent_df = df[['timestamp', 'mood', 'intensity', 'notes']].tail(10)
                recent_df = recent_df.rename(columns={
                    'timestamp': 'Time',
                    'mood': 'Mood',
                    'intensity': 'Intensity',
                    'notes': 'Notes'
                })
                st.dataframe(recent_df, use_container_width=True)

                # Mood statistics
                st.subheader("Mood Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Highest Intensity", f"{df['intensity'].max()}/10")
                with col2:
                    st.metric("Lowest Intensity", f"{df['intensity'].min()}/10")
                with col3:
                    st.metric("Avg Intensity", f"{df['intensity'].mean():.1f}/10")

            else:
                st.info("No mood logs found yet. Start chatting to track your mood!")

            # Manual mood logging
            st.subheader("Log Current Mood")
            mood_options = ["Happy", "Sad", "Anxious", "Angry", "Neutral", "Calm", "Excited", "Tired"]
            current_mood = st.selectbox("How are you feeling?", mood_options)
            intensity = st.slider("Intensity (1-10)", 1, 10, 5)
            notes = st.text_area("Additional notes (optional)", max_chars=200)

            if st.button("Log Mood"):
                if user_id:
                    # Import here to avoid circular imports
                    from core.tools import log_mood_tool
                    result = log_mood_tool(mood=current_mood, intensity=intensity, notes=notes)
                    st.success(result)
                    st.rerun()
                else:
                    st.error("Please select a user first!")

    except Exception as e:
        st.sidebar.error(f"Could not load mood data: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())

if __name__ == "__main__":
    main()