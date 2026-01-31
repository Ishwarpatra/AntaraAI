"""
Service layer for the LTM application.

This module serves as an interface between the UI and the core functionality.
It provides clean, high-level methods that can be used by any UI implementation
(CLI, web, desktop) without needing to understand the internal implementation.
"""

import uuid
from typing import Dict, List, Any, Generator, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config.app_config import get_config
from core.tools import all_tools
from core.graph_builder import build_graph

class LTMService:
    """Service class to manage LTM agent interactions."""

    def __init__(self):
        """Initialize the LTM service."""
        self.model = None
        self.model_with_tools = None
        self.graph = None
        self.live_session_manager = None
        self.wellness_scheduler = None

        # Store model configuration to avoid repeated config calls
        self.model_provider = get_config("model_provider")
        self.model_name = get_config("model_name")
        self.ollama_host = get_config("ollama_host")
        self.ollama_model = get_config("ollama_model")

        try:
            self._initialize_model()
            self._initialize_live_session()
            self._initialize_scheduler()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LTM service: {e}") from e

    def _initialize_live_session(self):
        """Initialize the live session manager for audio/video sessions."""
        try:
            from core.live_session import LiveSessionManager
            from config.system_config import SystemConfig
            
            if SystemConfig.LIVE_SESSION_ENABLED:
                self.live_session_manager = LiveSessionManager()
                print("✅ Live session manager initialized")
            else:
                print("ℹ️ Live sessions disabled via LIVE_SESSION_ENABLED config")
        except ImportError as e:
            print(f"⚠️ Live session dependencies not available: {e}")
            self.live_session_manager = None
        except Exception as e:
            print(f"⚠️ Could not initialize live session manager: {e}")
            self.live_session_manager = None

    def _initialize_scheduler(self):
        """Initialize the wellness scheduler for background tasks."""
        try:
            from core.scheduler import get_wellness_scheduler
            
            self.wellness_scheduler = get_wellness_scheduler(service=self)
            self.wellness_scheduler.start()
            print("✅ Wellness scheduler initialized and started")
        except ImportError as e:
            print(f"⚠️ Scheduler dependencies not available: {e}")
            self.wellness_scheduler = None
        except Exception as e:
            print(f"⚠️ Could not initialize wellness scheduler: {e}")
            self.wellness_scheduler = None

    def schedule_wellness_tasks(self, user_id: str):
        """Schedule wellness tasks (selfie requests, mood check-ins) for a user.
        
        Args:
            user_id: The user ID to schedule tasks for
        """
        if self.wellness_scheduler:
            self.wellness_scheduler.schedule_selfie_request(user_id)
            self.wellness_scheduler.schedule_daily_mood_checkin(user_id)
            return True
        return False

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get the status of the wellness scheduler."""
        if self.wellness_scheduler:
            return self.wellness_scheduler.get_status()
        return {"running": False, "tasks": {}, "users_with_tasks": []}


    def _initialize_model(self):
        """Initialize the language model based on configuration."""
        # Determine which model provider to use
        if self.model_provider == "groq":
            # Use Groq for Llama models
            self.model = ChatGroq(model=self.model_name)
        else:
            # Use Ollama for other models
            self.model = ChatOllama(base_url=self.ollama_host, model=self.ollama_model)

        # Bind tools to the model
        self.model_with_tools = self.model.bind_tools(all_tools)

        # Build the conversation graph
        self.graph = build_graph(self.model_with_tools)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the currently loaded model.

        Returns:
            Dict[str, str]: Dictionary with model information
        """
        if self.model_provider == "groq":
            provider = "Groq"
            model_name = self.model_name
        else:
            provider = "Ollama"
            model_name = self.ollama_model

        return {
            "provider": provider,
            "model_name": model_name
        }

    def get_available_users(self) -> List[str]:
        """Get a list of existing user IDs from memory.

        Queries the database for distinct user_ids.

        Returns:
            List[str]: List of user IDs
        """
        try:
            # Access the MongoDB collection directly to get user IDs
            from core.memory_manager import memory_store
            # This is a simplified approach - in a real implementation,
            # we would query the actual database for user IDs
            # For now, return empty list since we're using InMemoryStore
            return []
        except Exception:
            # Fallback if DB is empty or connection fails
            return []

    def create_user_id(self) -> str:
        """Generate a new user ID.

        Returns:
            str: A new user ID
        """
        return str(uuid.uuid4())[:8]

    def get_threads_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation threads for a given user.

        Args:
            user_id: The user ID to get threads for

        Returns:
            List[Dict[str, Any]]: List of thread information
        """
        # For now, return empty list since we're using InMemoryStore
        # In a real implementation, this would query the database
        return []

    def create_thread_id(self) -> str:
        """Generate a new thread ID.

        Returns:
            str: A new thread ID
        """
        return str(uuid.uuid4())[:8]

    def process_message(self, user_prompt: str, user_id: str,
                        thread_id: str) -> Generator[Dict[str, Any], None, None]:
        """Process a user message and generate a response.

        Args:
            user_prompt: The user's message
            user_id: The user ID
            thread_id: The conversation thread ID

        Yields:
            Dict[str, Any]: Output chunks from the graph execution
        """
        # Configure runtime settings
        config = {"configurable": {
            "user_id": user_id,
            "thread_id": thread_id
        }}

        # Process the user input and yield results
        for chunk in self.graph.stream({"messages": [("user", user_prompt)]}, config=config):
            yield chunk

    def start_live_session(self, user_id: str, video_mode: str = "camera",
                          on_audio_receive=None, on_text_receive=None, on_visual_alert=None) -> Optional[str]:
        """Start a live audio/video session for a user.

        Args:
            user_id: The user ID to start the session for
            video_mode: Camera mode ('camera', 'screen', or 'none')
            on_audio_receive: Callback for received audio
            on_text_receive: Callback for received text
            on_visual_alert: Callback for visual safety alerts

        Returns:
            Session ID if successful, None otherwise
        """
        if not self.live_session_manager:
            return None

        import asyncio
        try:
            # Run the async function in the event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            session_id = loop.run_until_complete(
                self.live_session_manager.start_session(
                    user_id, video_mode, on_audio_receive, on_text_receive, on_visual_alert
                )
            )
            return session_id
        except Exception as e:
            print(f"Error starting live session: {e}")
            return None

    def stop_live_session(self, session_id: str) -> bool:
        """Stop an active live session.

        Args:
            session_id: The session ID to stop

        Returns:
            True if the session was stopped, False otherwise
        """
        if not self.live_session_manager:
            return False

        return self.live_session_manager.stop_session(session_id)

    def get_active_sessions(self) -> Dict[str, Any]:
        """Get information about all active live sessions.

        Returns:
            Dictionary with session information
        """
        if not self.live_session_manager:
            return {}

        return self.live_session_manager.get_active_sessions()

    def send_notification(self, user_id: str, message: str, notification_type: str = "info") -> bool:
        """Send a notification to a user.

        Args:
            user_id: The user ID to send the notification to
            message: The notification message
            notification_type: Type of notification ('info', 'warning', 'alert')

        Returns:
            True if the notification was sent successfully, False otherwise
        """
        try:
            # In a real implementation, this would send notifications via various channels
            # (push notifications, email, SMS, etc.)
            print(f"[{notification_type.upper()}] Notification for user {user_id}: {message}")

            # Log the notification to the database
            from core.memory_manager import db
            from datetime import datetime
            db["notifications"].insert_one({
                "user_id": user_id,
                "message": message,
                "type": notification_type,
                "timestamp": datetime.now(),
                "status": "sent"
            })

            return True
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False

    def schedule_reminder(self, user_id: str, message: str, reminder_time: datetime,
                         repeat_interval: Optional[str] = None) -> str:
        """Schedule a reminder for a user.

        Args:
            user_id: The user ID to schedule the reminder for
            message: The reminder message
            reminder_time: When to send the reminder
            repeat_interval: How often to repeat ('daily', 'weekly', 'monthly', None)

        Returns:
            Reminder ID if scheduled successfully
        """
        try:
            from core.memory_manager import db
            import uuid

            reminder_id = str(uuid.uuid4())
            db["reminders"].insert_one({
                "reminder_id": reminder_id,
                "user_id": user_id,
                "message": message,
                "scheduled_time": reminder_time,
                "repeat_interval": repeat_interval,
                "status": "active"
            })

            return reminder_id
        except Exception as e:
            print(f"Error scheduling reminder: {e}")
            return ""

    def get_user_notifications(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications for a user.

        Args:
            user_id: The user ID to get notifications for
            limit: Maximum number of notifications to return

        Returns:
            List of notification dictionaries
        """
        try:
            from core.memory_manager import db
            from datetime import datetime

            notifications = list(db["notifications"].find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit))

            # Convert ObjectId to string for JSON serialization
            for note in notifications:
                if "_id" in note:
                    note["_id"] = str(note["_id"])
                if "timestamp" in note and isinstance(note["timestamp"], datetime):
                    note["timestamp"] = note["timestamp"].isoformat()

            return notifications
        except Exception as e:
            print(f"Error getting notifications: {e}")
            return []