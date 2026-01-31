"""
API layer for the LTM application using FastAPI.
Provides REST endpoints for all core functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from core.service import LTMService
from core.tools import log_mood_tool

app = FastAPI(
    title="LTMAgent API",
    description="REST API for the Long-Term Memory Agent application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
service = LTMService()

# Pydantic models for request/response bodies
class MessageRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

class MessageResponse(BaseModel):
    response: str
    timestamp: datetime

class UserCreateRequest(BaseModel):
    user_id: Optional[str] = None

class ThreadCreateRequest(BaseModel):
    user_id: str

class MoodLogRequest(BaseModel):
    user_id: str
    mood: str
    intensity: int
    notes: str = ""

class AlertRequest(BaseModel):
    message: str
    user_id: str
    specific_contact: str = "guardian"

class ReminderRequest(BaseModel):
    user_id: str
    message: str
    reminder_time: datetime
    repeat_interval: Optional[str] = None

class LiveSessionRequest(BaseModel):
    user_id: str
    video_mode: str = "camera"

# API Routes
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "LTMAgent API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    return service.get_model_info()

@app.get("/users")
async def get_users():
    """Get a list of existing user IDs."""
    users = service.get_available_users()
    return {"users": users}

@app.post("/users")
async def create_user(request: UserCreateRequest):
    """Create a new user ID."""
    if request.user_id:
        user_id = request.user_id
    else:
        user_id = service.create_user_id()
    
    return {"user_id": user_id}

@app.post("/threads")
async def create_thread(request: ThreadCreateRequest):
    """Create a new conversation thread for a user."""
    thread_id = service.create_thread_id()
    return {
        "user_id": request.user_id,
        "thread_id": thread_id
    }

@app.get("/threads/{user_id}")
async def get_threads(user_id: str):
    """Get conversation threads for a given user."""
    threads = service.get_threads_for_user(user_id)
    return {"user_id": user_id, "threads": threads}

@app.post("/chat")
async def process_message(request: MessageRequest):
    """Process a user message and return the agent's response."""
    try:
        # Process the message and collect the response
        response_parts = []
        for chunk in service.process_message(request.message, request.user_id, request.thread_id):
            for node, updates in chunk.items():
                if node == "agent" and "messages" in updates:
                    message = updates["messages"][-1]
                    if hasattr(message, 'content'):
                        response_parts.append(message.content)
        
        response_text = " ".join(response_parts) if response_parts else "I'm sorry, I couldn't process that message."
        
        return MessageResponse(
            response=response_text,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/mood/log")
async def log_mood(request: MoodLogRequest):
    """Log a user's current mood."""
    try:
        result = log_mood_tool(mood=request.mood, intensity=request.intensity, notes=request.notes)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging mood: {str(e)}")

@app.post("/alerts/send")
async def send_alert(request: AlertRequest):
    """Send an emergency alert."""
    try:
        from core.tools import send_alert_tool
        result = send_alert_tool(message=request.message, specific_contact=request.specific_contact)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending alert: {str(e)}")

@app.post("/reminders/schedule")
async def schedule_reminder(request: ReminderRequest):
    """Schedule a reminder for a user."""
    try:
        reminder_id = service.schedule_reminder(
            user_id=request.user_id,
            message=request.message,
            reminder_time=request.reminder_time,
            repeat_interval=request.repeat_interval
        )
        return {"status": "success", "reminder_id": reminder_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling reminder: {str(e)}")

@app.get("/notifications/{user_id}")
async def get_notifications(user_id: str, limit: int = 10):
    """Get recent notifications for a user."""
    try:
        notifications = service.get_user_notifications(user_id, limit)
        return {"user_id": user_id, "notifications": notifications}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting notifications: {str(e)}")

@app.post("/live/start")
async def start_live_session(request: LiveSessionRequest):
    """Start a live audio/video session."""
    try:
        session_id = service.start_live_session(request.user_id, request.video_mode)
        if session_id:
            return {"status": "success", "session_id": session_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to start live session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting live session: {str(e)}")

@app.post("/live/stop/{session_id}")
async def stop_live_session(session_id: str):
    """Stop an active live session."""
    try:
        success = service.stop_live_session(session_id)
        if success:
            return {"status": "success", "message": "Session stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or already stopped")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping live session: {str(e)}")

@app.get("/live/status")
async def get_live_status():
    """Get information about all active live sessions."""
    try:
        sessions = service.get_active_sessions()
        return {"active_sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting live status: {str(e)}")

@app.get("/integrations/status")
async def get_integration_status():
    """Get status of all external integrations."""
    try:
        from core.integrations import get_integration_manager
        integration_manager = get_integration_manager()
        
        return {
            "whatsapp": integration_manager.whatsapp.is_available(),
            "telegram": integration_manager.telegram.is_available(),
            "ehr": integration_manager.ehr.is_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting integration status: {str(e)}")

@app.post("/integrations/test")
async def test_integrations():
    """Test all configured integrations."""
    try:
        from core.integrations import get_integration_manager
        integration_manager = get_integration_manager()
        
        results = {
            "whatsapp": integration_manager.whatsapp.is_available(),
            "telegram": integration_manager.telegram.is_available(),
            "ehr": integration_manager.ehr.is_available()
        }
        
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing integrations: {str(e)}")

@app.get("/mood/history/{user_id}")
async def get_mood_history(user_id: str, limit: int = 50):
    """Get mood history for a user."""
    try:
        from core.memory_manager import db
        mood_logs = list(db["mood_logs"].find(
            {"user_id": user_id},
            {"_id": 0, "timestamp": 1, "intensity": 1, "mood": 1, "notes": 1}
        ).sort("timestamp", -1).limit(limit))
        
        return {"user_id": user_id, "mood_logs": mood_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting mood history: {str(e)}")

@app.post("/scheduler/wellness/{user_id}")
async def schedule_wellness_tasks(user_id: str):
    """Schedule wellness tasks (selfie requests, mood check-ins) for a user."""
    try:
        success = service.schedule_wellness_tasks(user_id)
        if success:
            return {"status": "success", "message": f"Wellness tasks scheduled for user {user_id}"}
        else:
            raise HTTPException(status_code=503, detail="Scheduler not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling wellness tasks: {str(e)}")

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get status of the wellness scheduler."""
    try:
        status = service.get_scheduler_status()
        return {"status": "success", "scheduler": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scheduler status: {str(e)}")

@app.get("/crisis/history/{user_id}")
async def get_crisis_history(user_id: str, limit: int = 10):
    """Get crisis event history for a user."""
    try:
        from core.memory_manager import db
        crisis_events = list(db["crisis_events"].find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        # Convert timestamps to ISO format
        for event in crisis_events:
            if "timestamp" in event:
                event["timestamp"] = event["timestamp"].isoformat()
        
        return {"user_id": user_id, "crisis_events": crisis_events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crisis history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)