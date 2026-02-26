"""
Tools configuration for the LTM application.

This module defines and configures the tools available to the agent.
"""
import os
from langchain_community.tools import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import StructuredTool
from typing import List
from pymongo import MongoClient
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from core.gamification import GamificationManager # Import GamificationManager

from config.app_config import get_config
from core.memory_manager import (
    memory_store,
    db, # Import db for direct access
    manage_episodic_memory_tool,
    search_episodic_memory_tool,
    manage_semantic_memory_tool,
    search_semantic_memory_tool,
    manage_procedural_memory_tool,
    search_procedural_memory_tool,
    manage_associative_memory_tool,
    search_associative_memory_tool,
    manage_general_memory_tool,
    search_general_memory_tool,
    get_emergency_contacts # Import the new function
)
from config.system_config import SystemConfig # Import SystemConfig

# Try to import LangMem tools (for fallback if needed)
try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False

search_internet_tool = SearxSearchResults(
    num_results=5,
    wrapper=SearxSearchWrapper(searx_host=get_config("searx_host"))
)

# Initialize embeddings model for RAG tool
def _initialize_embeddings_model_for_rag():
    if SystemConfig.EMBEDDING_PROVIDER == "openai":
        if not SystemConfig.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set for OpenAI embeddings.")
        return OpenAIEmbeddings(openai_api_key=SystemConfig.OPENAI_API_KEY, model=SystemConfig.EMBEDDING_MODEL)
    elif SystemConfig.EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddings(model_name=SystemConfig.HF_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding provider: {SystemConfig.EMBEDDING_PROVIDER}")

rag_embeddings_model = _initialize_embeddings_model_for_rag()

@tool
def consult_academic_resources_tool(query: str) -> str:
    """
    Consults a knowledge base of academic resources (e.g., study guides, coping strategies)
    to answer student-specific questions or provide relevant information.
    Performs a vector similarity search against ingested academic materials.
    """
    # Connect to MongoDB
    client = MongoClient(SystemConfig.MONGODB_URI)
    db = client[SystemConfig.DATABASE_NAME]
    collection = db[SystemConfig.RAG_COLLECTION_NAME]

    # Generate embedding for the query
    query_embedding = rag_embeddings_model.embed_query(query)

    # Perform vector similarity search (using a basic approximation for now)
    # In a real-world scenario, you'd use MongoDB Atlas Vector Search or a dedicated vector DB
    # For this example, we'll manually calculate cosine similarity for simplicity if Atlas Vector Search is not configured
    # This is a placeholder for actual vector search capabilities.

    # TODO: Replace with proper MongoDB Atlas Vector Search when available/configured
    # For now, we'll fetch all and do a very basic similarity sort (not efficient for large datasets)
    # This section needs to be improved with actual vector indexing and search
    # For demonstration, we'll assume a simplified search:
    
    results = []
    # This part needs a proper vector search implementation, likely with MongoDB Atlas Vector Search.
    # For a basic simulation:
    # This is highly inefficient and only for demonstration without a proper vector search index.
    # It assumes the 'embedding' field exists in the documents.
    if collection.estimated_document_count() > 0:
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,  # Number of nearest neighbors to consider
                    "limit": 5,           # Return top 5 results
                    "index": "vector_index", # This needs to be created in MongoDB Atlas
                }
            },
            {
                "$project": {
                    "content": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
        except Exception as e:
            print(f"Vector search failed (Is vector_index configured?): {e}. Falling back to keyword search.")
            # Fallback to basic keyword search if vector search fails
            results = list(collection.find({"content": {"$regex": query, "$options": "i"}}).limit(5))
            results = [{"content": doc["content"], "source": doc.get("source", "unknown"), "score": 0} for doc in results] # Assign dummy score


    if not results:
        return "No relevant academic resources found for your query."

    formatted_results = []
    for doc in results:
        content = doc.get("content", "N/A")
        source = doc.get("source", "N/A")
        score = doc.get("score", "N/A")
        formatted_results.append(f"Content: {content}\nSource: {source}\nRelevance Score: {score}")

    return "Found the following academic resources:\n\n" + "\n\n---\n\n".join(formatted_results)

# Mood Tracking Tool
from langchain_core.tools import tool
from datetime import datetime

@tool
def log_mood_tool(mood: str, intensity: int, notes: str = "", user_id: str = "default_user"):
    """Logs the user's current mood. Useful when the user explicitly states how they feel.
    Args:
        mood: One of 'Happy', 'Sad', 'Anxious', 'Angry', 'Neutral'.
        intensity: 1-10 scale.
        notes: Optional context.
        user_id: The ID of the user logging the mood.
    """
    db["mood_logs"].insert_one({
        "user_id": user_id,
        "mood": mood,
        "intensity": intensity,
        "notes": notes,
        "timestamp": datetime.now()
    })

    gamification_manager = GamificationManager(user_id)
    gamification_update = gamification_manager.log_mood_event()

    return (
        f"Logged mood: {mood} ({intensity}/10). "
        f"Gamification update: {gamification_update['xp_gained']} XP gained (Total: {gamification_update['total_xp']} XP). "
        f"{gamification_update['streak_message']}"
    )

@tool
def check_progress_tool(user_id: str) -> str:
    """
    Checks the user's current gamification progress, including XP and streak days.
    Useful for providing positive reinforcement or tracking user engagement.
    """
    gamification_manager = GamificationManager(user_id)
    progress = gamification_manager.get_user_progress()
    return (
        f"Your current progress: {progress['xp']} XP, "
        f"{progress['streak_days']}-day streak. "
        f"Last mood logged on: {progress['last_mood_log_date']}."
    )

@tool
def send_alert_tool(message: str, specific_contact: str = "guardian"):
    """Sends an emergency alert to a guardian.
    Use ONLY in crisis situations (suicide, self-harm, immediate danger).
    """
    # Mock Twilio/WhatsApp sending
    print(f"CRITICAL ALERT SENT TO {specific_contact}: {message}")
    return f"Critical alert sent to {specific_contact}: {message}"

# Music Therapy Tool
import random
from typing import List

class MusicTherapyTool:
    """Class-based implementation of the Music Therapy tool.

    This class encapsulates recommendation logic and session logging so it can be
    instantiated and used directly or wrapped by a `tool`/`StructuredTool` adapter.
    """
    def __init__(self, db_client=None):
        # db_client is expected to be the pymongo `db` object imported from memory_manager
        self.db = db_client

    def recommend(self, mood: str, duration_minutes: int = 10) -> str:
        """Provide a music recommendation and log the session."""
        music_recommendations = {
            "happy": [
                "Upbeat pop music to maintain positive energy",
                "Classical music for cognitive enhancement",
                "Jazz for creative stimulation"
            ],
            "sad": [
                "Gentle classical music for emotional processing",
                "Nature sounds for comfort",
                "Soft instrumental music for reflection"
            ],
            "anxious": [
                "Ambient music for relaxation",
                "Binaural beats for stress reduction",
                "Nature sounds (rain, ocean waves) for calm"
            ],
            "calm": [
                "Meditation music for deeper relaxation",
                "Acoustic guitar for peaceful atmosphere",
                "Piano compositions for introspection"
            ],
            "energetic": [
                "Motivational rock for energy boost",
                "Upbeat electronic music for focus",
                "Folk music for positive vibes"
            ],
            "sleepy": [
                "Slow tempo classical music for sleep",
                "White noise for better sleep quality",
                "Guided meditation music for rest"
            ]
        }

        mood_normalized = (mood or "").lower()
        if mood_normalized not in music_recommendations:
            mood_normalized = "calm"

        selected_music = random.choice(music_recommendations[mood_normalized]) # nosec B311

        # Log session if db available
        try:
            if self.db:
                self.db["music_therapy_sessions"].insert_one({
                    "mood": mood_normalized,
                    "recommendation": selected_music,
                    "duration": duration_minutes,
                    "timestamp": datetime.now()
                })
        except Exception: # nosec B110: Intentionally suppress non-critical logging errors to avoid tool failure
            # Never fail the tool due to logging issues
            pass

        return f"Music Therapy Recommendation: {selected_music}. Duration: {duration_minutes} minutes. Therapeutic target: {mood_normalized}."

# Adapter: keep the existing functional tool for backwards compatibility
@tool
def music_therapy_tool(mood: str, duration_minutes: int = 10) -> str:
    """Provides music therapy recommendations based on the user's mood.
    This function wraps the class-based implementation for backward compatibility.
    """
    instance = MusicTherapyTool(db_client=db)
    return instance.recommend(mood, duration_minutes)

# Expose a class-based tool instance as a StructuredTool if needed
try:
    music_therapy_class_tool = StructuredTool(
        name="music_therapy_class",
        description="Music therapy recommendations (class-based)",
        func=MusicTherapyTool(db_client=db).recommend
    )
except Exception: # nosec B110: Intentionally suppress if StructuredTool isn't available to avoid application crash
    # If StructuredTool isn't available, continue silently (function wrapper exists)
    pass

# End of Music Therapy Tool
@tool
def request_selfie_tool(reason: str = "routine check-in") -> str:
    """Requests the user to take a selfie for mood assessment.
    Args:
        reason: Reason for requesting the selfie ('routine check-in', 'wellness monitoring', 'crisis assessment')
    Returns:
        Message prompting the user to take a selfie
    """
    # Log the selfie request
    db["selfie_requests"].insert_one({
        "reason": reason,
        "timestamp": datetime.now(),
        "status": "requested"
    })

    return f"I'd like to check in on your well-being. Could you please take a quick selfie? This will help me assess your mood and provide better support. Reason: {reason}"

@tool
def analyze_visual_context_tool(image_description: str) -> str:
    """Analyzes visual context from user's environment or appearance.
    Args:
        image_description: Description of what's visible in the user's camera feed
    Returns:
        Analysis of visual context and its implications for user's wellbeing
    """
    # This would normally connect to a computer vision model in a real implementation
    # For now, we'll simulate analysis based on the description

    # Log the visual analysis request
    db["visual_analyses"].insert_one({
        "description": image_description,
        "timestamp": datetime.now()
    })

    # Analyze the description for indicators
    description_lower = image_description.lower()

    # Look for environmental and appearance indicators
    environmental_indicators = {
        "disorganized_space": ["messy", "cluttered", "untidy", "chaotic"],
        "isolated_setting": ["alone", "empty room", "no people", "quiet"],
        "stress_indicators": ["dark", "dim lighting", "poor hygiene", "unkempt"],
        "positive_indicators": ["natural light", "plants", "organized", "clean"]
    }

    findings = []
    for category, keywords in environmental_indicators.items():
        for keyword in keywords:
            if keyword in description_lower:
                findings.append(category.replace("_", " ").title())

    # Physical appearance indicators
    appearance_indicators = [
        "tired eyes", "slouched posture", "tears", "distressed facial expression",
        "pale complexion", "disheveled appearance", "restless movements"
    ]

    for indicator in appearance_indicators:
        if indicator in description_lower:
            findings.append(f"Physical sign: {indicator}")

    if not findings:
        analysis_result = "Visual context appears normal. No concerning indicators detected."
    else:
        analysis_result = f"Visual analysis detected: {', '.join(findings)}. This may suggest the user needs additional support."

    return analysis_result

# Integration tools
from core.integrations import get_integration_manager

@tool
def send_whatsapp_message_tool(phone_number: str, message: str) -> str:
    """Send a message to a user via WhatsApp.
    Args:
        phone_number: Recipient's phone number in international format
        message: Message to send
    Returns:
        Status of the message delivery
    """
    integration_manager = get_integration_manager()
    if integration_manager.whatsapp.is_available():
        success = integration_manager.whatsapp.send_message(phone_number, message)
        if success:
            return f"Message sent successfully to {phone_number} via WhatsApp"
        else:
            return f"Failed to send message to {phone_number} via WhatsApp"
    else:
        return "WhatsApp integration not configured"

@tool
def send_telegram_message_tool(chat_id: str, message: str) -> str:
    """Send a message to a user via Telegram.
    Args:
        chat_id: Recipient's Telegram chat ID
        message: Message to send
    Returns:
        Status of the message delivery
    """
    integration_manager = get_integration_manager()
    if integration_manager.telegram.is_available():
        success = integration_manager.telegram.send_message(chat_id, message)
        if success:
            return f"Message sent successfully to chat {chat_id} via Telegram"
        else:
            return f"Failed to send message to chat {chat_id} via Telegram"
    else:
        return "Telegram integration not configured"

@tool
def log_to_ehr_tool(patient_id: str, note: str, category: str = "general") -> str:
    """Log a note to the patient's Electronic Health Record.
    Args:
        patient_id: Patient identifier
        note: Clinical note to log
        category: Category of the note ('general', 'therapy_session', 'crisis_alert', etc.)
    Returns:
        Status of the logging operation
    """
    integration_manager = get_integration_manager()
    if integration_manager.ehr.is_available():
        success = integration_manager.ehr.log_patient_note(patient_id, note, category)
        if success:
            return f"Note logged successfully to patient {patient_id} EHR"
        else:
            return f"Failed to log note to patient {patient_id} EHR"
    else:
        return "EHR integration not configured"

@tool
def send_sms_tool(phone_number: str, message: str) -> str:
    """Send an SMS message via Twilio.
    Args:
        phone_number: Recipient's phone number in E.164 format (e.g., +1234567890)
        message: Message to send
    Returns:
        Status of the message delivery
    """
    integration_manager = get_integration_manager()
    if integration_manager.twilio.is_available():
        success = integration_manager.twilio.send_message(phone_number, message)
        if success:
            return f"SMS sent successfully to {phone_number}"
        else:
            return f"Failed to send SMS to {phone_number}"
    else:
        return "Twilio integration not configured"

@tool
def crisis_escalation_tool(user_id: str, severity: str, context: str = "") -> str:
    """Execute a full crisis escalation protocol.
    This tool sends alerts via ALL available channels simultaneously.
    Use ONLY in confirmed crisis situations.
    
    Args:
        user_id: The user's ID for tracking
        severity: 'CRITICAL', 'WARNING', or 'CAUTION'
        context: Additional context about the crisis situation
        
    Returns:
        Summary of all escalation actions taken
    """
    integration_manager = get_integration_manager()
    results = []
    timestamp = datetime.now().isoformat()
    
    alert_message = f"🚨 {severity} CRISIS ALERT 🚨\nUser: {user_id}\nTime: {timestamp}\n{context}"
    
    # Log to database first
    try:
        db["crisis_alerts"].insert_one({
            "user_id": user_id,
            "severity": severity,
            "context": context,
            "timestamp": datetime.now(),
            "status": "triggered"
        })
        results.append("Crisis logged to database")
    except Exception as e:
        results.append(f"Failed to log crisis: {e}")
    
    # Get user-specific emergency contacts
    emergency_contacts = get_emergency_contacts(user_id)
    if not emergency_contacts:
        results.append("No user-specific emergency contacts found.")

    for contact in emergency_contacts:
        contact_type = contact.get("type")
        contact_value = contact.get("value")
        contact_name = contact.get("name", contact_value)

        if not contact_type or not contact_value:
            results.append(f"Skipping malformed contact: {contact}")
            continue

        alert_message_with_contact = f"🚨 {severity} CRISIS ALERT for {user_id} 🚨\nContact Name: {contact_name}\nTime: {timestamp}\nContext: {context}"

        if contact_type == "phone" and integration_manager.twilio.is_available():
            try:
                success = integration_manager.twilio.send_message(contact_value, alert_message_with_contact)
                if success:
                    results.append(f"Twilio SMS sent to {contact_name} ({contact_value})")
                else:
                    results.append(f"Twilio SMS failed for {contact_name} ({contact_value})")
            except Exception as e:
                results.append(f"Twilio error for {contact_name} ({contact_value}): {e}")
        elif contact_type == "whatsapp" and integration_manager.whatsapp.is_available():
            try:
                success = integration_manager.whatsapp.send_message(contact_value, alert_message_with_contact)
                if success:
                    results.append(f"WhatsApp sent to {contact_name} ({contact_value})")
                else:
                    results.append(f"WhatsApp failed for {contact_name} ({contact_value})")
            except Exception as e:
                results.append(f"WhatsApp error for {contact_name} ({contact_value}): {e}")
        elif contact_type == "telegram" and integration_manager.telegram.is_available():
            try:
                success = integration_manager.telegram.send_message(contact_value, alert_message_with_contact)
                if success:
                    results.append(f"Telegram sent to {contact_name} ({contact_value})")
                else:
                    results.append(f"Telegram failed for {contact_name} ({contact_value})")
            except Exception as e:
                results.append(f"Telegram error for {contact_name} ({contact_value}): {e}")
        else:
            results.append(f"No suitable integration for contact {contact_name} ({contact_type}: {contact_value})")
    
    # Log to EHR
    if integration_manager.ehr.is_available():
        try:
            success = integration_manager.ehr.log_patient_note(user_id, alert_message, "crisis_alert")
            if success:
                results.append("EHR updated")
        except Exception as e:
            results.append(f"EHR error: {e}")
    
    return f"Crisis escalation completed: {'; '.join(results)}"

all_tools = [
    search_internet_tool,
    consult_academic_resources_tool,
    log_mood_tool,
    check_progress_tool,
    send_alert_tool,
    music_therapy_tool,
    request_selfie_tool,
    analyze_visual_context_tool,
    send_whatsapp_message_tool,
    send_telegram_message_tool,
    log_to_ehr_tool,
    send_sms_tool,
    crisis_escalation_tool
]

all_tools.extend(memory_tools)

# Ensure music_therapy_class_tool is appended if it was created
try:
    if 'music_therapy_class_tool' in locals() and isinstance(music_therapy_class_tool, StructuredTool):
        all_tools.append(music_therapy_class_tool)
except NameError:
    pass