"""
Tools configuration for the LTM application.

This module defines and configures the tools available to the agent.
"""
from langchain_community.tools import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import StructuredTool
from typing import List

from config.app_config import get_config
from core.memory_manager import (
    memory_store,
    manage_episodic_memory_tool,
    search_episodic_memory_tool,
    manage_semantic_memory_tool,
    search_semantic_memory_tool,
    manage_procedural_memory_tool,
    search_procedural_memory_tool,
    manage_associative_memory_tool,
    search_associative_memory_tool,
    manage_general_memory_tool,
    search_general_memory_tool
)

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

# Memory management tools - all types available
memory_tools = [
    manage_episodic_memory_tool,    # Create/update/delete episodic memories
    search_episodic_memory_tool,    # Search episodic memories (experiences/learning)
    manage_semantic_memory_tool,    # Create/update/delete semantic memories (facts/triples)
    search_semantic_memory_tool,    # Search semantic memories (facts/relationships)
    manage_procedural_memory_tool,  # Create/update/delete procedural memories (instructions/rules)
    search_procedural_memory_tool,  # Search procedural memories (how-to knowledge)
    manage_associative_memory_tool, # Create/update/delete associative memories (concept connections)
    search_associative_memory_tool, # Search associative memories (relationship patterns)
    manage_general_memory_tool,     # General memory management (mixed usage)
    search_general_memory_tool,     # General memory search (mixed retrieval)
]

all_tools = [
    search_internet_tool
]

# Add all memory tools
all_tools.extend(memory_tools)

# Mood Tracking Tool
from langchain_core.tools import tool
from datetime import datetime
from core.memory_manager import db

@tool
def log_mood_tool(mood: str, intensity: int, notes: str = ""):
    """Logs the user's current mood. Useful when the user explicitly states how they feel.
    Args:
        mood: One of 'Happy', 'Sad', 'Anxious', 'Angry', 'Neutral'.
        intensity: 1-10 scale.
        notes: Optional context.
    """
    db["mood_logs"].insert_one({
        "mood": mood,
        "intensity": intensity,
        "notes": notes,
        "timestamp": datetime.now()
    })
    return f"Logged mood: {mood} ({intensity}/10)"

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

@tool
def music_therapy_tool(mood: str, duration_minutes: int = 10) -> str:
    """Provides music therapy recommendations based on the user's mood.
    Args:
        mood: Current mood ('happy', 'sad', 'anxious', 'calm', 'energetic', 'sleepy')
        duration_minutes: Duration for the music session (default 10 minutes)
    Returns:
        String with music recommendations and therapeutic benefits
    """
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

    mood = mood.lower()
    if mood not in music_recommendations:
        mood = "calm"  # Default to calm if mood not recognized

    selected_music = random.choice(music_recommendations[mood])

    # Log the music therapy session
    db["music_therapy_sessions"].insert_one({
        "mood": mood,
        "recommendation": selected_music,
        "duration": duration_minutes,
        "timestamp": datetime.now()
    })

    return f"Music Therapy Recommendation: {selected_music}. Duration: {duration_minutes} minutes. This music is known to help with {mood} feelings."

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

all_tools.append(log_mood_tool)
all_tools.append(send_alert_tool)
all_tools.append(music_therapy_tool)
all_tools.append(request_selfie_tool)
all_tools.append(analyze_visual_context_tool)
all_tools.append(send_whatsapp_message_tool)
all_tools.append(send_telegram_message_tool)
all_tools.append(log_to_ehr_tool)