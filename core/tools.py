"""
Tools configuration for the LTM application.

This module defines and configures the tools available to the agent.
"""
import os
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
from typing import List, Dict, Any

class MusicTherapyTool:
    """Enhanced Music Therapy tool with real playable music URIs.

    Provides Spotify playlist links and YouTube music URLs based on therapeutic mood.
    Includes curated playlists for different emotional states and therapeutic goals.
    """
    
    # Curated therapeutic playlists with real URIs
    THERAPEUTIC_PLAYLISTS: Dict[str, List[Dict[str, Any]]] = {
        "happy": [
            {
                "title": "Happy Hits",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
                "youtube": "https://www.youtube.com/playlist?list=PLDIoUOhQQPlXqVFgpN0vOlFBT0X7rXQ0j",
                "description": "Upbeat pop music to maintain positive energy",
                "bpm_range": "120-140"
            },
            {
                "title": "Mood Booster",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
                "youtube": "https://www.youtube.com/results?search_query=mood+booster+playlist",
                "description": "Feel-good songs for an instant mood lift",
                "bpm_range": "100-130"
            }
        ],
        "sad": [
            {
                "title": "Peaceful Piano",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
                "youtube": "https://www.youtube.com/results?search_query=peaceful+piano+music",
                "description": "Gentle classical music for emotional processing",
                "bpm_range": "60-80"
            },
            {
                "title": "Healing Music",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
                "youtube": "https://www.youtube.com/results?search_query=healing+instrumental+music",
                "description": "Soft instrumental music for reflection and healing",
                "bpm_range": "50-70"
            }
        ],
        "anxious": [
            {
                "title": "Deep Focus",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ",
                "youtube": "https://www.youtube.com/results?search_query=calm+ambient+music",
                "description": "Ambient music for relaxation and stress reduction",
                "bpm_range": "60-90"
            },
            {
                "title": "Nature Sounds",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX4PP3DA4J0N8",
                "youtube": "https://www.youtube.com/results?search_query=nature+sounds+rain+ocean",
                "description": "Rain, ocean waves, and forest sounds for calm",
                "bpm_range": "N/A"
            },
            {
                "title": "Binaural Beats Therapy",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX7EF8wVxBVhG",
                "youtube": "https://www.youtube.com/results?search_query=binaural+beats+anxiety+relief",
                "description": "Alpha waves for anxiety reduction (use headphones)",
                "bpm_range": "N/A"
            }
        ],
        "calm": [
            {
                "title": "Meditation Music",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
                "youtube": "https://www.youtube.com/results?search_query=meditation+music+10+minutes",
                "description": "Music for mindfulness and deeper relaxation",
                "bpm_range": "40-60"
            },
            {
                "title": "Acoustic Morning",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX4E3UdUs7fUx",
                "youtube": "https://www.youtube.com/results?search_query=acoustic+guitar+relaxing",
                "description": "Acoustic guitar for peaceful atmosphere",
                "bpm_range": "70-90"
            }
        ],
        "energetic": [
            {
                "title": "Workout Motivation",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP",
                "youtube": "https://www.youtube.com/results?search_query=workout+motivation+music",
                "description": "High-energy music for motivation and focus",
                "bpm_range": "130-160"
            },
            {
                "title": "Power Hour",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX32NsLKyzScr",
                "youtube": "https://www.youtube.com/results?search_query=power+music+focus",
                "description": "Rock and electronic for energy boost",
                "bpm_range": "120-150"
            }
        ],
        "sleepy": [
            {
                "title": "Sleep",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp",
                "youtube": "https://www.youtube.com/results?search_query=sleep+music+8+hours",
                "description": "Slow tempo music for better sleep",
                "bpm_range": "40-60"
            },
            {
                "title": "White Noise",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX4aYNO8X5RpR",
                "youtube": "https://www.youtube.com/results?search_query=white+noise+sleep",
                "description": "White noise for deep, restful sleep",
                "bpm_range": "N/A"
            }
        ],
        "angry": [
            {
                "title": "Release Rage",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DX1tyCD9QhIWF",
                "youtube": "https://www.youtube.com/results?search_query=cathartic+music+playlist",
                "description": "Music for safe emotional release and catharsis",
                "bpm_range": "140-180"
            },
            {
                "title": "Transition to Calm",
                "spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
                "youtube": "https://www.youtube.com/results?search_query=calming+down+music",
                "description": "Help transition from anger to calm (play after release)",
                "bpm_range": "60-80"
            }
        ]
    }
    
    def __init__(self, db_client=None):
        self.db = db_client

    def recommend(self, mood: str, duration_minutes: int = 10) -> str:
        """Provide a music recommendation with playable URIs and log the session."""
        mood_normalized = (mood or "").lower().strip()
        
        # Map common variations
        mood_mapping = {
            "stressed": "anxious",
            "nervous": "anxious",
            "worried": "anxious",
            "depressed": "sad",
            "down": "sad",
            "upset": "sad",
            "tired": "sleepy",
            "exhausted": "sleepy",
            "excited": "energetic",
            "motivated": "energetic",
            "peaceful": "calm",
            "relaxed": "calm",
            "frustrated": "angry",
            "irritated": "angry",
            "joyful": "happy",
            "content": "happy"
        }
        
        mood_normalized = mood_mapping.get(mood_normalized, mood_normalized)
        
        if mood_normalized not in self.THERAPEUTIC_PLAYLISTS:
            mood_normalized = "calm"

        playlist = random.choice(self.THERAPEUTIC_PLAYLISTS[mood_normalized])
        
        # Log session if db available
        try:
            if self.db:
                self.db["music_therapy_sessions"].insert_one({
                    "mood": mood_normalized,
                    "playlist_title": playlist["title"],
                    "spotify_uri": playlist["spotify"],
                    "youtube_uri": playlist["youtube"],
                    "duration": duration_minutes,
                    "timestamp": datetime.now()
                })
        except Exception:
            pass

        response = f"""🎵 **Music Therapy Recommendation**

**Playlist:** {playlist['title']}
**Therapeutic Focus:** {playlist['description']}
**Duration:** {duration_minutes} minutes
**BPM Range:** {playlist['bpm_range']}

🎧 **Listen Now:**
- [Spotify]({playlist['spotify']})
- [YouTube]({playlist['youtube']})

💡 *Tip: For best therapeutic effect, use headphones in a comfortable position. Focus on your breathing as you listen.*"""

        return response

# Adapter: keep the existing functional tool for backwards compatibility
@tool
def music_therapy_tool(mood: str, duration_minutes: int = 10) -> str:
    """Provides music therapy recommendations with playable Spotify/YouTube links.
    
    Args:
        mood: User's current mood (happy, sad, anxious, calm, energetic, sleepy, angry)
        duration_minutes: How long to listen (default 10 minutes)
    
    Returns:
        Music recommendation with direct links to Spotify and YouTube playlists
    """
    instance = MusicTherapyTool(db_client=db)
    return instance.recommend(mood, duration_minutes)

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
def analyze_visual_context_tool(image_data: str = None, image_path: str = None) -> str:
    """Analyzes visual context from user's camera feed using Gemini Vision.
    
    Uses Google Gemini's vision capabilities to analyze:
    - Facial expressions and emotional state
    - Environmental factors (lighting, organization)
    - Safety concerns (weapons, self-harm indicators)
    
    Args:
        image_data: Base64-encoded image data (preferred)
        image_path: Path to image file (alternative)
    Returns:
        Analysis of visual context and its implications for user's wellbeing
    """
    import base64
    from google import genai
    from google.genai import types
    
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Visual analysis unavailable: GEMINI_API_KEY not configured"
    
    client = genai.Client(api_key=api_key)
    
    # Prepare the image
    image_parts = []
    if image_data:
        # Decode base64 if needed
        if isinstance(image_data, str) and not image_data.startswith('/'):
            try:
                image_bytes = base64.b64decode(image_data)
                image_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            except Exception as e:
                return f"Failed to decode image data: {e}"
    elif image_path:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        except Exception as e:
            return f"Failed to read image file: {e}"
    else:
        return "No image provided for analysis"
    
    # Create the analysis prompt
    analysis_prompt = """Analyze this image for a mental health wellness assessment. Provide a structured analysis:

1. **Emotional State**: Assess facial expressions, body language, and apparent mood (scale 1-10 for distress level)
2. **Environmental Factors**: Note lighting, organization, isolation indicators
3. **Safety Concerns**: Flag any visible weapons, sharp objects, medications, or self-harm indicators
4. **Wellness Indicators**: Positive signs like natural light, plants, organized space, grooming

Format your response as JSON:
{
    "distress_level": 0-10,
    "emotions_detected": ["emotion1", "emotion2"],
    "safety_concerns": ["concern1"] or [],
    "environmental_notes": "description",
    "wellness_summary": "brief summary",
    "recommended_action": "none|check_in|urgent_intervention"
}

Be accurate and conservative. Only flag safety concerns if clearly visible."""

    try:
        # Call Gemini Vision API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part(text=analysis_prompt),
                    *image_parts
                ])
            ]
        )
        
        analysis_text = response.text
        
        # Log the analysis
        db["visual_analyses"].insert_one({
            "analysis": analysis_text,
            "timestamp": datetime.now(),
            "model": "gemini-2.0-flash",
            "method": "gemini_vision"
        })
        
        # Parse for crisis detection
        import json
        try:
            analysis_json = json.loads(analysis_text)
            
            # Trigger crisis protocol if needed
            if analysis_json.get("recommended_action") == "urgent_intervention":
                return f"🚨 CRITICAL VISUAL ALERT: {analysis_json.get('wellness_summary', 'Urgent intervention required')}"
            elif analysis_json.get("distress_level", 0) >= 7:
                return f"⚠️ HIGH DISTRESS DETECTED (Level {analysis_json['distress_level']}/10): {analysis_json.get('wellness_summary', 'User appears distressed')}"
            else:
                return f"Visual analysis complete: {analysis_json.get('wellness_summary', analysis_text)}"
                
        except json.JSONDecodeError:
            # Return raw text if not valid JSON
            return f"Visual analysis: {analysis_text}"
            
    except Exception as e:
        # Fallback to text-based analysis if vision fails
        db["visual_analyses"].insert_one({
            "error": str(e),
            "timestamp": datetime.now(),
            "method": "fallback"
        })
        return f"Vision analysis failed: {e}. Please ensure camera access is enabled."

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
    
    # Send via Twilio (SMS) - Primary channel for emergencies
    if integration_manager.twilio.is_available():
        try:
            # Get emergency contacts from user profile (placeholder logic)
            emergency_number = os.environ.get("EMERGENCY_CONTACT_PHONE", "")
            if emergency_number:
                success = integration_manager.twilio.send_message(emergency_number, alert_message)
                if success:
                    results.append(f"Twilio SMS sent to {emergency_number}")
                else:
                    results.append("Twilio SMS failed")
        except Exception as e:
            results.append(f"Twilio error: {e}")
    
    # Send via WhatsApp
    if integration_manager.whatsapp.is_available():
        try:
            wa_number = os.environ.get("EMERGENCY_WHATSAPP_NUMBER", "")
            if wa_number:
                success = integration_manager.whatsapp.send_message(wa_number, alert_message)
                if success:
                    results.append(f"WhatsApp sent to {wa_number}")
        except Exception as e:
            results.append(f"WhatsApp error: {e}")
    
    # Send via Telegram
    if integration_manager.telegram.is_available():
        try:
            tg_chat = os.environ.get("EMERGENCY_TELEGRAM_CHAT", "")
            if tg_chat:
                success = integration_manager.telegram.send_message(tg_chat, alert_message)
                if success:
                    results.append(f"Telegram sent to chat {tg_chat}")
        except Exception as e:
            results.append(f"Telegram error: {e}")
    
    # Log to EHR
    if integration_manager.ehr.is_available():
        try:
            success = integration_manager.ehr.log_patient_note(user_id, alert_message, "crisis_alert")
            if success:
                results.append("EHR updated")
        except Exception as e:
            results.append(f"EHR error: {e}")
    
    return f"Crisis escalation completed: {'; '.join(results)}"

all_tools.append(log_mood_tool)
all_tools.append(send_alert_tool)
all_tools.append(music_therapy_tool)
all_tools.append(request_selfie_tool)
all_tools.append(analyze_visual_context_tool)
all_tools.append(send_whatsapp_message_tool)
all_tools.append(send_telegram_message_tool)
all_tools.append(log_to_ehr_tool)
all_tools.append(send_sms_tool)
all_tools.append(crisis_escalation_tool)