"""
System configuration for the LTM application.
Centralizes all configurable parameters and settings.
"""

import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SystemConfig:
    """Centralized configuration class for the LTM application."""
    
    # Database Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "ltm_database")
    
    # Model Configuration
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")  # groq, ollama, openai, anthropic
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Embedding Configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai, google, huggingface
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Crisis Detection Configuration
    CRISIS_PATTERNS = [
        r'\bsuicid\w*',  # suicide, suicidal
        r'\bkill\b.*\bmyself\b',
        r'\bend\b.*\blife\b',
        r'\bhurt\b.*\bmyself\b',
        r'\boverdose\b',
        r'\bwant\b.*\bdie\b',
        r'\bcan\'t\b.*\bgo\b.*\bon\b',  # "can't go on"
        r'\bno\b.*\bpoint\b.*\bliving\b',  # "no point living"
        r'\bgive\b.*\bup\b',  # "give up" in context of life
        r'\bworthless\b.*\bdead\b',  # "better off dead"
        r'\bno\b.*\bone\b.*\bcare\b',  # "no one cares"
        r'\bend\b.*\bit\b',  # "end it" - refers to ending life
        r'\brelief\b.*\bdeath\b',  # "relief in death"
        r'\bpermanent\b.*\brelease\b',  # euphemism for suicide
        r'\bgone\b.*\baway\b.*\bforever\b',  # "gone away forever"
    ]

    HIGH_NEG_INDICATORS = [
        'alone', 'nobody understands', 'nothing matters', 'tired of living',
        'everyone would be better off', 'life is meaningless', 'can\'t take it anymore',
        'fed up', 'done with suffering', 'ready to go', 'time to leave',
        'can\'t face another day', 'wish I could disappear', 'want to fade away'
    ]

    CRISIS_MARKERS = [
        'last time', 'final decision', 'won\'t bother anyone', 'won\'t be here',
        'goodbye everyone', 'see you later alligator', 'final goodbye',
        'done with everything', 'had enough', 'can\'t handle this anymore',
        'this is it', 'my time', 'final chapter', 'closing the book',
        'taking the leap', 'going to sleep forever'
    ]

    NEGATIVE_WORDS = [
        'depressed', 'desperate', 'hopeless', 'trapped', 'isolated',
        'worthless', 'guilty', 'ashamed', 'angry', 'frustrated',
        'overwhelmed', 'helpless', 'panic', 'terrified', 'afraid',
        'tormented', 'agonizing', 'devastated', 'numb', 'broken',
        'defeated', 'powerless', 'drowning', 'suffocating', 'crumbling'
    ]

    POSITIVE_WORDS = [
        'happy', 'joy', 'love', 'excited', 'grateful', 'thankful',
        'peaceful', 'calm', 'hopeful', 'optimistic', 'confident',
        'rejuvenated', 'inspired', 'motivated', 'content', 'fulfilled'
    ]

    ISOLATION_PATTERNS = [
        r'\balone\b.*\bforever\b',
        r'\bnever\b.*\banyone\b',
        r'\bcan\'t\b.*\btalk\b.*\banyone\b',
        r'\bno\b.*\bway\b.*\bout\b',
        r'\bby\b.*\bmyself\b.*\band\b.*\bmyself\b',
        r'\bno\b.*\bneed\b.*\banyone\b',
        r'\beveryone\b.*\bleft\b.*\bme\b'
    ]

    # Enhanced Crisis Detection Configuration
    EMOTIONAL_ESCALATION_PATTERNS = [
        r'\bcan\'t\b.*\btake\b.*\banymore\b',
        r'\bjust\b.*\bend\b.*\bit\b',
        r'\bneed\b.*\bway\b.*\bout\b',
        r'\blooking\b.*\bfor\b.*\bway\b.*\bout\b',
        r'\bthink\b.*\bit\b.*\bover\b',  # "think it over" in context of ending life
        r'\bready\b.*\bto\b.*\bgo\b',
        r'\btime\b.*\bis\b.*\bup\b',
        r'\benough\b.*\bis\b.*\benough\b',
    ]

    PHYSICAL_DISTRESS_INDICATORS = [
        'can\'t breathe', 'chest tightness', 'heart racing', 'can\'t sleep',
        'not eating', 'can\'t function', 'paralyzed', 'frozen',
        'can\'t concentrate', 'mind blank', 'numb inside'
    ]

    RELATIONAL_STRESS_INDICATORS = [
        'betrayed', 'abandoned', 'unwanted', 'rejected', 'unloved',
        'unworthy', 'failure', 'disappointment', 'burden to others'
    ]
    
    # Selfie Request Configuration
    SELFIE_REQUEST_INTERVAL_HOURS = int(os.getenv("SELFIE_REQUEST_INTERVAL_HOURS", "24"))
    SELFIE_REQUEST_MOOD_THRESHOLD = float(os.getenv("SELFIE_REQUEST_MOOD_THRESHOLD", "0.6"))
    SELFIE_RANDOM_CHANCE = float(os.getenv("SELFIE_RANDOM_CHANCE", "0.1"))
    
    # Mood Tracking Configuration
    MOOD_OPTIONS = [
        "Happy", "Sad", "Anxious", "Angry", "Neutral", 
        "Calm", "Excited", "Tired", "Stressed", "Hopeful"
    ]
    
    # Visual Context Configuration
    ENABLE_VISUAL_CONTEXT = os.getenv("ENABLE_VISUAL_CONTEXT", "true").lower() == "true"
    VISUAL_ANALYSIS_ENABLED = os.getenv("VISUAL_ANALYSIS_ENABLED", "true").lower() == "true"
    
    # Notification Configuration
    NOTIFICATION_CHANNELS = os.getenv("NOTIFICATION_CHANNELS", "console,email").split(",")
    CRITICAL_ALERT_CONTACTS = os.getenv("CRITICAL_ALERT_CONTACTS", "guardian,parent,crisis_hotline").split(",")
    
    # Live Session Configuration
    LIVE_SESSION_ENABLED = os.getenv("LIVE_SESSION_ENABLED", "false").lower() == "true"
    DEFAULT_LIVE_MODE = os.getenv("DEFAULT_LIVE_MODE", "camera")
    
    # Memory Configuration
    MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "365"))
    MAX_MEMORY_ENTRIES = int(os.getenv("MAX_MEMORY_ENTRIES", "1000"))
    
    # Security Configuration
    ENABLE_SECURITY_LOGGING = os.getenv("ENABLE_SECURITY_LOGGING", "true").lower() == "true"
    
    @classmethod
    def get_crisis_threshold(cls) -> float:
        """Get the threshold for crisis detection."""
        return float(os.getenv("CRISIS_THRESHOLD", "0.7"))
    
    @classmethod
    def get_sentiment_weight(cls) -> Dict[str, float]:
        """Get weights for different sentiment factors."""
        return {
            "negative_word_weight": float(os.getenv("NEGATIVE_WORD_WEIGHT", "1.0")),
            "positive_word_weight": float(os.getenv("POSITIVE_WORD_WEIGHT", "1.0")),
            "pattern_match_weight": float(os.getenv("PATTERN_MATCH_WEIGHT", "2.0")),
            "contextual_indicator_weight": float(os.getenv("CONTEXTUAL_INDICATOR_WEIGHT", "1.5"))
        }
    
    @classmethod
    def get_api_endpoints(cls) -> Dict[str, str]:
        """Get API endpoint configurations."""
        return {
            "whatsapp_endpoint": os.getenv("WHATSAPP_ENDPOINT", ""),
            "telegram_endpoint": os.getenv("TELEGRAM_ENDPOINT", ""),
            "ehr_endpoint": os.getenv("EHR_ENDPOINT", "")
        }
    
    @classmethod
    def get_embedding_dimensions(cls) -> int:
        """Get embedding dimensions based on model."""
        model_dims = {
            "text-embedding-ada-002": 1536,
            "models/embedding-001": 768,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768
        }
        return model_dims.get(cls.EMBEDDING_MODEL, 384)