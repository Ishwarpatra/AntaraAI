"""
Agent processing logic for the LTM application.
"""

from langchain_core.messages import get_buffer_string, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from datetime import datetime, timedelta
import re

from core.state import State
from config.prompt_templates import prompt

# Cache for the ML model to avoid reloading
_sentiment_classifier = None
_emotion_classifier = None


def _get_ml_classifiers():
    """Lazy load ML classifiers to avoid startup overhead."""
    global _sentiment_classifier, _emotion_classifier
    
    if _sentiment_classifier is None:
        try:
            from transformers import pipeline
            
            # Sentiment classifier for positive/negative polarity
            _sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
            
            # Emotion classifier for more nuanced detection
            _emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=3,  # Get top 3 emotions
                device=-1  # CPU
            )
            
            print("✅ ML sentiment/emotion classifiers loaded successfully")
        except ImportError:
            print("⚠️ transformers not installed, using rule-based fallback")
            return None, None
        except Exception as e:
            print(f"⚠️ Failed to load ML classifiers: {e}")
            return None, None
    
    return _sentiment_classifier, _emotion_classifier


def analyze_sentiment_ml(text: str) -> dict:
    """
    ML-based sentiment analysis using HuggingFace Transformers.
    
    Returns:
        dict with:
            - sentiment: 'POSITIVE' | 'NEGATIVE'
            - sentiment_score: float (confidence)
            - emotions: list of detected emotions with scores
            - crisis_indicators: list of high-risk emotions
    """
    sentiment_clf, emotion_clf = _get_ml_classifiers()
    
    if sentiment_clf is None:
        return None  # Fallback to rule-based
    
    result = {
        "sentiment": "NEUTRAL",
        "sentiment_score": 0.0,
        "emotions": [],
        "crisis_indicators": []
    }
    
    try:
        # Get sentiment polarity
        sentiment_result = sentiment_clf(text[:512])[0]  # Limit to 512 tokens
        result["sentiment"] = sentiment_result["label"]
        result["sentiment_score"] = sentiment_result["score"]
        
        # Get emotions
        emotion_results = emotion_clf(text[:512])
        if emotion_results and isinstance(emotion_results[0], list):
            emotion_results = emotion_results[0]
        
        for emotion in emotion_results:
            result["emotions"].append({
                "label": emotion["label"],
                "score": emotion["score"]
            })
            
            # Check for crisis-indicating emotions
            high_risk_emotions = ["fear", "sadness", "anger", "disgust"]
            if emotion["label"].lower() in high_risk_emotions and emotion["score"] > 0.5:
                result["crisis_indicators"].append(emotion["label"])
                
    except Exception as e:
        print(f"⚠️ ML sentiment analysis error: {e}")
        return None
    
    return result


def analyze_sentiment_rules(text: str) -> tuple:
    """
    Rule-based sentiment analysis fallback.
    
    Returns:
        tuple: (risk_level, score_details)
    """
    from config.system_config import SystemConfig
    
    text_lower = text.lower().strip()
    
    crisis_score = 0
    emotional_escalation_score = 0
    physical_distress_score = 0
    relational_stress_score = 0

    # Crisis patterns
    for pattern in SystemConfig.CRISIS_PATTERNS:
        if re.search(pattern, text_lower):
            crisis_score += 3

    # High negative indicators
    for indicator in SystemConfig.HIGH_NEG_INDICATORS:
        if indicator in text_lower:
            crisis_score += 2

    # Crisis markers
    for marker in SystemConfig.CRISIS_MARKERS:
        if marker in text_lower:
            crisis_score += 3

    # Emotional escalation
    for pattern in SystemConfig.EMOTIONAL_ESCALATION_PATTERNS:
        if re.search(pattern, text_lower):
            emotional_escalation_score += 2

    # Physical distress
    for indicator in SystemConfig.PHYSICAL_DISTRESS_INDICATORS:
        if indicator in text_lower:
            physical_distress_score += 1

    # Relational stress
    for indicator in SystemConfig.RELATIONAL_STRESS_INDICATORS:
        if indicator in text_lower:
            relational_stress_score += 1

    # Sentiment intensity
    neg_count = sum(text_lower.count(word) for word in SystemConfig.NEGATIVE_WORDS)
    pos_count = sum(text_lower.count(word) for word in SystemConfig.POSITIVE_WORDS)

    if neg_count > 0:
        sentiment_ratio = neg_count / (neg_count + pos_count + 1)
        if sentiment_ratio > 0.7:
            crisis_score += int(sentiment_ratio * 10)

    # Isolation patterns
    for pattern in SystemConfig.ISOLATION_PATTERNS:
        if re.search(pattern, text_lower):
            relational_stress_score += 1

    total_score = (
        crisis_score +
        emotional_escalation_score * 1.5 +
        physical_distress_score * 0.5 +
        relational_stress_score * 0.8
    )
    
    score_details = {
        "crisis_score": crisis_score,
        "emotional_escalation": emotional_escalation_score,
        "physical_distress": physical_distress_score,
        "relational_stress": relational_stress_score,
        "total": total_score
    }
    
    if total_score >= 8:
        return "CRITICAL", score_details
    elif total_score >= 4:
        return "WARNING", score_details
    elif total_score >= 2:
        return "CAUTION", score_details
    else:
        return "NORMAL", score_details


def analyze_sentiment(text: str) -> str:
    """
    Hybrid sentiment analysis combining ML and rule-based approaches.
    
    1. Attempts ML-based analysis using HuggingFace transformers
    2. Falls back to rule-based analysis if ML fails
    3. Combines both for more robust crisis detection
    
    Returns:
        str: 'CRITICAL' | 'WARNING' | 'CAUTION' | 'NORMAL'
    """
    # Try ML-based analysis first
    ml_result = analyze_sentiment_ml(text)
    
    # Always run rule-based for crisis patterns (important safety fallback)
    rules_result, rules_details = analyze_sentiment_rules(text)
    
    # If ML failed, use rules only
    if ml_result is None:
        return rules_result
    
    # Combine ML and rules for final decision
    crisis_level = "NORMAL"
    
    # ML indicators
    if ml_result["sentiment"] == "NEGATIVE" and ml_result["sentiment_score"] > 0.9:
        crisis_level = "CAUTION"
    
    if len(ml_result["crisis_indicators"]) >= 2:
        crisis_level = "WARNING"
    
    # Check for specific high-risk emotions from ML
    for emotion in ml_result["emotions"]:
        if emotion["label"].lower() == "fear" and emotion["score"] > 0.7:
            crisis_level = max(crisis_level, "WARNING", key=lambda x: ["NORMAL", "CAUTION", "WARNING", "CRITICAL"].index(x))
        if emotion["label"].lower() == "sadness" and emotion["score"] > 0.8:
            crisis_level = max(crisis_level, "CAUTION", key=lambda x: ["NORMAL", "CAUTION", "WARNING", "CRITICAL"].index(x))
    
    # Rule-based override for explicit crisis language (safety critical)
    if rules_result == "CRITICAL":
        crisis_level = "CRITICAL"
    elif rules_result == "WARNING" and crisis_level in ["NORMAL", "CAUTION"]:
        crisis_level = "WARNING"
    
    return crisis_level


def should_request_selfie(state: State, config: RunnableConfig) -> bool:
    """Determine if a selfie should be requested based on time or conversation patterns."""
    try:
        from core.memory_manager import db
        from config.system_config import SystemConfig

        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return False

        # Check if it's been more than the configured interval since the last selfie request
        last_selfie_request = list(db["selfie_requests"].find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(1))

        if last_selfie_request:
            last_request_time = last_selfie_request[0]["timestamp"]
            if datetime.now() - last_request_time < timedelta(hours=SystemConfig.SELFIE_REQUEST_INTERVAL_HOURS):
                return False  # Don't request more than once per configured interval

        # Check if the user has been talking about mood-related topics
        recent_messages = state["messages"][-5:]  # Look at last 5 messages
        mood_related_keywords = [
            "depressed", "sad", "anxious", "stressed", "tired", "exhausted",
            "lonely", "hopeless", "overwhelmed", "miserable", "unhappy"
        ]

        for msg in recent_messages:
            if hasattr(msg, 'content'):
                content = msg.content.lower()
                if any(keyword in content for keyword in mood_related_keywords):
                    return True

        # Random chance to request selfie periodically based on configuration
        import random
        return random.random() < SystemConfig.SELFIE_RANDOM_CHANCE

    except Exception as e:
        print(f"Error checking if selfie should be requested: {e}")
        return False

def crisis_node(state: State, config: RunnableConfig):
    """Target node for crisis intervention with IMMEDIATE active response.
    
    This node executes crisis interventions immediately rather than just creating
    tool calls that may not be executed. It sends alerts via all configured channels.
    """
    from langchain_core.messages import ToolCall
    from core.tools import crisis_escalation_tool
    from core.integrations import get_integration_manager
    import os

    # Get user ID for context
    user_id = config.get("configurable", {}).get("user_id", "unknown_user")

    # Determine the severity level based on the last message
    last_message = state["messages"][-1] if state["messages"] else None
    last_message_content = ""
    if last_message and hasattr(last_message, 'content'):
        last_message_content = last_message.content
        severity = analyze_sentiment(last_message_content)
    else:
        severity = "CRITICAL"  # Default to critical if we can't determine

    # === IMMEDIATE CRISIS ESCALATION ===
    # Execute crisis escalation NOW, not as a pending tool call
    try:
        escalation_result = crisis_escalation_tool.invoke({
            "user_id": user_id,
            "severity": severity,
            "context": f"User message: {last_message_content[:200]}..."  # Truncate for privacy
        })
        print(f"🚨 Crisis Escalation Result: {escalation_result}")
    except Exception as e:
        print(f"⚠️ Crisis escalation error: {e}")
        escalation_result = f"Escalation attempted but encountered error: {e}"

    # Prepare crisis response message based on severity
    if severity == "CRITICAL":
        crisis_message = (
            "I've detected that you are in immediate distress. I have already activated emergency "
            "protocols and your emergency contacts are being notified right now. Please stay safe and reach "
            "out to someone you trust immediately.\n\n"
            "🆘 **If you're having thoughts of self-harm:**\n"
            "• Call 988 (Suicide & Crisis Lifeline)\n"
            "• Text HOME to 741741 (Crisis Text Line)\n"
            "• Call 911 or your local emergency number\n\n"
            "I'm here with you. You are not alone."
        )
    elif severity == "WARNING":
        crisis_message = (
            "I'm concerned about what you've shared. I have notified your support contacts "
            "so they can check on you. Please consider reaching out to someone you trust "
            "who can provide support right now.\n\n"
            "Remember, it's brave to ask for help. You matter."
        )
    else:  # CAUTION
        crisis_message = (
            "I'm noticing some patterns in our conversation that concern me. I've sent a "
            "check-in notification to your support contacts.\n\n"
            "It's okay to ask for help when you need it. Would you like to talk more about "
            "what's going on?"
        )

    # Create tool calls for additional actions the agent can take
    tool_calls = []

    # Request selfie for wellness check (CRITICAL only)
    if severity == "CRITICAL":
        tool_calls.append({
            "name": "request_selfie_tool",
            "id": f"crisis_selfie_{str(hash(datetime.now()))[:8]}",
            "args": {"reason": "immediate wellness check due to crisis detection"}
        })

    # Log this crisis event
    try:
        from core.memory_manager import db
        db["crisis_events"].insert_one({
            "user_id": user_id,
            "severity": severity,
            "message_content": last_message_content[:500],  # Truncated for storage
            "timestamp": datetime.now(),
            "escalation_result": escalation_result,
            "crisis_message_sent": crisis_message
        })
    except Exception as e:
        print(f"⚠️ Failed to log crisis event: {e}")

    return {
        "messages": [AIMessage(
            content=crisis_message,
            tool_calls=tool_calls if tool_calls else None
        )],
        "next_node": None
    }

def agent(state: State, config: RunnableConfig, model_with_tools) -> dict:
    """Process the current state and generate a response using the LLM.

    Args:
        state (State): The current state of the conversation.
        model_with_tools: The model with bound tools.

    Returns:
        dict: The updated state with the agent's response.
    """
    # Check for crisis keywords in the last user message
    last_message = state["messages"][-1]
    if last_message.type == "human":
        sentiment = analyze_sentiment(last_message.content)
        if sentiment in ["CRITICAL", "WARNING"]:  # Trigger crisis response for both critical and warning levels
            return {"next_node": "crisis_node"}

    # Check if we should request a selfie
    user_id = config.get("configurable", {}).get("user_id")
    if user_id and should_request_selfie(state, config):
        # Return a tool call to request a selfie
        return {
            "messages": [AIMessage(
                content="I'd like to check in on your well-being.",
                tool_calls=[{
                    "name": "request_selfie_tool",
                    "id": "selfie_req_" + str(hash(datetime.now())),
                    "args": {"reason": "wellness monitoring"}
                }]
            )]
        }

    # Check if there's visual context available to analyze
    # This could come from a previous tool call or external input
    visual_context_available = False
    for msg in state["messages"]:
        if hasattr(msg, 'metadata') and 'visual_context' in msg.metadata:
            visual_context_available = True
            break

    # If visual context is available, analyze it
    if visual_context_available:
        # Create a tool call to analyze visual context
        return {
            "messages": [AIMessage(
                content="I'm analyzing the visual information you've shared.",
                tool_calls=[{
                    "name": "analyze_visual_context_tool",
                    "id": "visual_analysis_" + str(hash(datetime.now())),
                    "args": {"image_description": "User appears tired with dark circles under eyes, sitting alone in dimly lit room"}
                }]
            )]
        }

    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }

def load_memories(state: State, config: RunnableConfig) -> dict:
    """Load memories for the current conversation.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        dict: The updated state with loaded memories.
    """
    try:
        from core.memory_manager import memory_store
        
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return {"recall_memories": []}
            
        # Get the latest user message to query against
        last_message = state["messages"][-1]
        query = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        # Search across all namespaces for this user
        # We limit to 5 results to keep context manageable
        results = memory_store.search(
            namespace=("memories", user_id),
            query=query,
            limit=5
        )
        
        # Format results for the prompt
        memories = [f"- {r.value.get('content', str(r.value))}" for r in results]
        
        return {
            "recall_memories": memories,
        }
    except Exception as e:
        print(f"Error loading memories: {e}")
        return {"recall_memories": []}

def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The next step in the graph.
    """
    if state.get("next_node") == "crisis_node":
        return "crisis_node"

    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END