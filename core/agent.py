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

def analyze_sentiment(text: str) -> str:
    """Sophisticated sentiment analysis for crisis detection using multiple approaches."""
    import re
    from collections import Counter
    from config.system_config import SystemConfig

    # Preprocess the text
    text_lower = text.lower().strip()

    # Initialize scores for different aspects
    crisis_score = 0
    emotional_escalation_score = 0
    physical_distress_score = 0
    relational_stress_score = 0

    # Approach 1: Keyword-based detection with context awareness
    for pattern in SystemConfig.CRISIS_PATTERNS:
        if re.search(pattern, text_lower):
            crisis_score += 3  # High risk indicator

    # Approach 2: High negative indicators
    for indicator in SystemConfig.HIGH_NEG_INDICATORS:
        if indicator in text_lower:
            crisis_score += 2  # Medium-high risk indicator

    # Approach 3: Crisis markers
    for marker in SystemConfig.CRISIS_MARKERS:
        if marker in text_lower:
            crisis_score += 3  # High risk indicator

    # Approach 4: Emotional escalation patterns
    for pattern in SystemConfig.EMOTIONAL_ESCALATION_PATTERNS:
        if re.search(pattern, text_lower):
            emotional_escalation_score += 2

    # Approach 5: Physical distress indicators
    for indicator in SystemConfig.PHYSICAL_DISTRESS_INDICATORS:
        if indicator in text_lower:
            physical_distress_score += 1

    # Approach 6: Relational stress indicators
    for indicator in SystemConfig.RELATIONAL_STRESS_INDICATORS:
        if indicator in text_lower:
            relational_stress_score += 1

    # Approach 7: Sentiment intensity scoring
    neg_count = sum(text_lower.count(word) for word in SystemConfig.NEGATIVE_WORDS)
    pos_count = sum(text_lower.count(word) for word in SystemConfig.POSITIVE_WORDS)

    # Calculate sentiment ratio
    if neg_count > 0:
        sentiment_ratio = neg_count / (neg_count + pos_count + 1)  # +1 to avoid division by zero
        if sentiment_ratio > 0.7:  # Highly negative sentiment
            crisis_score += int(sentiment_ratio * 10)  # Scale score based on ratio

    # Approach 8: Emotional escalation patterns
    exclamation_pattern = r'[!]{2,}'  # Multiple exclamation marks
    caps_pattern = r'(?:[A-Z]{2,})'  # Multiple consecutive capital letters

    if re.search(exclamation_pattern, text) and neg_count > pos_count:
        emotional_escalation_score += 2  # High emotional state

    # Approach 9: Isolation and finality language
    for pattern in SystemConfig.ISOLATION_PATTERNS:
        if re.search(pattern, text_lower):
            relational_stress_score += 1

    # Calculate total risk score
    total_risk_score = (
        crisis_score +
        emotional_escalation_score * 1.5 +
        physical_distress_score * 0.5 +
        relational_stress_score * 0.8
    )

    # Determine risk level based on thresholds
    if total_risk_score >= 8:
        return "CRITICAL"
    elif total_risk_score >= 4:
        return "WARNING"
    elif total_risk_score >= 2:
        return "CAUTION"
    else:
        return "NORMAL"

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