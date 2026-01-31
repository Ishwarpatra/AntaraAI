"""
Gamification Engine for the LTM Therapeutic Application.

This module implements a complete gamification system with:
- XP and leveling (non-linear curve)
- Streak tracking with loss aversion
- Badge system for therapeutic milestones
- MongoDB persistence

All state is persisted to MongoDB in the 'user_progress' collection.
"""

import math
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class BadgeCategory(str, Enum):
    """Categories of badges for organization."""
    CONSISTENCY = "consistency"
    EMOTIONAL_AWARENESS = "emotional_awareness"
    ACTION_ORIENTED = "action_oriented"
    MILESTONE = "milestone"
    SPECIAL = "special"


class Badge(BaseModel):
    """Badge definition with metadata."""
    id: str = Field(..., description="Unique badge identifier")
    name: str = Field(..., description="Display name of the badge")
    description: str = Field(..., description="How to earn this badge")
    icon: str = Field(..., description="Emoji or icon identifier")
    xp_bonus: int = Field(default=0, description="XP awarded when badge is earned")
    category: BadgeCategory = Field(..., description="Badge category")
    requirement_value: int = Field(default=1, description="Numeric requirement to earn")


class UserProgress(BaseModel):
    """User progress tracking model - persisted to MongoDB."""
    user_id: str = Field(..., description="Unique user identifier")
    total_xp: int = Field(default=0, description="Total XP accumulated")
    current_level: int = Field(default=1, description="Current level (calculated from XP)")
    current_streak: int = Field(default=0, description="Current consecutive day streak")
    longest_streak: int = Field(default=0, description="All-time longest streak")
    last_activity_date: Optional[date] = Field(default=None, description="Last activity date (calendar day)")
    unlocked_badges: List[str] = Field(default_factory=list, description="List of unlocked badge IDs")
    
    # Activity counters for badge tracking
    mood_logs_count: int = Field(default=0, description="Total mood logs")
    music_sessions_count: int = Field(default=0, description="Music therapy sessions completed")
    selfies_taken: int = Field(default=0, description="Wellness selfies taken")
    crisis_interventions_received: int = Field(default=0, description="Times user received crisis support")
    cbt_exercises_completed: int = Field(default=0, description="CBT exercises completed")
    dbt_exercises_completed: int = Field(default=0, description="DBT exercises completed")
    conversations_count: int = Field(default=0, description="Total conversation sessions")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
            datetime: lambda v: v.isoformat() if v else None
        }


class GamificationResult(BaseModel):
    """Result of a gamification action - returned to tools for LLM feedback."""
    xp_earned: int = Field(default=0)
    total_xp: int = Field(default=0)
    level: int = Field(default=1)
    level_up: bool = Field(default=False, description="Did user level up from this action?")
    previous_level: int = Field(default=1)
    streak: int = Field(default=0)
    streak_change: str = Field(default="maintained", description="'increased', 'maintained', 'reset'")
    new_badges: List[Badge] = Field(default_factory=list)
    message: str = Field(default="", description="Human-readable summary")


# =============================================================================
# BADGE DEFINITIONS
# =============================================================================

BADGE_DEFINITIONS: Dict[str, Badge] = {
    # Consistency badges (streak-based)
    "streak_3": Badge(
        id="streak_3",
        name="Getting Started",
        description="Maintain a 3-day check-in streak",
        icon="🌱",
        xp_bonus=25,
        category=BadgeCategory.CONSISTENCY,
        requirement_value=3
    ),
    "streak_7": Badge(
        id="streak_7",
        name="Week Warrior",
        description="Maintain a 7-day check-in streak",
        icon="🔥",
        xp_bonus=75,
        category=BadgeCategory.CONSISTENCY,
        requirement_value=7
    ),
    "streak_14": Badge(
        id="streak_14",
        name="Fortnight Fighter",
        description="Maintain a 14-day check-in streak",
        icon="⚡",
        xp_bonus=150,
        category=BadgeCategory.CONSISTENCY,
        requirement_value=14
    ),
    "streak_30": Badge(
        id="streak_30",
        name="Monthly Master",
        description="Maintain a 30-day check-in streak",
        icon="🏆",
        xp_bonus=300,
        category=BadgeCategory.CONSISTENCY,
        requirement_value=30
    ),
    "streak_100": Badge(
        id="streak_100",
        name="Century Champion",
        description="Maintain a 100-day check-in streak",
        icon="👑",
        xp_bonus=1000,
        category=BadgeCategory.CONSISTENCY,
        requirement_value=100
    ),
    
    # Emotional Awareness badges (mood logging)
    "mood_1": Badge(
        id="mood_1",
        name="First Reflection",
        description="Log your mood for the first time",
        icon="📝",
        xp_bonus=10,
        category=BadgeCategory.EMOTIONAL_AWARENESS,
        requirement_value=1
    ),
    "mood_10": Badge(
        id="mood_10",
        name="Mood Monitor",
        description="Log your mood 10 times",
        icon="🎭",
        xp_bonus=50,
        category=BadgeCategory.EMOTIONAL_AWARENESS,
        requirement_value=10
    ),
    "mood_50": Badge(
        id="mood_50",
        name="Emotional Explorer",
        description="Log your mood 50 times",
        icon="🧭",
        xp_bonus=150,
        category=BadgeCategory.EMOTIONAL_AWARENESS,
        requirement_value=50
    ),
    "mood_100": Badge(
        id="mood_100",
        name="Self-Awareness Sage",
        description="Log your mood 100 times",
        icon="🔮",
        xp_bonus=300,
        category=BadgeCategory.EMOTIONAL_AWARENESS,
        requirement_value=100
    ),
    
    # Action-Oriented badges (therapy engagement)
    "music_1": Badge(
        id="music_1",
        name="First Melody",
        description="Complete your first music therapy session",
        icon="🎵",
        xp_bonus=15,
        category=BadgeCategory.ACTION_ORIENTED,
        requirement_value=1
    ),
    "music_10": Badge(
        id="music_10",
        name="Rhythm Regular",
        description="Complete 10 music therapy sessions",
        icon="🎶",
        xp_bonus=75,
        category=BadgeCategory.ACTION_ORIENTED,
        requirement_value=10
    ),
    "music_25": Badge(
        id="music_25",
        name="Harmony Hunter",
        description="Complete 25 music therapy sessions",
        icon="🎸",
        xp_bonus=200,
        category=BadgeCategory.ACTION_ORIENTED,
        requirement_value=25
    ),
    "selfie_1": Badge(
        id="selfie_1",
        name="Face Forward",
        description="Take your first wellness selfie",
        icon="📸",
        xp_bonus=10,
        category=BadgeCategory.ACTION_ORIENTED,
        requirement_value=1
    ),
    "selfie_10": Badge(
        id="selfie_10",
        name="Visual Voyager",
        description="Take 10 wellness selfies",
        icon="🖼️",
        xp_bonus=50,
        category=BadgeCategory.ACTION_ORIENTED,
        requirement_value=10
    ),
    
    # Milestone badges
    "level_5": Badge(
        id="level_5",
        name="Rising Star",
        description="Reach Level 5",
        icon="⭐",
        xp_bonus=50,
        category=BadgeCategory.MILESTONE,
        requirement_value=5
    ),
    "level_10": Badge(
        id="level_10",
        name="Shining Bright",
        description="Reach Level 10",
        icon="🌟",
        xp_bonus=100,
        category=BadgeCategory.MILESTONE,
        requirement_value=10
    ),
    "level_25": Badge(
        id="level_25",
        name="Supernova",
        description="Reach Level 25",
        icon="💫",
        xp_bonus=250,
        category=BadgeCategory.MILESTONE,
        requirement_value=25
    ),
    "conversations_10": Badge(
        id="conversations_10",
        name="Open Book",
        description="Have 10 therapy conversations",
        icon="💬",
        xp_bonus=50,
        category=BadgeCategory.MILESTONE,
        requirement_value=10
    ),
    "conversations_50": Badge(
        id="conversations_50",
        name="Deep Diver",
        description="Have 50 therapy conversations",
        icon="🌊",
        xp_bonus=200,
        category=BadgeCategory.MILESTONE,
        requirement_value=50
    ),
    
    # Special badges
    "crisis_survivor": Badge(
        id="crisis_survivor",
        name="Crisis Navigator",
        description="Successfully worked through a crisis moment",
        icon="💪",
        xp_bonus=100,
        category=BadgeCategory.SPECIAL,
        requirement_value=1
    ),
    "comeback": Badge(
        id="comeback",
        name="Phoenix Rising",
        description="Return after losing a streak (shows resilience)",
        icon="🔄",
        xp_bonus=25,
        category=BadgeCategory.SPECIAL,
        requirement_value=1
    ),
}


# =============================================================================
# GAMIFICATION ENGINE
# =============================================================================

class GamificationEngine:
    """
    Core gamification engine handling XP, levels, streaks, and badges.
    
    All state is persisted to MongoDB. The engine is stateless itself;
    it reads/writes user progress from the database on each operation.
    """
    
    # XP rewards for different actions
    XP_REWARDS = {
        "mood_log": 10,
        "music_therapy": 15,
        "selfie_taken": 10,
        "conversation": 5,
        "crisis_support": 20,
        "cbt_exercise": 25,
        "dbt_exercise": 25,
        "daily_check_in": 5,
    }
    
    # Level curve: Level = floor(sqrt(XP / 10))
    # This means: L1=0XP, L2=40XP, L3=90XP, L4=160XP, L5=250XP, etc.
    LEVEL_DIVISOR = 10
    
    def __init__(self, db=None):
        """
        Initialize the gamification engine with a MongoDB database.
        
        Args:
            db: PyMongo database object (should be the same as memory_manager.db)
        """
        self.db = db
        self.collection_name = "user_progress"
        
    def _get_collection(self):
        """Get the user_progress collection."""
        if self.db is None:
            raise RuntimeError("Gamification engine requires MongoDB connection")
        return self.db[self.collection_name]
    
    def calculate_level(self, xp: int) -> int:
        """
        Calculate level from XP using non-linear curve.
        
        Formula: Level = floor(sqrt(XP / 10)) + 1
        
        This creates an accelerating XP requirement:
        - Level 1: 0 XP
        - Level 2: 10 XP (need 10)
        - Level 3: 40 XP (need 30 more)
        - Level 4: 90 XP (need 50 more)
        - Level 5: 160 XP (need 70 more)
        - Level 10: 810 XP
        - Level 25: 5,760 XP
        """
        return int(math.sqrt(xp / self.LEVEL_DIVISOR)) + 1
    
    def xp_for_level(self, level: int) -> int:
        """Calculate XP required to reach a specific level."""
        return ((level - 1) ** 2) * self.LEVEL_DIVISOR
    
    def xp_to_next_level(self, current_xp: int) -> int:
        """Calculate XP needed to reach the next level."""
        current_level = self.calculate_level(current_xp)
        next_level_xp = self.xp_for_level(current_level + 1)
        return next_level_xp - current_xp
    
    def get_user_progress(self, user_id: str) -> UserProgress:
        """
        Get or create user progress from MongoDB.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            UserProgress model (created if doesn't exist)
        """
        collection = self._get_collection()
        doc = collection.find_one({"user_id": user_id})
        
        if doc:
            # Convert MongoDB doc to Pydantic model
            doc.pop("_id", None)
            # Handle date conversion
            if doc.get("last_activity_date"):
                if isinstance(doc["last_activity_date"], str):
                    doc["last_activity_date"] = date.fromisoformat(doc["last_activity_date"])
                elif isinstance(doc["last_activity_date"], datetime):
                    doc["last_activity_date"] = doc["last_activity_date"].date()
            return UserProgress(**doc)
        else:
            # Create new user progress
            progress = UserProgress(user_id=user_id)
            self._save_progress(progress)
            return progress
    
    def _save_progress(self, progress: UserProgress) -> None:
        """Save user progress to MongoDB."""
        collection = self._get_collection()
        progress.updated_at = datetime.now()
        
        # Convert to dict for MongoDB
        doc = progress.model_dump()
        
        # Convert date to string for MongoDB
        if doc.get("last_activity_date"):
            doc["last_activity_date"] = doc["last_activity_date"].isoformat()
        
        collection.update_one(
            {"user_id": progress.user_id},
            {"$set": doc},
            upsert=True
        )
    
    def _calculate_streak(self, progress: UserProgress, check_in_date: date) -> tuple:
        """
        Calculate streak based on last activity date.
        
        Returns:
            tuple: (new_streak, streak_change: 'increased'|'maintained'|'reset')
        """
        if progress.last_activity_date is None:
            # First ever check-in
            return 1, "increased"
        
        last_date = progress.last_activity_date
        if isinstance(last_date, datetime):
            last_date = last_date.date()
        
        days_diff = (check_in_date - last_date).days
        
        if days_diff == 0:
            # Same day - idempotent, maintain streak
            return progress.current_streak, "maintained"
        elif days_diff == 1:
            # Consecutive day - increment streak
            return progress.current_streak + 1, "increased"
        else:
            # Gap > 1 day - reset streak
            return 1, "reset"
    
    def _check_and_award_badges(self, progress: UserProgress) -> List[Badge]:
        """
        Check all badge conditions and award any newly earned badges.
        
        Returns:
            List of newly awarded Badge objects
        """
        new_badges = []
        
        # Streak badges
        streak_badges = [
            ("streak_3", 3),
            ("streak_7", 7),
            ("streak_14", 14),
            ("streak_30", 30),
            ("streak_100", 100),
        ]
        for badge_id, required_streak in streak_badges:
            if badge_id not in progress.unlocked_badges and progress.current_streak >= required_streak:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Mood logging badges
        mood_badges = [
            ("mood_1", 1),
            ("mood_10", 10),
            ("mood_50", 50),
            ("mood_100", 100),
        ]
        for badge_id, required_count in mood_badges:
            if badge_id not in progress.unlocked_badges and progress.mood_logs_count >= required_count:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Music therapy badges
        music_badges = [
            ("music_1", 1),
            ("music_10", 10),
            ("music_25", 25),
        ]
        for badge_id, required_count in music_badges:
            if badge_id not in progress.unlocked_badges and progress.music_sessions_count >= required_count:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Selfie badges
        selfie_badges = [
            ("selfie_1", 1),
            ("selfie_10", 10),
        ]
        for badge_id, required_count in selfie_badges:
            if badge_id not in progress.unlocked_badges and progress.selfies_taken >= required_count:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Level badges
        level_badges = [
            ("level_5", 5),
            ("level_10", 10),
            ("level_25", 25),
        ]
        for badge_id, required_level in level_badges:
            if badge_id not in progress.unlocked_badges and progress.current_level >= required_level:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Conversation badges
        convo_badges = [
            ("conversations_10", 10),
            ("conversations_50", 50),
        ]
        for badge_id, required_count in convo_badges:
            if badge_id not in progress.unlocked_badges and progress.conversations_count >= required_count:
                new_badges.append(BADGE_DEFINITIONS[badge_id])
                progress.unlocked_badges.append(badge_id)
        
        # Crisis survivor badge
        if "crisis_survivor" not in progress.unlocked_badges and progress.crisis_interventions_received >= 1:
            new_badges.append(BADGE_DEFINITIONS["crisis_survivor"])
            progress.unlocked_badges.append("crisis_survivor")
        
        return new_badges
    
    def record_activity(
        self,
        user_id: str,
        activity_type: str,
        custom_xp: Optional[int] = None
    ) -> GamificationResult:
        """
        Record a user activity and calculate all gamification updates.
        
        This is the main entry point for gamification. Call this whenever
        the user performs a tracked action.
        
        Args:
            user_id: The user's unique identifier
            activity_type: Type of activity (mood_log, music_therapy, etc.)
            custom_xp: Optional override for XP award
            
        Returns:
            GamificationResult with all updates
        """
        # Get current progress
        progress = self.get_user_progress(user_id)
        previous_level = progress.current_level
        
        # Calculate streak
        today = date.today()
        new_streak, streak_change = self._calculate_streak(progress, today)
        
        # Check if streak was reset and award comeback badge
        if streak_change == "reset" and progress.current_streak >= 3:
            # User had a meaningful streak and is coming back
            if "comeback" not in progress.unlocked_badges:
                progress.unlocked_badges.append("comeback")
        
        # Update streak
        progress.current_streak = new_streak
        if new_streak > progress.longest_streak:
            progress.longest_streak = new_streak
        progress.last_activity_date = today
        
        # Calculate XP
        base_xp = custom_xp if custom_xp is not None else self.XP_REWARDS.get(activity_type, 5)
        
        # Update activity counters
        if activity_type == "mood_log":
            progress.mood_logs_count += 1
        elif activity_type == "music_therapy":
            progress.music_sessions_count += 1
        elif activity_type == "selfie_taken":
            progress.selfies_taken += 1
        elif activity_type == "conversation":
            progress.conversations_count += 1
        elif activity_type == "crisis_support":
            progress.crisis_interventions_received += 1
        elif activity_type == "cbt_exercise":
            progress.cbt_exercises_completed += 1
        elif activity_type == "dbt_exercise":
            progress.dbt_exercises_completed += 1
        
        # Check for new badges BEFORE adding XP (so level badges trigger correctly)
        new_badges = self._check_and_award_badges(progress)
        
        # Add badge bonus XP
        badge_xp = sum(badge.xp_bonus for badge in new_badges)
        total_xp_earned = base_xp + badge_xp
        
        # Update total XP and recalculate level
        progress.total_xp += total_xp_earned
        progress.current_level = self.calculate_level(progress.total_xp)
        
        # Check for level-up badges after XP is added
        level_badges = self._check_and_award_badges(progress)
        new_badges.extend(level_badges)
        badge_xp += sum(badge.xp_bonus for badge in level_badges)
        progress.total_xp += sum(badge.xp_bonus for badge in level_badges)
        progress.current_level = self.calculate_level(progress.total_xp)
        
        # Save progress
        self._save_progress(progress)
        
        # Build result message
        level_up = progress.current_level > previous_level
        
        message_parts = [f"+{total_xp_earned} XP"]
        
        if streak_change == "increased":
            message_parts.append(f"🔥 Streak: {progress.current_streak} days")
        elif streak_change == "reset":
            message_parts.append(f"Streak reset to 1 day")
        
        if level_up:
            message_parts.append(f"🎉 LEVEL UP! Now Level {progress.current_level}")
        
        if new_badges:
            badge_names = [f"{b.icon} {b.name}" for b in new_badges]
            message_parts.append(f"New badges: {', '.join(badge_names)}")
        
        return GamificationResult(
            xp_earned=total_xp_earned,
            total_xp=progress.total_xp,
            level=progress.current_level,
            level_up=level_up,
            previous_level=previous_level,
            streak=progress.current_streak,
            streak_change=streak_change,
            new_badges=new_badges,
            message=" | ".join(message_parts)
        )
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by XP."""
        collection = self._get_collection()
        top_users = list(collection.find(
            {},
            {"_id": 0, "user_id": 1, "total_xp": 1, "current_level": 1, "current_streak": 1}
        ).sort("total_xp", -1).limit(limit))
        return top_users
    
    def get_user_badges(self, user_id: str) -> List[Badge]:
        """Get all badges unlocked by a user."""
        progress = self.get_user_progress(user_id)
        return [BADGE_DEFINITIONS[bid] for bid in progress.unlocked_badges if bid in BADGE_DEFINITIONS]
    
    def get_available_badges(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all badges with progress towards each."""
        progress = self.get_user_progress(user_id)
        result = []
        
        for badge_id, badge in BADGE_DEFINITIONS.items():
            current_value = 0
            
            # Determine current progress based on badge type
            if badge_id.startswith("streak_"):
                current_value = progress.current_streak
            elif badge_id.startswith("mood_"):
                current_value = progress.mood_logs_count
            elif badge_id.startswith("music_"):
                current_value = progress.music_sessions_count
            elif badge_id.startswith("selfie_"):
                current_value = progress.selfies_taken
            elif badge_id.startswith("level_"):
                current_value = progress.current_level
            elif badge_id.startswith("conversations_"):
                current_value = progress.conversations_count
            elif badge_id == "crisis_survivor":
                current_value = progress.crisis_interventions_received
            elif badge_id == "comeback":
                current_value = 1 if badge_id in progress.unlocked_badges else 0
            
            result.append({
                "badge": badge.model_dump(),
                "unlocked": badge_id in progress.unlocked_badges,
                "progress": min(current_value / badge.requirement_value * 100, 100),
                "current_value": current_value,
                "required_value": badge.requirement_value
            })
        
        return result


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_gamification_engine: Optional[GamificationEngine] = None


def get_gamification_engine() -> GamificationEngine:
    """Get or create the global gamification engine instance."""
    global _gamification_engine
    
    if _gamification_engine is None:
        try:
            from core.memory_manager import db
            _gamification_engine = GamificationEngine(db=db)
            print("✅ Gamification engine initialized with MongoDB")
        except Exception as e:
            print(f"⚠️ Failed to initialize gamification engine: {e}")
            # Create without DB (will fail on use, but won't crash import)
            _gamification_engine = GamificationEngine(db=None)
    
    return _gamification_engine
