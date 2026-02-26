from datetime import datetime, timedelta
from pymongo import MongoClient
import os

class GamificationManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = os.getenv("DATABASE_NAME", "ltm_database")
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]
        self.gamification_collection = self.db["gamification"]

        # Ensure a document exists for the user
        self.gamification_collection.update_one(
            {"user_id": self.user_id},
            {"$setOnInsert": {"xp": 0, "streak_days": 0, "last_mood_log_date": None}},
            upsert=True
        )

    def _get_user_data(self):
        return self.gamification_collection.find_one({"user_id": self.user_id})

    def _update_user_data(self, data):
        self.gamification_collection.update_one({"user_id": self.user_id}, {"$set": data})

    def log_mood_event(self) -> dict:
        """
        Logs a mood event, updates streak, and awards XP.
        Returns a dictionary with XP gained and current streak.
        """
        user_data = self._get_user_data()
        current_xp = user_data.get("xp", 0)
        current_streak = user_data.get("streak_days", 0)
        last_log_date = user_data.get("last_mood_log_date")

        xp_gained = 0
        streak_message = ""
        current_date = datetime.utcnow().date()

        if last_log_date:
            last_log_datetime = datetime.combine(last_log_date, datetime.min.time())
            
            # Check if today is the day after the last log to continue streak
            if (current_date - last_log_datetime.date()) == timedelta(days=1):
                current_streak += 1
                xp_gained += 10 + (current_streak * 2)  # More XP for longer streaks
                streak_message = f"Mood log streak continued! You're on a {current_streak}-day streak!"
            # If logged today already, no change to streak, no XP
            elif current_date == last_log_datetime.date():
                xp_gained = 0 # No XP for logging multiple times a day
                streak_message = f"You've already logged your mood today. Current streak: {current_streak} days."
            # If missed a day, reset streak
            else:
                current_streak = 1
                xp_gained += 10
                streak_message = "Streak reset. New streak started: 1 day!"
        else:
            # First mood log
            current_streak = 1
            xp_gained += 10
            streak_message = "First mood log recorded! New streak started: 1 day!"

        new_xp = current_xp + xp_gained
        
        self._update_user_data({
            "xp": new_xp,
            "streak_days": current_streak,
            "last_mood_log_date": current_date
        })

        return {
            "xp_gained": xp_gained,
            "total_xp": new_xp,
            "current_streak": current_streak,
            "streak_message": streak_message
        }

    def get_user_progress(self) -> dict:
        """
        Retrieves the user's current gamification progress.
        """
        user_data = self._get_user_data()
        return {
            "xp": user_data.get("xp", 0),
            "streak_days": user_data.get("streak_days", 0),
            "last_mood_log_date": str(user_data.get("last_mood_log_date")) if user_data.get("last_mood_log_date") else "N/A"
        }

if __name__ == "__main__":
    # Example usage for testing
    user_id_test = "test_user_123"
    gm = GamificationManager(user_id_test)

    print("--- Initial Progress ---")
    print(gm.get_user_progress())

    print("--- First Mood Log ---")
    result = gm.log_mood_event()
    print(result)
    print(gm.get_user_progress())

    print("--- Second Mood Log (same day) ---")
    result = gm.log_mood_event()
    print(result)
    print(gm.get_user_progress())

    # Simulate next day
    print("--- Simulate Next Day and Log Mood ---")
    # Manually adjust last_mood_log_date for testing streak continuation
    user_data = gm._get_user_data()
    if user_data["last_mood_log_date"]:
        user_data["last_mood_log_date"] = user_data["last_mood_log_date"] - timedelta(days=1)
        gm._update_user_data(user_data)
    result = gm.log_mood_event()
    print(result)
    print(gm.get_user_progress())
    
    # Simulate skipping a day
    print("--- Simulate Skipping a Day and Log Mood ---")
    user_data = gm._get_user_data()
    if user_data["last_mood_log_date"]:
        user_data["last_mood_log_date"] = user_data["last_mood_log_date"] - timedelta(days=2) # Skip a day
        gm._update_user_data(user_data)
    result = gm.log_mood_event()
    print(result)
    print(gm.get_user_progress())

    # Clean up test data
    # gm.gamification_collection.delete_one({"user_id": user_id_test})