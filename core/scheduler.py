"""
Background scheduler for periodic tasks in the LTM application.

This module handles scheduled tasks like:
- Periodic selfie/wellness check requests
- Daily mood check-in reminders
- Memory consolidation
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional
import threading
import time

from config.system_config import SystemConfig

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Background task scheduler for periodic wellness checks and reminders."""
    
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def add_periodic_task(self, task_id: str, callback: Callable, 
                          interval_seconds: int, user_id: str = None):
        """Add a periodic task to the scheduler.
        
        Args:
            task_id: Unique identifier for this task
            callback: Function to call when task runs
            interval_seconds: How often to run this task
            user_id: Optional user ID this task is associated with
        """
        with self._lock:
            self._tasks[task_id] = {
                "callback": callback,
                "interval": interval_seconds,
                "user_id": user_id,
                "last_run": None,
                "next_run": datetime.now() + timedelta(seconds=interval_seconds)
            }
            logger.info(f"Added periodic task: {task_id} (interval: {interval_seconds}s)")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Removed task: {task_id}")
                return True
            return False
    
    def _run_loop(self):
        """Main scheduler loop that checks and runs due tasks."""
        while self._running:
            now = datetime.now()
            
            with self._lock:
                tasks_to_run = []
                for task_id, task in self._tasks.items():
                    if task["next_run"] <= now:
                        tasks_to_run.append((task_id, task))
            
            for task_id, task in tasks_to_run:
                try:
                    logger.info(f"Running scheduled task: {task_id}")
                    task["callback"](task.get("user_id"))
                    
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id]["last_run"] = now
                            self._tasks[task_id]["next_run"] = now + timedelta(
                                seconds=task["interval"]
                            )
                except Exception as e:
                    logger.error(f"Error running task {task_id}: {e}")
            
            # Sleep for a short interval before checking again
            time.sleep(10)  # Check every 10 seconds
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Task scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Task scheduler stopped")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks."""
        with self._lock:
            return {
                task_id: {
                    "interval": task["interval"],
                    "last_run": task["last_run"].isoformat() if task["last_run"] else None,
                    "next_run": task["next_run"].isoformat() if task["next_run"] else None,
                    "user_id": task.get("user_id")
                }
                for task_id, task in self._tasks.items()
            }


class WellnessScheduler:
    """Specialized scheduler for wellness-related tasks."""
    
    def __init__(self, service=None):
        self.scheduler = TaskScheduler()
        self.service = service
        self._user_tasks: Dict[str, list] = {}  # Track tasks per user
    
    def schedule_selfie_request(self, user_id: str):
        """Schedule periodic selfie requests for a user.
        
        Args:
            user_id: The user to schedule selfie requests for
        """
        interval_hours = SystemConfig.SELFIE_REQUEST_INTERVAL_HOURS
        interval_seconds = interval_hours * 3600
        
        task_id = f"selfie_request_{user_id}"
        
        def request_selfie(uid):
            """Callback to request a selfie from the user."""
            try:
                from core.memory_manager import db
                
                # Log the scheduled selfie request
                db["selfie_requests"].insert_one({
                    "user_id": uid,
                    "reason": "scheduled_wellness_check",
                    "timestamp": datetime.now(),
                    "status": "scheduled",
                    "source": "scheduler"
                })
                
                # If service is available, send a notification
                if self.service:
                    self.service.send_notification(
                        user_id=uid,
                        message="Time for a wellness check! How are you feeling right now?",
                        notification_type="info"
                    )
                    
                logger.info(f"Scheduled selfie request triggered for user {uid}")
                
            except Exception as e:
                logger.error(f"Failed to trigger selfie request for {uid}: {e}")
        
        self.scheduler.add_periodic_task(
            task_id=task_id,
            callback=request_selfie,
            interval_seconds=interval_seconds,
            user_id=user_id
        )
        
        # Track this task for the user
        if user_id not in self._user_tasks:
            self._user_tasks[user_id] = []
        self._user_tasks[user_id].append(task_id)
        
        logger.info(f"Scheduled selfie requests for user {user_id} every {interval_hours} hours")
    
    def schedule_daily_mood_checkin(self, user_id: str, hour: int = 9):
        """Schedule a daily mood check-in for a user.
        
        Args:
            user_id: The user to schedule check-ins for
            hour: Hour of day (0-23) to send the check-in
        """
        # Calculate seconds until the next occurrence of this hour
        now = datetime.now()
        next_checkin = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if next_checkin <= now:
            next_checkin += timedelta(days=1)
        
        initial_delay = (next_checkin - now).total_seconds()
        
        task_id = f"mood_checkin_{user_id}"
        
        def mood_checkin(uid):
            """Callback for daily mood check-in."""
            try:
                if self.service:
                    self.service.send_notification(
                        user_id=uid,
                        message="Good morning! How are you feeling today? Take a moment to log your mood.",
                        notification_type="info"
                    )
                logger.info(f"Daily mood check-in sent to user {uid}")
            except Exception as e:
                logger.error(f"Failed to send mood check-in to {uid}: {e}")
        
        # Schedule with 24-hour interval (86400 seconds)
        self.scheduler.add_periodic_task(
            task_id=task_id,
            callback=mood_checkin,
            interval_seconds=86400,  # 24 hours
            user_id=user_id
        )
        
        if user_id not in self._user_tasks:
            self._user_tasks[user_id] = []
        self._user_tasks[user_id].append(task_id)
        
        logger.info(f"Scheduled daily mood check-in for user {user_id} at {hour}:00")
    
    def remove_user_tasks(self, user_id: str):
        """Remove all scheduled tasks for a user."""
        if user_id in self._user_tasks:
            for task_id in self._user_tasks[user_id]:
                self.scheduler.remove_task(task_id)
            del self._user_tasks[user_id]
            logger.info(f"Removed all scheduled tasks for user {user_id}")
    
    def start(self):
        """Start the wellness scheduler."""
        self.scheduler.start()
    
    def stop(self):
        """Stop the wellness scheduler."""
        self.scheduler.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all wellness tasks."""
        return {
            "running": self.scheduler._running,
            "tasks": self.scheduler.get_task_status(),
            "users_with_tasks": list(self._user_tasks.keys())
        }


# Global wellness scheduler instance
_wellness_scheduler: Optional[WellnessScheduler] = None


def get_wellness_scheduler(service=None) -> WellnessScheduler:
    """Get or create the global wellness scheduler instance."""
    global _wellness_scheduler
    if _wellness_scheduler is None:
        _wellness_scheduler = WellnessScheduler(service=service)
    elif service is not None and _wellness_scheduler.service is None:
        _wellness_scheduler.service = service
    return _wellness_scheduler
