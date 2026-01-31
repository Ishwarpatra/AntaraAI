"""
Integration module for external systems (WhatsApp, Telegram, EHR).
"""

import os
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import json

from config.system_config import SystemConfig


class ExternalIntegration(ABC):
    """Abstract base class for external integrations."""
    
    @abstractmethod
    def send_message(self, recipient: str, message: str) -> bool:
        """Send a message to the specified recipient."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the integration is properly configured and available."""
        pass


class WhatsAppIntegration(ExternalIntegration):
    """WhatsApp Business API integration."""
    
    def __init__(self):
        self.api_url = os.getenv("WHATSAPP_API_URL", "https://graph.facebook.com/v18.0")
        self.access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    
    def is_available(self) -> bool:
        """Check if WhatsApp integration is properly configured."""
        return all([self.access_token, self.phone_number_id])
    
    def send_message(self, recipient: str, message: str) -> bool:
        """Send a message via WhatsApp Business API."""
        if not self.is_available():
            print("WhatsApp integration not configured")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/{self.phone_number_id}/messages",
                headers=headers,
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False


class TelegramIntegration(ExternalIntegration):
    """Telegram Bot API integration."""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""
    
    def is_available(self) -> bool:
        """Check if Telegram integration is properly configured."""
        return bool(self.bot_token)
    
    def send_message(self, recipient: str, message: str) -> bool:
        """Send a message via Telegram Bot API."""
        if not self.is_available():
            print("Telegram integration not configured")
            return False
        
        payload = {
            "chat_id": recipient,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False


class TwilioIntegration(ExternalIntegration):
    """Twilio SMS integration."""
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self._client = None
        # Lazy import to avoid hard dependency at import-time
        try:
            from twilio.rest import Client
            if self.is_available():
                self._client = Client(self.account_sid, self.auth_token)
        except Exception:
            self._client = None

    def is_available(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number and self._client)

    def send_message(self, recipient: str, message: str) -> bool:
        if not self.is_available():
            print("Twilio integration not configured")
            return False
        try:
            sent = self._client.messages.create(
                body=message,
                from_=self.from_number,
                to=recipient
            )
            return bool(getattr(sent, 'sid', None))
        except Exception as e:
            print(f"Error sending Twilio message: {e}")
            return False


class EHRIntegration:
    """Electronic Health Records integration."""
    
    def __init__(self):
        self.api_url = os.getenv("EHR_API_URL")
        self.api_key = os.getenv("EHR_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if self.api_key else {}
    
    def is_available(self) -> bool:
        """Check if EHR integration is properly configured."""
        return bool(self.api_url and self.api_key)
    
    def log_patient_note(self, patient_id: str, note: str, category: str = "general") -> bool:
        """Log a clinical note to the patient's EHR."""
        if not self.is_available():
            print("EHR integration not configured")
            return False
        
        payload = {
            "patient_id": patient_id,
            "note": note,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "source": "AntaraAI_LTMAgent"
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/patients/{patient_id}/notes",
                headers=self.headers,
                json=payload
            )
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"Error logging to EHR: {e}")
            return False
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient information from EHR."""
        if not self.is_available():
            print("EHR integration not configured")
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/patients/{patient_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error retrieving patient info from EHR: {e}")
            return None


class IntegrationManager:
    """Manages all external integrations."""
    
    def __init__(self):
        self.whatsapp = WhatsAppIntegration()
        self.telegram = TelegramIntegration()
        self.twilio = TwilioIntegration()
        self.ehr = EHRIntegration()
        
        # Log integration statuses
        print(f"WhatsApp integration: {'Available' if self.whatsapp.is_available() else 'Not configured'}")
        print(f"Telegram integration: {'Available' if self.telegram.is_available() else 'Not configured'}")
        print(f"Twilio integration: {'Available' if self.twilio.is_available() else 'Not configured'}")
        print(f"EHR integration: {'Available' if self.ehr.is_available() else 'Not configured'}")
    
    def send_crisis_alert(self, message: str, recipients: Dict[str, list]) -> Dict[str, bool]:
        """Send crisis alerts to configured contacts via multiple channels."""
        results = {}
        
        # Send via WhatsApp if configured and recipients provided
        if self.whatsapp.is_available() and recipients.get("whatsapp"):
            results["whatsapp"] = []
            for recipient in recipients["whatsapp"]:
                success = self.whatsapp.send_message(recipient, message)
                results["whatsapp"].append({"recipient": recipient, "success": success})
        
        # Send via Telegram if configured and recipients provided
        if self.telegram.is_available() and recipients.get("telegram"):
            results["telegram"] = []
            for recipient in recipients["telegram"]:
                success = self.telegram.send_message(recipient, message)
                results["telegram"].append({"recipient": recipient, "success": success})
        
        # Log to EHR if configured
        if self.ehr.is_available() and recipients.get("ehr_patients"):
            results["ehr"] = []
            for patient_id in recipients["ehr_patients"]:
                success = self.ehr.log_patient_note(patient_id, message, "crisis_alert")
                results["ehr"].append({"patient_id": patient_id, "success": success})
        
        return results
    
    def log_session_to_ehr(self, patient_id: str, session_data: Dict[str, Any]) -> bool:
        """Log a therapy session to the patient's EHR."""
        if not self.ehr.is_available():
            return False
        
        note = f"Therapy session log: {session_data.get('summary', 'Session details')}"
        return self.ehr.log_patient_note(patient_id, note, "therapy_session")
    
    def get_patient_context(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient context from EHR to inform therapy."""
        return self.ehr.get_patient_info(patient_id)


# Global integration manager instance
integration_manager = IntegrationManager()


def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager instance."""
    return integration_manager