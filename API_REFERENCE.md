# Antara AI - API Reference

This document provides a detailed reference for the Antara AI REST API.

## Base URL

The API is served from `http://127.0.0.1:8000`.

## Authentication

The current version of the API does not implement authentication. This is a critical feature to be added in a future release.

## Endpoints

### Health & Status

#### `GET /`

- **Description:** Root endpoint for a basic health check.
- **Response:**
  - `200 OK`: `{"message": "LTMAgent API is running", "status": "healthy"}`

#### `GET /health`

- **Description:** Health check endpoint.
- **Response:**
  - `200 OK`: `{"status": "healthy", "timestamp": "..."}`

#### `GET /model/info`

- **Description:** Get information about the currently loaded model.
- **Response:**
  - `200 OK`: A JSON object with model information.

### User and Thread Management

#### `GET /users`

- **Description:** Get a list of all user IDs.
- **Response:**
  - `200 OK`: `{"users": ["user1", "user2", ...]}`

#### `POST /users`

- **Description:** Create a new user.
- **Request Body:**
  ```json
  {
    "user_id": "string (optional)"
  }
  ```
- **Response:**
  - `200 OK`: `{"user_id": "new_user_id"}`

#### `POST /threads`

- **Description:** Create a new conversation thread for a user.
- **Request Body:**
  ```json
  {
    "user_id": "string"
  }
  ```
- **Response:**
  - `200 OK`: `{"user_id": "user_id", "thread_id": "new_thread_id"}`

#### `GET /threads/{user_id}`

- **Description:** Get all conversation threads for a user.
- **Response:**
  - `200 OK`: `{"user_id": "user_id", "threads": ["thread1", "thread2", ...]}`

### Chat

#### `POST /chat`

- **Description:** Process a user message and get a response from the agent.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "thread_id": "string",
    "message": "string"
  }
  ```
- **Response:**
  - `200 OK`: `{"response": "Agent's response", "timestamp": "..."}`

### Mood & Wellness

#### `POST /mood/log`

- **Description:** Log the user's current mood and trigger gamification updates.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "mood": "string (e.g., 'Happy', 'Sad')",
    "intensity": "integer (1-10)",
    "notes": "string (optional)"
  }
  ```
- **Response:**
  - `200 OK`: `{"status": "success", "result": "Gamification update details"}`

#### `GET /mood/history/{user_id}`

- **Description:** Get the mood history for a user.
- **Query Parameters:**
  - `limit`: `integer (optional, default: 50)`
- **Response:**
  - `200 OK`: `{"user_id": "user_id", "mood_logs": [...]}`

### Crisis & Alerts

#### `POST /alerts/send`

- **Description:** Send an emergency alert.
- **Request Body:**
  ```json
  {
    "message": "string",
    "user_id": "string",
    "specific_contact": "string (optional, e.g., 'guardian')"
  }
  ```
- **Response:**
  - `200 OK`: `{"status": "success", "result": "Alert sending status"}`

#### `GET /crisis/history/{user_id}`

- **Description:** Get the crisis event history for a user.
- **Query Parameters:**
  - `limit`: `integer (optional, default: 10)`
- **Response:**
  - `200 OK`: `{"user_id": "user_id", "crisis_events": [...]}`

### Reminders & Scheduling

#### `POST /reminders/schedule`

- **Description:** Schedule a reminder for a user.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "message": "string",
    "reminder_time": "datetime (ISO 8601 format)",
    "repeat_interval": "string (optional)"
  }
  ```
- **Response:**
  - `200 OK`: `{"status": "success", "reminder_id": "new_reminder_id"}`

#### `POST /scheduler/wellness/{user_id}`

- **Description:** Schedule wellness tasks for a user.
- **Response:**
  - `200 OK`: `{"status": "success", "message": "Wellness tasks scheduled"}`

#### `GET /scheduler/status`

- **Description:** Get the status of the wellness scheduler.
- **Response:**
  - `200 OK`: `{"status": "success", "scheduler": "..."}`

### Live Sessions

#### `POST /live/start`

- **Description:** Start a live audio/video session.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "video_mode": "string (optional, e.g., 'camera')"
  }
  ```
- **Response:**
  - `200 OK`: `{"status": "success", "session_id": "new_session_id"}`

#### `POST /live/stop/{session_id}`

- **Description:** Stop an active live session.
- **Response:**
  - `200 OK`: `{"status": "success", "message": "Session stopped"}`

#### `GET /live/status`

- **Description:** Get information about all active live sessions.
- **Response:**
  - `200 OK`: `{"active_sessions": [...]}`

### Integrations

#### `GET /integrations/status`

- **Description:** Get the status of all external integrations.
- **Response:**
  - `200 OK`: `{"whatsapp": true/false, "telegram": true/false, "ehr": true/false}`

#### `POST /integrations/test`

- **Description:** Test all configured integrations.
- **Response:**
  - `200 OK`: `{"status": "success", "results": "..."}`
