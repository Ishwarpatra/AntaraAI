# Antara AI - A Memory-Driven Conversational AI

Antara AI is a sophisticated conversational AI system built on the LangGraph framework, featuring persistent memory, student-centric features, gamification, and advanced crisis detection. This project aims to provide a supportive and engaging conversational experience for students, helping them manage academic stress and peer pressure.

## Features

- **Memory-Driven Conversations:** The agent remembers past interactions and learns from them, providing a personalized experience.
- **Student-Centric Features:** Includes a RAG pipeline to provide information on study materials and academic coping strategies.
- **Gamification:** A "Snap Streak" and XP system to encourage user engagement.
- **Advanced Crisis Detection:** A multi-layered sentiment analysis system to detect and respond to user distress.
- **Modular and Extensible:** Built with a clean architecture that allows for easy extension and integration.

## Architecture Overview

The application is built with a modular, layered architecture:

- **UI Layer (`streamlit_app.py`, `cli_app.py`):** User interface components for interacting with the agent.
- **API Layer (`api.py`):** A FastAPI-based REST API that exposes the agent's functionality.
- **Service Layer (`core/service.py`):** A high-level interface that orchestrates the core logic.
- **Core Layer (`core/`):** Contains the core agent logic, including the LangGraph state machine, memory management, and tools.
- **Configuration Layer (`config/`):** Centralized configuration for the application, including prompt templates and system settings.

## Setup and Installation

### Prerequisites

- Python 3.10+
- MongoDB
- An API key for an embedding provider (e.g., OpenAI, Google)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ishwarpatra/AntaraAI.git
    cd AntaraAI
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project is missing a `requirements.txt` file. You will need to manually install the following dependencies:
    ```bash
    pip install fastapi uvicorn pymongo langchain langgraph langchain-openai langchain-google-genai langchain-community sentence-transformers python-dotenv bandit tenacity pytest
    ```
    *Note: This list is based on the issues encountered during development. It might not be exhaustive.*

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following:
    ```
    # MongoDB
    MONGODB_URI="mongodb://localhost:27017/"
    DATABASE_NAME="antara_ai"

    # Embedding Provider (choose one)
    EMBEDDING_PROVIDER="openai" # or "google" or "huggingface"
    OPENAI_API_KEY="your-openai-api-key"
    # GOOGLE_API_KEY="your-google-api-key"

    # Crisis Contact (for crisis_escalation_tool)
    EMERGENCY_CONTACT_PHONE="+1234567890" # Example phone number for Twilio
    ```

5.  **Ingest academic resources:**
    Run the ingestion script to populate the academic resources knowledge base:
    ```bash
    python utils/ingestion.py
    ```

### Running the Application

- **API Server:**
    ```bash
    uvicorn api:app --host 127.0.0.1 --port 8000 --reload
    ```

- **Streamlit UI:**
    ```bash
    streamlit run streamlit_app.py
    ```

- **CLI App:**
    ```bash
    python cli_app.py
    ```

## API Reference

The API is documented using OpenAPI (Swagger). Once the API server is running, you can access the interactive documentation at `http://127.0.0.1:8000/docs`.

### Key Endpoints

- **`POST /chat`:** Process a user message and get a response.
- **`POST /mood/log`:** Log the user's mood and get a gamification update.
- **`GET /users`:** Get a list of users.
- **`POST /users`:** Create a new user.

## Usage Examples

### CLI

```bash
python cli_app.py
```

### API (using `curl`)

- **Create a user:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/users" -H "Content-Type: application/json" -d '{"user_id": "test_user"}'
  ```

- **Send a message:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "test_user", "thread_id": "some_thread", "message": "Hello, how are you?"}'
  ```

## Contribution Guidelines

Please see `CONTRIBUTING.md` for details on how to contribute to this project.