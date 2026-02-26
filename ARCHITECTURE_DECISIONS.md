# Architecture Decision Records (ADR)

This document serves as a log of significant architectural decisions made during the development of the Antara AI system. Each record captures the context, the decision made, the alternatives considered, and the consequences.

---

## ADR 001: Choosing LangGraph for Agent Orchestration

**Date:** 2026-02-26

**Context:**
The Antara AI system requires a robust framework for orchestrating complex agent behaviors, managing conversational state, and enabling tool use. The agent needs to be able to make decisions, execute actions, and maintain a persistent memory across turns.

**Decision:**
LangGraph was chosen as the primary framework for agent orchestration.

**Alternatives Considered:**
*   **Vanilla LangChain:** While powerful for individual LLM calls and tool integration, it lacks native support for stateful, cyclical agentic workflows and checkpointing, requiring more manual orchestration.
*   **Custom State Machine:** Building a custom state machine would offer maximum flexibility but would be time-consuming to develop and maintain, lacking the battle-tested features and community support of an established library.

**Consequences:**
*   **Positive:**
    *   Provides a clear, declarative way to define agentic workflows, including conditional routing and loops.
    *   Native support for checkpointing (via `MongoDBSaver`) simplifies persistent memory management and enables features like "time travel" and recovery.
    *   Seamless integration with LangChain tools and models.
    *   Promotes modularity and testability of individual agent nodes.
    *   Facilitates debugging of complex agent behaviors.
*   **Negative:**
    *   Introduces an additional layer of abstraction and learning curve.
    *   Requires careful design of state representation and graph edges to avoid complex debugging.
    *   May introduce performance overhead compared to simpler, linear chains for basic interactions.

---

## ADR 002: Implementing MongoDB for Persistent Memory and Vector Store

**Date:** 2026-02-26

**Context:**
The Antara AI requires a persistent memory store for user conversations, gamification data, crisis events, and academic resources. This store needs to support both structured data (for fast retrieval of user profiles, mood logs) and vector-based semantic search (for RAG over academic materials and memory recall).

**Decision:**
MongoDB was chosen as the primary database for persistent memory and vector storage.

**Alternatives Considered:**
*   **PostgreSQL with pgvector:** A strong contender, offering robust relational capabilities and efficient vector search. However, MongoDB's flexible schema was deemed more advantageous for the evolving nature of agent memories and disparate data types (e.g., mood logs, crisis events, user profiles, RAG documents).
*   **Dedicated Vector Database (e.g., Pinecone, Weaviate):** Excellent for vector search performance, but would introduce an additional dependency and architectural complexity for managing both structured and unstructured data across two different databases. MongoDB Atlas Vector Search provides a unified solution.
*   **SQLite:** Too limited for a scalable, multi-user application with complex data types and vector search requirements.

**Consequences:**
*   **Positive:**
    *   Unified storage for all data types (conversational history, user profiles, gamification, RAG documents).
    *   Flexible schema (document-based) accommodates the evolving nature of agent memories without rigid migrations.
    *   MongoDB Atlas Vector Search provides integrated, scalable semantic search capabilities.
    *   Strong community support and managed services (MongoDB Atlas).
    *   Facilitates easy debugging and inspection of agent state and memories.
*   **Negative:**
    *   Requires careful indexing and query optimization to maintain performance for large datasets.
    *   Managed service costs (e.g., MongoDB Atlas) can accumulate.
    *   Potential for schema ambiguity if not managed carefully within the application logic.

---

## ADR 003: Rule-Based Sentiment Analysis for Crisis Detection

**Date:** 2026-02-26

**Context:**
Early detection of crisis situations (e.g., self-harm, severe distress) is paramount for the Antara AI system. The initial approach involved a highly complex rule-based system with regex and keyword counting. The feedback indicated this was brittle and excessively complex.

**Decision:**
Refactored the rule-based sentiment analysis into a structured, configurable classification dictionary approach using `SystemConfig`.

**Alternatives Considered:**
*   **Dedicated Sentiment Analysis Model (e.g., HuggingFace Transformers):** Would offer higher accuracy and generalization but introduce a large model dependency, increased inference latency, and computational cost. For immediate crisis detection, a fast, rule-based approach can be more appropriate as a first line of defense.
*   **External Sentiment Analysis API:** Simplifies integration but introduces latency, cost, and reliance on external service availability.
*   **Keep Original Complex Regex:** Rejected due to technical debt, maintainability issues, and excessive cyclomatic complexity.

**Consequences:**
*   **Positive:**
    *   Improved maintainability and readability of the sentiment analysis logic.
    *   Centralized configuration in `SystemConfig` makes thresholds and keywords easily adjustable.
    *   Fast and efficient for real-time crisis detection.
    *   Provides a clear, auditable logic for severity classification.
*   **Negative:**
    *   Still limited by the scope of defined rules and keywords (i.e., less generalizable than ML models).
    *   Requires manual tuning and updates as new crisis indicators emerge.
    *   False positives/negatives can occur if rules are not carefully crafted.

---

## ADR 004: Gamification System for User Engagement

**Date:** 2026-02-26

**Context:**
The project aims to enhance student engagement and provide positive reinforcement through gamification features like mood logging streaks and experience points (XP). This functionality needs to be integrated with the agent's tools and persisted in the database.

**Decision:**
Implemented a dedicated `GamificationManager` class to encapsulate gamification logic and integrate it with mood logging tools.

**Alternatives Considered:**
*   **Embedding Gamification Logic Directly in Tools:** Would lead to duplicated code and tightly coupled concerns within individual tools, making maintenance difficult.
*   **External Gamification Service:** Overkill for the initial scope, introducing unnecessary complexity and latency for simple streak/XP tracking.

**Consequences:**
*   **Positive:**
    *   Centralized gamification logic in `GamificationManager` promotes reusability and maintainability.
    *   Clear separation of concerns for updating user progress and calculating rewards.
    *   Enables verbal reinforcement by the AI agent through tool returns.
    *   Persistent storage in MongoDB ensures user progress is saved across sessions.
*   **Negative:**
    *   Adds a new module and class, increasing the codebase size slightly.
    *   Requires careful design of gamification mechanics to ensure fairness and effectiveness.

---

## ADR 005: Exponential Backoff and Retries for External API Calls

**Date:** 2026-02-26

**Context:**
External API calls (e.g., to WhatsApp, Telegram, EHR) are prone to transient network failures, rate limiting, and temporary service unavailability. Without robust error handling, these failures can lead to crashes or inconsistent behavior, impacting the reliability of critical features like crisis escalation.

**Decision:**
Implemented exponential backoff and retry mechanisms using the `tenacity` library for all network-dependent tools and integrations.

**Alternatives Considered:**
*   **Manual Retry Logic:** Implementing custom retry loops with delays would be repetitive, error-prone, and difficult to standardize across multiple integrations.
*   **Ignoring Transient Failures:** Would lead to a brittle system that is highly susceptible to external service disruptions, especially critical for features like crisis alerts.

**Consequences:**
*   **Positive:**
    *   Significantly improves the reliability and resilience of external API calls.
    *   Reduces the likelihood of cascading failures due to transient network issues.
    *   Standardizes error handling for external integrations.
    *   Enhances user experience by making the system more robust.
*   **Negative:**
    *   Introduces `tenacity` as a new dependency.
    *   Retries can increase the perceived latency for operations that experience delays.
    *   Requires careful configuration of retry parameters (e.g., `stop_after_attempt`, `wait_exponential`) to avoid excessive retries.

---

## ADR 006: User-Specific Emergency Contacts

**Date:** 2026-02-26

**Context:**
The `crisis_escalation_tool` initially relied on a single, hardcoded environment variable for an emergency contact. This lacked the flexibility for user-specific configurations and proper privacy/security separation for different users' emergency contacts.

**Decision:**
Implemented a mechanism to store and retrieve user-specific emergency contacts in MongoDB, associated with individual `user_id`s. The `crisis_escalation_tool` now fetches and dispatches alerts to these personalized contacts.

**Alternatives Considered:**
*   **Keeping Hardcoded Contacts:** Rejected due to lack of personalization, privacy concerns, and inability to scale for multiple users.
*   **External Contact Management Service:** Overkill for the current scope and would introduce additional integration complexity for what is essentially user profile data.

**Consequences:**
*   **Positive:**
    *   Enables personalized and secure emergency contact management for each user.
    *   Improves the privacy and security posture by allowing users to define their own sensitive contacts.
    *   Scalable solution for managing emergency contacts across a growing user base.
    *   Centralized management of contact details in MongoDB.
*   **Negative:**
    *   Adds a new data model (`user_profiles` collection) and associated CRUD operations in `core/memory_manager.py`.
    *   Requires secure handling of contact data in the database (necessitating future field-level encryption).

---
