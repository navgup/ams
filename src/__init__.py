"""
AMS - Agent Memory Scaffolding

A standalone chatbot agent that achieves State-of-the-Art performance on the 
LoCoMo benchmark by treating Reasoning Traces and Intermediate Work as 
structured, versioned "Artifacts" in a database.

Key Innovation:
Unlike standard RAG or A-MEM, this agent captures and structures:
- Reasoning patterns that can be reused
- Entity and event information with proper relationships
- Facts with conflict detection and consolidation
- Temporal information for time-aware reasoning

Components:
- schemas: Pydantic models for all artifact types
- storage: Versioned artifact store with semantic search
- retrieval: DSPy-based hybrid retrieval engine
- lifecycle: Observer for artifact extraction from traces
- agent: Main AMSAgent orchestration class
"""

from .schemas import (
    Artifact,
    EventArtifact,
    FactArtifact,
    ReasoningArtifact,
    SummaryArtifact,
    ConversationTurnArtifact,
    ValidityStatus,
    QueryIntent,
    ArtifactType,
)

from .storage import ArtifactStore

from .retrieval import (
    QueryRouter,
    ContextSelector,
    MultiHopPlanner,
    HybridRetrievalEngine,
    create_retrieval_engine,
)

from .lifecycle import (
    ReasoningExtractor,
    EventExtractor,
    FactExtractor,
    FactConsolidator,
    LifecycleManager,
)

from .agent import (
    AMSAgent,
    AMSAgentResponse,
    create_ams_agent,
    create_ollama_agent,
)

__version__ = "0.1.0"
__all__ = [
    # Schemas
    "Artifact",
    "EventArtifact",
    "FactArtifact",
    "ReasoningArtifact",
    "SummaryArtifact",
    "ConversationTurnArtifact",
    "ValidityStatus",
    "QueryIntent",
    "ArtifactType",
    # Storage
    "ArtifactStore",
    # Retrieval
    "QueryRouter",
    "ContextSelector",
    "MultiHopPlanner",
    "HybridRetrievalEngine",
    "create_retrieval_engine",
    # Lifecycle
    "ReasoningExtractor",
    "EventExtractor",
    "FactExtractor",
    "FactConsolidator",
    "LifecycleManager",
    # Agent
    "AMSAgent",
    "AMSAgentResponse",
    "create_ams_agent",
    "create_ollama_agent",
]

