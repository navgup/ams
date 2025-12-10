"""
AMS Schemas - Schema-First Artifact Memory System

Defines structured Pydantic models for all artifact types with built-in
versioning support. Unlike A-MEM's generic JSON notes, each artifact type
has strict schema enforcement for improved retrieval and reasoning.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, ClassVar, TYPE_CHECKING
from uuid import uuid4
import json

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from storage import ArtifactStore


# ============================================================================
# Enums
# ============================================================================

class ValidityStatus(str, Enum):
    """Validity status for facts."""
    VERIFIED = "verified"
    CONFLICTING = "conflicting"
    DEPRECATED = "deprecated"


class QueryIntent(str, Enum):
    """Intent classification for queries."""
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    REASONING = "reasoning"
    MULTI_HOP = "multi_hop"


class ArtifactType(str, Enum):
    """Enumeration of all artifact types for filtering."""
    EVENT = "EventArtifact"
    FACT = "FactArtifact"
    REASONING = "ReasoningArtifact"
    SUMMARY = "SummaryArtifact"


# ============================================================================
# Base Artifact Class
# ============================================================================

class Artifact(BaseModel):
    """
    Base class for all artifacts in the AMS system.
    
    Every artifact has:
    - Unique ID (UUID)
    - Creation timestamp
    - Provenance ID linking to source message/artifact
    - Version chain support for non-destructive updates
    """
    
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_id: Optional[str] = Field(
        default=None,
        description="ID of the source message or artifact that created this"
    )
    
    # Version chain fields
    previous_version_id: Optional[str] = Field(
        default=None,
        description="ID of the previous version of this artifact"
    )
    latest_version_id: Optional[str] = Field(
        default=None,
        description="Pointer to the most recent version (null if this is latest)"
    )
    is_current: bool = Field(
        default=True,
        description="Whether this is the current/active version"
    )
    
    # Embedding support
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic search"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @computed_field
    @property
    def artifact_type(self) -> str:
        """Return the artifact type name for filtering."""
        return self.__class__.__name__
    
    def get_embedding_text(self) -> str:
        """
        Return text representation for embedding generation.
        Subclasses should override this to provide meaningful text.
        """
        return str(self.model_dump(exclude={'embedding', 'id', 'previous_version_id', 'latest_version_id'}))
    
    def get_summary(self) -> str:
        """
        Return a brief summary for candidate selection.
        Subclasses should override this.
        """
        return f"{self.artifact_type}: {self.id[:8]}..."
    
    def save(self, store: "ArtifactStore") -> "Artifact":
        """
        Save this artifact with proper versioning.
        
        If this artifact updates an existing one (same entity/fact), this method:
        1. Creates a new row with a new UUID
        2. Sets previous_version_id to the old UUID
        3. Updates the latest_version_id pointer on all versions
        4. Marks old version as is_current=False
        
        Args:
            store: The artifact store to save to
            
        Returns:
            The saved artifact with updated IDs
        """
        return store.save_artifact(self)
    
    def to_context_string(self) -> str:
        """Convert artifact to a string suitable for LLM context."""
        return self.get_summary()


# ============================================================================
# Event Artifact
# ============================================================================

class EventArtifact(Artifact):
    """
    Represents an event bound to a specific time.
    
    CRUCIAL for temporal reasoning in LoCoMo benchmark.
    The timestamp field enables structured filtering like:
    {"type": "EventArtifact", "timestamp": {"$gt": "2022"}}
    """
    
    description: str = Field(
        ...,
        description="Description of the event"
    )
    timestamp: datetime = Field(
        ...,
        description="When the event occurred (crucial for temporal queries)"
    )
    involved_entity_ids: List[str] = Field(
        default_factory=list,
        description="IDs of entities involved in this event"
    )
    entity_tags: List[str] = Field(
        default_factory=list,
        description="Entities this event involves (include speaker and referenced entities)"
    )
    natural_date_text: Optional[str] = Field(
        default=None,
        description="Original/natural text describing the date (e.g., '2 weeks before June 1')"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration of the event (e.g., '2 hours', '3 days')"
    )
    location: Optional[str] = Field(
        default=None,
        description="Where the event occurred"
    )
    event_type: Optional[str] = Field(
        default=None,
        description="Category of event (meeting, travel, milestone, etc.)"
    )
    
    def get_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            self.description,
            f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ]
        if self.natural_date_text:
            parts.append(f"Natural date: {self.natural_date_text}")
        if self.entity_tags:
            parts.append(f"Entities: {', '.join(self.entity_tags)}")
        if self.duration:
            parts.append(f"Duration: {self.duration}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.event_type:
            parts.append(f"Type: {self.event_type}")
        return " | ".join(parts)
    
    def get_summary(self) -> str:
        """Return brief summary with timestamp."""
        date_str = self.timestamp.strftime('%Y-%m-%d')
        return f"[Event {date_str}] {self.description[:80]}..."
    
    def to_context_string(self) -> str:
        """Convert to LLM context string."""
        lines = [f"Event: {self.description}"]
        lines.append(f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}")
        if self.natural_date_text:
            lines.append(f"Natural date: {self.natural_date_text}")
        if self.entity_tags:
            lines.append(f"Entities: {', '.join(self.entity_tags)}")
        if self.duration:
            lines.append(f"Duration: {self.duration}")
        if self.location:
            lines.append(f"Location: {self.location}")
        return "\n".join(lines)


# ============================================================================
# Fact Artifact
# ============================================================================

class FactArtifact(Artifact):
    """
    Represents atomic units of information verified from a source.
    
    Facts can be verified, conflicting (with other facts), or deprecated.
    This enables the system to handle contradictions and updates gracefully.
    """
    
    claim: str = Field(
        ...,
        description="The factual claim or statement"
    )
    entity_tags: List[str] = Field(
        default_factory=list,
        description="Entities this fact applies to (include speaker and referenced entities)"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Source URL if from external reference"
    )
    validity_status: ValidityStatus = Field(
        default=ValidityStatus.VERIFIED,
        description="Current validity status of this fact"
    )
    topic_tags: List[str] = Field(
        default_factory=list,
        description="Topic tags for categorization and retrieval"
    )
    conflicting_fact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of facts that conflict with this one"
    )
    supporting_fact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of facts that support this one"
    )
    
    def get_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [self.claim]
        if self.topic_tags:
            parts.append(f"Topics: {', '.join(self.topic_tags)}")
        if self.entity_tags:
            parts.append(f"Entities: {', '.join(self.entity_tags)}")
        parts.append(f"Status: {self.validity_status.value}")
        return " | ".join(parts)
    
    def get_summary(self) -> str:
        """Return brief summary with status."""
        return f"[Fact {self.validity_status.value}] {self.claim[:80]}..."
    
    def to_context_string(self) -> str:
        """Convert to LLM context string."""
        lines = [f"Fact: {self.claim}"]
        lines.append(f"Status: {self.validity_status.value}")
        if self.topic_tags:
            lines.append(f"Topics: {', '.join(self.topic_tags)}")
        if self.entity_tags:
            lines.append(f"Entities: {', '.join(self.entity_tags)}")
        return "\n".join(lines)


# ============================================================================
# Reasoning Artifact (Meta-Cognitive Layer)
# ============================================================================

class ReasoningArtifact(Artifact):
    """
    Represents a successful logic pattern or strategy used to solve a problem.
    
    This is the "meta-cognitive" layer that captures reusable reasoning patterns.
    Unlike A-MEM which only stores raw notes, this enables the agent to learn
    from past problem-solving approaches.
    
    Critical for Multi-hop LoCoMo tasks:
    - Captures "bridge" strategies (e.g., "To find X's wife's job, first search for X's wife")
    - Stores abstract step-by-step logic that can be applied to similar queries
    """
    
    goal_category: str = Field(
        ...,
        description="Category of goal (e.g., 'multi-hop-retrieval', 'temporal-reasoning', 'debugging')"
    )
    trigger_pattern: str = Field(
        ...,
        description="Pattern/signature that should trigger this reasoning strategy"
    )
    abstract_strategy: List[str] = Field(
        ...,
        description="Step-by-step abstract logic that can be reused"
    )
    outcome_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating of how successful this strategy was (1-5)"
    )
    original_query: Optional[str] = Field(
        default=None,
        description="The original query this strategy solved"
    )
    example_execution: Optional[str] = Field(
        default=None,
        description="Example of how this strategy was executed"
    )
    applicable_domains: List[str] = Field(
        default_factory=list,
        description="Domains where this strategy applies"
    )
    prerequisite_info: List[str] = Field(
        default_factory=list,
        description="Information needed before applying this strategy"
    )
    
    def get_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Goal: {self.goal_category}",
            f"Trigger: {self.trigger_pattern}",
            f"Strategy: {' -> '.join(self.abstract_strategy)}"
        ]
        if self.applicable_domains:
            parts.append(f"Domains: {', '.join(self.applicable_domains)}")
        return " | ".join(parts)
    
    def get_summary(self) -> str:
        """Return brief summary."""
        stars = "★" * self.outcome_rating + "☆" * (5 - self.outcome_rating)
        return f"[Reasoning {stars}] {self.goal_category}: {self.trigger_pattern[:50]}..."
    
    def to_context_string(self) -> str:
        """Convert to LLM context string for applying this strategy."""
        lines = [f"Reasoning Strategy: {self.goal_category}"]
        lines.append(f"Use when: {self.trigger_pattern}")
        lines.append("Steps:")
        for i, step in enumerate(self.abstract_strategy, 1):
            lines.append(f"  {i}. {step}")
        if self.prerequisite_info:
            lines.append(f"Prerequisites: {', '.join(self.prerequisite_info)}")
        return "\n".join(lines)


# ============================================================================
# Summary Artifact (For Consolidated Facts/Entities)
# ============================================================================

class SummaryArtifact(Artifact):
    """
    Represents a consolidated summary of multiple related facts or entities.
    
    Created by the FactConsolidator when multiple facts agree on a topic.
    """
    
    title: str = Field(
        ...,
        description="Title of the summary"
    )
    content: str = Field(
        ...,
        description="The consolidated summary content"
    )
    source_artifact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of artifacts that were consolidated into this summary"
    )
    topic_tags: List[str] = Field(
        default_factory=list,
        description="Topic tags for categorization"
    )
    consolidated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this summary was created"
    )
    
    def get_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [self.title, self.content]
        if self.topic_tags:
            parts.append(f"Topics: {', '.join(self.topic_tags)}")
        return " | ".join(parts)
    
    def get_summary(self) -> str:
        """Return brief summary."""
        return f"[Summary] {self.title}: {self.content[:80]}..."
    
    def to_context_string(self) -> str:
        """Convert to LLM context string."""
        lines = [f"Summary: {self.title}"]
        lines.append(self.content)
        if self.topic_tags:
            lines.append(f"Topics: {', '.join(self.topic_tags)}")
        return "\n".join(lines)


# ============================================================================
# Conversation Turn Artifact (Raw Input Preservation)
# ============================================================================

class ConversationTurnArtifact(Artifact):
    """
    Represents a single turn in the conversation.
    
    Preserves the raw input for reference while other artifacts
    extract structured information from it.
    """
    
    speaker: str = Field(
        ...,
        description="Who spoke this turn"
    )
    content: str = Field(
        ...,
        description="The raw content of the turn"
    )
    turn_timestamp: Optional[datetime] = Field(
        default=None,
        description="When this turn occurred in the conversation"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID of the conversation session"
    )
    extracted_artifact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of artifacts extracted from this turn"
    )
    
    def get_embedding_text(self) -> str:
        """Generate text for embedding."""
        return f"{self.speaker}: {self.content}"
    
    def get_summary(self) -> str:
        """Return brief summary."""
        return f"[Turn] {self.speaker}: {self.content[:80]}..."
    
    def to_context_string(self) -> str:
        """Convert to LLM context string."""
        timestamp_str = ""
        if self.turn_timestamp:
            timestamp_str = f" [{self.turn_timestamp.strftime('%Y-%m-%d %H:%M')}]"
        return f"{self.speaker}{timestamp_str}: {self.content}"


# ============================================================================
# Type Registry for Serialization
# ============================================================================

ARTIFACT_TYPES: Dict[str, type] = {
    "EventArtifact": EventArtifact,
    "FactArtifact": FactArtifact,
    "ReasoningArtifact": ReasoningArtifact,
    "SummaryArtifact": SummaryArtifact,
    "ConversationTurnArtifact": ConversationTurnArtifact,
}


def deserialize_artifact(data: Dict[str, Any]) -> Artifact:
    """
    Deserialize a dictionary to the appropriate Artifact subclass.
    
    Args:
        data: Dictionary with artifact data, must include 'artifact_type'
        
    Returns:
        The appropriate Artifact subclass instance
    """
    artifact_type = data.get("artifact_type")
    if not artifact_type:
        raise ValueError("Missing 'artifact_type' in artifact data")
    
    artifact_class = ARTIFACT_TYPES.get(artifact_type)
    if not artifact_class:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    return artifact_class(**{k: v for k, v in data.items() if k != "artifact_type"})


# ============================================================================
# Query Filter Schemas (For QueryRouter)
# ============================================================================

class TemporalFilter(BaseModel):
    """Temporal filter for date-based queries."""
    field: str = Field(default="timestamp", description="Field to filter on")
    operator: str = Field(..., description="Comparison operator ($gt, $lt, $gte, $lte, $eq)")
    value: str = Field(..., description="ISO date string to compare against")


class StructuredFilter(BaseModel):
    """Structured filter output from QueryRouter."""
    artifact_type: Optional[ArtifactType] = Field(
        default=None,
        description="Filter by artifact type"
    )
    temporal_filter: Optional[TemporalFilter] = Field(
        default=None,
        description="Temporal constraint"
    )
    topic_tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by topic tags"
    )
    validity_status: Optional[ValidityStatus] = Field(
        default=None,
        description="Filter facts by validity status"
    )
    
    def to_mongo_filter(self) -> Dict[str, Any]:
        """Convert to MongoDB-style filter dict."""
        filters = {}
        
        if self.artifact_type:
            filters["artifact_type"] = self.artifact_type.value
        
        if self.temporal_filter:
            op_map = {
                "$gt": "$gt",
                "$lt": "$lt", 
                "$gte": "$gte",
                "$lte": "$lte",
                "$eq": "$eq"
            }
            filters[self.temporal_filter.field] = {
                op_map[self.temporal_filter.operator]: self.temporal_filter.value
            }
        
        if self.topic_tags:
            filters["topic_tags"] = {"$in": self.topic_tags}
        
        if self.validity_status:
            filters["validity_status"] = self.validity_status.value
        
        return filters

