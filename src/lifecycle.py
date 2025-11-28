"""
AMS Lifecycle Layer - The "Observer" for Artifact Extraction

Runs after the agent generates a response to:
1. Extract structured artifacts from the thinking trace
2. Consolidate facts and identify conflicts
3. Create reusable reasoning patterns

This is what makes AMS superior to A-MEM - it captures and structures
the intermediate work that would otherwise be lost.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

import dspy
from pydantic import BaseModel, Field

from schemas import (
    Artifact,
    EntityArtifact,
    EventArtifact,
    FactArtifact,
    ReasoningArtifact,
    SummaryArtifact,
    ConversationTurnArtifact,
    ValidityStatus,
    QueryIntent,
)
from storage import ArtifactStore


# ============================================================================
# DSPy Signatures for Extraction
# ============================================================================

class ReasoningExtractorSignature(dspy.Signature):
    """
    Extract reusable reasoning patterns from a model's thinking trace.
    
    Analyzes the CoT output to identify:
    1. Was a reusable strategy used? (e.g., "Check X, then Y")
    2. What was the abstract pattern that could apply to similar queries?
    3. How successful was this approach?
    """
    
    user_query: str = dspy.InputField(
        desc="The original user question"
    )
    raw_model_trace: str = dspy.InputField(
        desc="The Chain-of-Thought reasoning trace from the model"
    )
    final_answer: str = dspy.InputField(
        desc="The final answer generated"
    )
    
    has_reusable_strategy: str = dspy.OutputField(
        desc="'yes' if the trace contains a reusable strategy, 'no' otherwise"
    )
    goal_category: str = dspy.OutputField(
        desc="Category: multi-hop-retrieval, temporal-reasoning, fact-verification, entity-lookup, or general"
    )
    trigger_pattern: str = dspy.OutputField(
        desc="Pattern that should trigger this strategy (e.g., 'questions about X's relationship to Y')"
    )
    abstract_strategy: str = dspy.OutputField(
        desc="JSON list of abstract steps: ['Step 1: ...', 'Step 2: ...']"
    )
    outcome_rating: str = dspy.OutputField(
        desc="Rating 1-5 based on how well the strategy worked"
    )


class EntityExtractorSignature(dspy.Signature):
    """Extract entities mentioned in text."""
    
    text: str = dspy.InputField(desc="Text to extract entities from")
    context: str = dspy.InputField(desc="Conversation context for disambiguation")
    
    entities: str = dspy.OutputField(
        desc="JSON list of entities: [{name, type, summary, aliases}]"
    )


class EventExtractorSignature(dspy.Signature):
    """Extract events with timestamps from text."""
    
    text: str = dspy.InputField(desc="Text to extract events from")
    reference_date: str = dspy.InputField(desc="Reference date for relative time parsing (ISO format)")
    
    events: str = dspy.OutputField(
        desc="JSON list of events: [{description, timestamp (ISO), duration, location}]"
    )


class FactExtractorSignature(dspy.Signature):
    """Extract atomic facts from text."""
    
    text: str = dspy.InputField(desc="Text to extract facts from")
    source_context: str = dspy.InputField(desc="Context about the source of this information")
    
    facts: str = dspy.OutputField(
        desc="JSON list of facts: [{claim, topic_tags, confidence}]"
    )


class FactConflictCheckerSignature(dspy.Signature):
    """Check if two facts conflict with each other."""
    
    fact1: str = dspy.InputField(desc="First fact claim")
    fact2: str = dspy.InputField(desc="Second fact claim")
    context: str = dspy.InputField(desc="Additional context for judgment")
    
    relationship: str = dspy.OutputField(
        desc="Relationship: 'agree', 'conflict', 'unrelated', or 'supersedes'"
    )
    explanation: str = dspy.OutputField(
        desc="Brief explanation of the relationship"
    )


class FactConsolidatorSignature(dspy.Signature):
    """Consolidate multiple agreeing facts into a summary."""
    
    facts: str = dspy.InputField(desc="JSON list of fact claims to consolidate")
    topic: str = dspy.InputField(desc="The common topic of these facts")
    
    consolidated_summary: str = dspy.OutputField(
        desc="A single consolidated summary of all the facts"
    )
    title: str = dspy.OutputField(
        desc="A brief title for the consolidated summary"
    )


# ============================================================================
# DSPy Modules
# ============================================================================

class ReasoningExtractor(dspy.Module):
    """
    Extracts reusable reasoning patterns from model thinking traces.
    
    This is the core of the "meta-cognitive" layer - learning from
    successful problem-solving to improve future performance.
    """
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ReasoningExtractorSignature)
    
    def forward(
        self,
        user_query: str,
        raw_model_trace: str,
        final_answer: str
    ) -> Optional[ReasoningArtifact]:
        """
        Extract a reasoning artifact from a trace.
        
        Args:
            user_query: The original question
            raw_model_trace: The CoT thinking trace
            final_answer: The final answer produced
            
        Returns:
            ReasoningArtifact if a reusable pattern was found, None otherwise
        """
        if not raw_model_trace or len(raw_model_trace.strip()) < 50:
            return None
        
        result = self.extractor(
            user_query=user_query,
            raw_model_trace=raw_model_trace,
            final_answer=final_answer
        )
        
        # Check if a reusable strategy was found
        if result.has_reusable_strategy.lower().strip() != "yes":
            return None
        
        # Parse the abstract strategy
        try:
            abstract_strategy = json.loads(result.abstract_strategy)
            if not isinstance(abstract_strategy, list):
                abstract_strategy = [result.abstract_strategy]
        except json.JSONDecodeError:
            # Try to parse as newline-separated steps
            abstract_strategy = [
                s.strip() for s in result.abstract_strategy.split("\n")
                if s.strip()
            ]
        
        # Parse rating
        try:
            rating = int(result.outcome_rating)
            rating = max(1, min(5, rating))
        except ValueError:
            rating = 3
        
        return ReasoningArtifact(
            goal_category=result.goal_category,
            trigger_pattern=result.trigger_pattern,
            abstract_strategy=abstract_strategy,
            outcome_rating=rating,
            original_query=user_query,
            example_execution=raw_model_trace[:500]  # Store truncated example
        )


class EntityExtractor(dspy.Module):
    """Extracts entity artifacts from text."""
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractorSignature)
    
    def forward(
        self,
        text: str,
        context: str = ""
    ) -> List[EntityArtifact]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract from
            context: Additional context
            
        Returns:
            List of EntityArtifacts
        """
        result = self.extractor(
            text=text,
            context=context or "No additional context"
        )
        
        try:
            entities_data = json.loads(result.entities)
        except json.JSONDecodeError:
            return []
        
        artifacts = []
        for entity in entities_data:
            if not isinstance(entity, dict):
                continue
            
            name = entity.get("name", "").strip()
            if not name:
                continue
            
            artifacts.append(EntityArtifact(
                name=name,
                summary=entity.get("summary", f"Entity: {name}"),
                aliases=entity.get("aliases", []),
                entity_type=entity.get("type"),
            ))
        
        return artifacts


class EventExtractor(dspy.Module):
    """Extracts event artifacts with timestamps from text."""
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EventExtractorSignature)
    
    def forward(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> List[EventArtifact]:
        """
        Extract events from text.
        
        Args:
            text: Text to extract from
            reference_date: Reference date for relative time parsing
            
        Returns:
            List of EventArtifacts
        """
        ref_date = reference_date or datetime.utcnow()
        
        result = self.extractor(
            text=text,
            reference_date=ref_date.isoformat()
        )
        
        try:
            events_data = json.loads(result.events)
        except json.JSONDecodeError:
            return []
        
        artifacts = []
        for event in events_data:
            if not isinstance(event, dict):
                continue
            
            description = event.get("description", "").strip()
            timestamp_str = event.get("timestamp", "").strip()
            
            if not description or not timestamp_str:
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    timestamp = ref_date  # Fallback to reference date
            
            artifacts.append(EventArtifact(
                description=description,
                timestamp=timestamp,
                duration=event.get("duration"),
                location=event.get("location"),
            ))
        
        return artifacts


class FactExtractor(dspy.Module):
    """Extracts fact artifacts from text."""
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(FactExtractorSignature)
    
    def forward(
        self,
        text: str,
        source_context: str = ""
    ) -> List[FactArtifact]:
        """
        Extract facts from text.
        
        Args:
            text: Text to extract from
            source_context: Context about the source
            
        Returns:
            List of FactArtifacts
        """
        result = self.extractor(
            text=text,
            source_context=source_context or "Extracted from conversation"
        )
        
        try:
            facts_data = json.loads(result.facts)
        except json.JSONDecodeError:
            return []
        
        artifacts = []
        for fact in facts_data:
            if not isinstance(fact, dict):
                continue
            
            claim = fact.get("claim", "").strip()
            if not claim:
                continue
            
            confidence = fact.get("confidence", 1.0)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 1.0
            
            artifacts.append(FactArtifact(
                claim=claim,
                topic_tags=fact.get("topic_tags", []),
                confidence_score=confidence,
                source_context=source_context,
            ))
        
        return artifacts


class FactConsolidator(dspy.Module):
    """
    Consolidates similar facts and detects conflicts.
    
    This runs as a background process to:
    1. Find facts with similar embeddings
    2. Merge agreeing facts into summaries
    3. Flag conflicting facts
    """
    
    def __init__(self, store: ArtifactStore):
        super().__init__()
        self.store = store
        self.conflict_checker = dspy.ChainOfThought(FactConflictCheckerSignature)
        self.consolidator = dspy.ChainOfThought(FactConsolidatorSignature)
    
    def check_conflict(
        self,
        fact1: FactArtifact,
        fact2: FactArtifact,
        context: str = ""
    ) -> Tuple[str, str]:
        """
        Check if two facts conflict.
        
        Returns:
            Tuple of (relationship, explanation)
            relationship is one of: 'agree', 'conflict', 'unrelated', 'supersedes'
        """
        result = self.conflict_checker(
            fact1=fact1.claim,
            fact2=fact2.claim,
            context=context or "No additional context"
        )
        
        relationship = result.relationship.lower().strip()
        if relationship not in ['agree', 'conflict', 'unrelated', 'supersedes']:
            relationship = 'unrelated'
        
        return relationship, result.explanation
    
    def consolidate_facts(
        self,
        facts: List[FactArtifact],
        topic: str
    ) -> Optional[SummaryArtifact]:
        """
        Consolidate agreeing facts into a summary.
        
        Args:
            facts: List of facts that agree on the topic
            topic: The common topic
            
        Returns:
            SummaryArtifact if consolidation succeeded
        """
        if len(facts) < 2:
            return None
        
        facts_json = json.dumps([f.claim for f in facts])
        
        result = self.consolidator(
            facts=facts_json,
            topic=topic
        )
        
        return SummaryArtifact(
            title=result.title,
            content=result.consolidated_summary,
            source_artifact_ids=[f.id for f in facts],
            topic_tags=list(set(tag for f in facts for tag in f.topic_tags)),
        )
    
    def process_new_fact(
        self,
        new_fact: FactArtifact,
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Process a new fact: check for conflicts and consolidation opportunities.
        
        Args:
            new_fact: The new fact to process
            similarity_threshold: Threshold for finding similar facts
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "conflicts": [],
            "agreements": [],
            "consolidated": None,
        }
        
        # Find similar facts
        similar = self.store.find_similar_facts(new_fact, threshold=similarity_threshold)
        
        agreeing_facts = [new_fact]
        
        for existing_fact, similarity in similar:
            relationship, explanation = self.check_conflict(
                new_fact, existing_fact,
                context=f"Similarity score: {similarity:.2f}"
            )
            
            if relationship == 'conflict':
                results["conflicts"].append({
                    "fact_id": existing_fact.id,
                    "claim": existing_fact.claim,
                    "explanation": explanation,
                })
                
                # Mark both as conflicting
                new_fact.validity_status = ValidityStatus.CONFLICTING
                new_fact.conflicting_fact_ids.append(existing_fact.id)
                
                existing_fact.validity_status = ValidityStatus.CONFLICTING
                existing_fact.conflicting_fact_ids.append(new_fact.id)
                self.store.save_artifact(existing_fact, update_of=existing_fact.id)
                
            elif relationship == 'agree':
                results["agreements"].append({
                    "fact_id": existing_fact.id,
                    "claim": existing_fact.claim,
                })
                agreeing_facts.append(existing_fact)
                
            elif relationship == 'supersedes':
                # New fact supersedes old one
                existing_fact.validity_status = ValidityStatus.DEPRECATED
                self.store.save_artifact(existing_fact, update_of=existing_fact.id)
        
        # Try to consolidate agreeing facts
        if len(agreeing_facts) >= 3:
            # Get common topic from tags
            common_tags = set.intersection(
                *[set(f.topic_tags) for f in agreeing_facts if f.topic_tags]
            ) if all(f.topic_tags for f in agreeing_facts) else set()
            
            topic = ", ".join(common_tags) if common_tags else "General"
            
            summary = self.consolidate_facts(agreeing_facts, topic)
            if summary:
                self.store.save_artifact(summary)
                results["consolidated"] = {
                    "summary_id": summary.id,
                    "title": summary.title,
                    "fact_count": len(agreeing_facts),
                }
        
        return results
    
    def run_consolidation_sweep(
        self,
        min_facts: int = 3,
        similarity_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Run a full consolidation sweep over all facts.
        
        This is a background maintenance task that:
        1. Groups similar facts
        2. Creates summaries for agreeing groups
        3. Flags conflicts
        
        Returns:
            Summary of consolidation results
        """
        all_facts = self.store.get_all_by_type("FactArtifact", current_only=True)
        
        results = {
            "total_facts": len(all_facts),
            "groups_found": 0,
            "summaries_created": 0,
            "conflicts_found": 0,
        }
        
        processed_ids = set()
        
        for fact in all_facts:
            if fact.id in processed_ids:
                continue
            
            # Find similar facts
            similar = self.store.find_similar_facts(fact, threshold=similarity_threshold)
            
            if len(similar) < min_facts - 1:
                continue
            
            results["groups_found"] += 1
            
            # Process the group
            group = [fact] + [f for f, _ in similar]
            processed_ids.update(f.id for f in group)
            
            # Check relationships within group
            agreeing = [fact]
            for other, _ in similar:
                relationship, _ = self.check_conflict(fact, other)
                if relationship == 'agree':
                    agreeing.append(other)
                elif relationship == 'conflict':
                    results["conflicts_found"] += 1
            
            # Consolidate if enough agreeing facts
            if len(agreeing) >= min_facts:
                topic = ", ".join(fact.topic_tags[:3]) if fact.topic_tags else "General"
                summary = self.consolidate_facts(agreeing, topic)
                if summary:
                    self.store.save_artifact(summary)
                    results["summaries_created"] += 1
        
        return results


# ============================================================================
# Lifecycle Manager
# ============================================================================

class LifecycleManager:
    """
    Manages the artifact lifecycle - extraction, storage, and consolidation.
    
    The "Observer" that runs after agent responses to capture and
    structure all valuable information.
    """
    
    def __init__(self, store: ArtifactStore):
        self.store = store
        
        # Extractors
        self.reasoning_extractor = ReasoningExtractor()
        self.entity_extractor = EntityExtractor()
        self.event_extractor = EventExtractor()
        self.fact_extractor = FactExtractor()
        self.fact_consolidator = FactConsolidator(store)
        
        # Background executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def process_thinking_trace(
        self,
        user_query: str,
        thinking_trace: str,
        final_answer: str,
        provenance_id: Optional[str] = None
    ) -> List[Artifact]:
        """
        Process a thinking trace to extract artifacts.
        
        Called after each agent response to capture reasoning patterns.
        
        Args:
            user_query: The original query
            thinking_trace: The <thinking> content from the model
            final_answer: The final answer
            provenance_id: ID of the source message
            
        Returns:
            List of extracted and saved artifacts
        """
        artifacts = []
        
        # Extract reasoning artifact
        reasoning = self.reasoning_extractor(
            user_query=user_query,
            raw_model_trace=thinking_trace,
            final_answer=final_answer
        )
        
        if reasoning:
            reasoning.provenance_id = provenance_id
            saved = self.store.save_artifact(reasoning)
            artifacts.append(saved)
        
        return artifacts
    
    def process_conversation_turn(
        self,
        speaker: str,
        content: str,
        turn_timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None
    ) -> Tuple[ConversationTurnArtifact, List[Artifact]]:
        """
        Process a conversation turn to extract all artifacts.
        
        Args:
            speaker: Who spoke
            content: What was said
            turn_timestamp: When it was said
            session_id: Session identifier
            
        Returns:
            Tuple of (turn_artifact, extracted_artifacts)
        """
        # Create the conversation turn artifact
        turn = ConversationTurnArtifact(
            speaker=speaker,
            content=content,
            turn_timestamp=turn_timestamp,
            session_id=session_id,
        )
        saved_turn = self.store.save_artifact(turn)
        
        extracted = []
        
        # Extract entities
        entities = self.entity_extractor(
            text=content,
            context=f"Speaker: {speaker}"
        )
        for entity in entities:
            entity.provenance_id = saved_turn.id
            saved = self.store.save_artifact(entity)
            extracted.append(saved)
            turn.extracted_artifact_ids.append(saved.id)
        
        # Extract events
        events = self.event_extractor(
            text=content,
            reference_date=turn_timestamp
        )
        for event in events:
            event.provenance_id = saved_turn.id
            saved = self.store.save_artifact(event)
            extracted.append(saved)
            turn.extracted_artifact_ids.append(saved.id)
        
        # Extract facts
        facts = self.fact_extractor(
            text=content,
            source_context=f"Stated by {speaker}"
        )
        for fact in facts:
            fact.provenance_id = saved_turn.id
            saved = self.store.save_artifact(fact)
            extracted.append(saved)
            turn.extracted_artifact_ids.append(saved.id)
            
            # Process for conflicts/consolidation (async in production)
            self.fact_consolidator.process_new_fact(saved)
        
        # Update turn with extracted IDs
        self.store.save_artifact(turn, update_of=saved_turn.id)
        
        return saved_turn, extracted
    
    def process_thinking_async(
        self,
        user_query: str,
        thinking_trace: str,
        final_answer: str,
        provenance_id: Optional[str] = None
    ):
        """
        Process thinking trace asynchronously.
        
        Use this to avoid blocking the response while extracting reasoning.
        """
        self.executor.submit(
            self.process_thinking_trace,
            user_query,
            thinking_trace,
            final_answer,
            provenance_id
        )
    
    def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance tasks like fact consolidation.
        
        Should be called periodically or after batch processing.
        """
        return self.fact_consolidator.run_consolidation_sweep()
    
    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """
        Extract thinking content from a response with <thinking> tags.
        
        Args:
            response: The full model response
            
        Returns:
            Tuple of (thinking_content, answer_content)
        """
        thinking_pattern = r"<thinking>(.*?)</thinking>"
        match = re.search(thinking_pattern, response, re.DOTALL)
        
        if match:
            thinking = match.group(1).strip()
            # Remove thinking from response to get answer
            answer = re.sub(thinking_pattern, "", response, flags=re.DOTALL).strip()
            return thinking, answer
        
        return "", response.strip()

