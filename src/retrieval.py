"""
AMS Retrieval Layer - Hybrid Retrieval Engine with DSPy

Implements a two-stage retrieval process:
1. QueryRouter: Analyzes query to generate structured filters + semantic query
2. ContextSelector: Filters candidates to select the most relevant artifacts

Uses DSPy Signatures and Modules for all cognitive components.
No raw prompt strings - everything is declarative.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json

import dspy
from pydantic import BaseModel, Field

from schemas import (
    Artifact,
    ArtifactType,
    QueryIntent,
    StructuredFilter,
    TemporalFilter,
    ValidityStatus,
)
from storage import ArtifactStore


# ============================================================================
# DSPy Signatures
# ============================================================================

class QueryRouterSignature(dspy.Signature):
    """
    Analyze a user query to determine:
    1. Structured filters for database queries
    2. Optimized semantic search query
    3. Query intent classification
    
    This enables schema-aware retrieval that can filter by artifact type,
    timestamp ranges, and other structured fields before semantic search.
    """
    
    user_query: str = dspy.InputField(
        desc="The user's question or request"
    )
    conversation_context: str = dspy.InputField(
        desc="Recent conversation context for disambiguation"
    )
    
    # Structured outputs
    artifact_type: str = dspy.OutputField(
        desc="Most relevant artifact type: EntityArtifact, EventArtifact, FactArtifact, ReasoningArtifact, or 'any'"
    )
    temporal_operator: str = dspy.OutputField(
        desc="Temporal operator if query involves time: $gt, $lt, $gte, $lte, $eq, or 'none'"
    )
    temporal_value: str = dspy.OutputField(
        desc="ISO date string for temporal filter (YYYY-MM-DD), or 'none'"
    )
    semantic_query: str = dspy.OutputField(
        desc="Optimized query for semantic search - extract key concepts and entities"
    )
    intent: str = dspy.OutputField(
        desc="Query intent: factual, temporal, reasoning, or multi_hop"
    )
    entity_filter: str = dspy.OutputField(
        desc="Entity name to filter by, or 'none'"
    )
    topic_tags: str = dspy.OutputField(
        desc="Comma-separated topic tags to filter by, or 'none'"
    )


class ContextSelectorSignature(dspy.Signature):
    """
    Given candidate artifacts and a query, select the most relevant ones.
    
    This filters out:
    - Duplicates (same information from different sources)
    - Outdated information (when newer versions exist)
    - Irrelevant artifacts (even if semantically similar)
    
    Goal: From ~50 candidates, select 5-10 that actually answer the question.
    """
    
    user_query: str = dspy.InputField(
        desc="The user's question to answer"
    )
    candidate_summaries: str = dspy.InputField(
        desc="JSON list of candidate artifacts with id and summary"
    )
    
    selected_ids: str = dspy.OutputField(
        desc="Comma-separated list of artifact IDs to keep (5-10 most relevant)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of selection criteria and why others were excluded"
    )


class MultiHopPlannerSignature(dspy.Signature):
    """
    For multi-hop questions, plan the retrieval strategy.
    
    Example: "What is Taylor Hawkins' wife's occupation?"
    Plan: 1. Find Taylor Hawkins entity → 2. Find spouse relationship → 3. Find spouse's occupation
    
    This enables "bridge" strategies crucial for LoCoMo multi-hop tasks.
    """
    
    user_query: str = dspy.InputField(
        desc="The multi-hop question requiring chain reasoning"
    )
    available_context: str = dspy.InputField(
        desc="Currently available context from previous retrieval"
    )
    
    requires_more_retrieval: str = dspy.OutputField(
        desc="'yes' if more retrieval is needed, 'no' if we have enough"
    )
    next_query: str = dspy.OutputField(
        desc="If more retrieval needed, the query for the next hop"
    )
    bridge_reasoning: str = dspy.OutputField(
        desc="Explanation of how this connects to the original query"
    )


# ============================================================================
# DSPy Modules
# ============================================================================

class QueryRouter(dspy.Module):
    """
    Routes queries to appropriate retrieval strategy.
    
    Analyzes the user query and conversation context to produce:
    1. Structured filters for the artifact database
    2. Optimized semantic search query
    3. Intent classification for downstream processing
    """
    
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(QueryRouterSignature)
    
    def forward(
        self,
        user_query: str,
        conversation_context: str = ""
    ) -> Tuple[StructuredFilter, str, QueryIntent]:
        """
        Route a query to structured filters and semantic query.
        
        Args:
            user_query: The user's question
            conversation_context: Recent conversation for context
            
        Returns:
            Tuple of (structured_filters, semantic_query, intent)
        """
        # Get routing decision from LLM
        result = self.router(
            user_query=user_query,
            conversation_context=conversation_context or "No prior context"
        )
        
        # Parse artifact type
        artifact_type = None
        if result.artifact_type and result.artifact_type.lower() != "any":
            try:
                artifact_type = ArtifactType(result.artifact_type)
            except ValueError:
                pass
        
        # Parse temporal filter
        temporal_filter = None
        if result.temporal_operator and result.temporal_operator.lower() != "none":
            if result.temporal_value and result.temporal_value.lower() != "none":
                temporal_filter = TemporalFilter(
                    field="timestamp",
                    operator=result.temporal_operator,
                    value=result.temporal_value
                )
        
        # Parse topic tags
        topic_tags = None
        if result.topic_tags and result.topic_tags.lower() != "none":
            topic_tags = [t.strip() for t in result.topic_tags.split(",") if t.strip()]
        
        # Parse entity filter
        entity_filter = None
        if result.entity_filter and result.entity_filter.lower() != "none":
            entity_filter = result.entity_filter
        
        # Build structured filter
        filters = StructuredFilter(
            artifact_type=artifact_type,
            temporal_filter=temporal_filter,
            topic_tags=topic_tags,
            entity_name=entity_filter,
        )
        
        # Parse intent
        intent_map = {
            "factual": QueryIntent.FACTUAL,
            "temporal": QueryIntent.TEMPORAL,
            "reasoning": QueryIntent.REASONING,
            "multi_hop": QueryIntent.MULTI_HOP,
        }
        intent = intent_map.get(result.intent.lower(), QueryIntent.FACTUAL)
        
        return filters, result.semantic_query, intent


class ContextSelector(dspy.Module):
    """
    Selects the most relevant artifacts from candidates.
    
    Given ~50 semantic search results, filters down to 5-10 that:
    - Actually answer the question
    - Are not duplicates
    - Are current (not outdated)
    """
    
    def __init__(self):
        super().__init__()
        self.selector = dspy.ChainOfThought(ContextSelectorSignature)
    
    def forward(
        self,
        user_query: str,
        candidates: List[Tuple[str, str, float]]  # (id, summary, score)
    ) -> Tuple[List[str], str]:
        """
        Select most relevant artifacts from candidates.
        
        Args:
            user_query: The user's question
            candidates: List of (artifact_id, summary, score) tuples
            
        Returns:
            Tuple of (selected_artifact_ids, reasoning)
        """
        if not candidates:
            return [], "No candidates to select from"
        
        # Format candidates for the LLM
        candidate_data = [
            {"id": cid, "summary": summary, "relevance_score": f"{score:.2f}"}
            for cid, summary, score in candidates
        ]
        
        result = self.selector(
            user_query=user_query,
            candidate_summaries=json.dumps(candidate_data, indent=2)
        )
        
        # Parse selected IDs
        selected_ids = [
            sid.strip() 
            for sid in result.selected_ids.split(",") 
            if sid.strip()
        ]
        
        # Validate IDs exist in candidates
        valid_ids = {cid for cid, _, _ in candidates}
        selected_ids = [sid for sid in selected_ids if sid in valid_ids]
        
        return selected_ids, result.reasoning


class MultiHopPlanner(dspy.Module):
    """
    Plans multi-hop retrieval for complex questions.
    
    Identifies when a question requires chained retrieval:
    "Who is X's wife's employer?" → needs to find wife first, then employer
    """
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(MultiHopPlannerSignature)
    
    def forward(
        self,
        user_query: str,
        available_context: str = ""
    ) -> Tuple[bool, str, str]:
        """
        Plan the next retrieval hop.
        
        Args:
            user_query: The original multi-hop question
            available_context: Context gathered so far
            
        Returns:
            Tuple of (needs_more_retrieval, next_query, bridge_reasoning)
        """
        result = self.planner(
            user_query=user_query,
            available_context=available_context or "No context retrieved yet"
        )
        
        needs_more = result.requires_more_retrieval.lower().strip() == "yes"
        
        return needs_more, result.next_query, result.bridge_reasoning


# ============================================================================
# Hybrid Retrieval Engine
# ============================================================================

class HybridRetrievalEngine:
    """
    Two-stage retrieval engine combining structured filtering with semantic search.
    
    Stage 1: QueryRouter generates filters → filtered semantic search → ~50 candidates
    Stage 2: ContextSelector filters to ~5-10 relevant artifacts
    
    Special handling for multi-hop queries via iterative retrieval.
    """
    
    def __init__(self, store: ArtifactStore, max_hops: int = 3):
        """
        Initialize the retrieval engine.
        
        Args:
            store: The artifact store to search
            max_hops: Maximum retrieval hops for multi-hop queries
        """
        self.store = store
        self.max_hops = max_hops
        
        # DSPy modules
        self.query_router = QueryRouter()
        self.context_selector = ContextSelector()
        self.multi_hop_planner = MultiHopPlanner()
    
    def retrieve(
        self,
        user_query: str,
        conversation_context: str = "",
        k_stage1: int = 50,
        k_stage2: int = 10
    ) -> Tuple[List[Artifact], Dict[str, Any]]:
        """
        Perform two-stage retrieval.
        
        Args:
            user_query: The user's question
            conversation_context: Recent conversation context
            k_stage1: Number of candidates from stage 1
            k_stage2: Number of final results from stage 2
            
        Returns:
            Tuple of (selected_artifacts, retrieval_metadata)
        """
        metadata = {
            "user_query": user_query,
            "stages": [],
            "hops": []
        }
        
        # Stage 1: Query routing and filtered search
        filters, semantic_query, intent = self.query_router(
            user_query=user_query,
            conversation_context=conversation_context
        )
        
        metadata["filters"] = filters.model_dump()
        metadata["semantic_query"] = semantic_query
        metadata["intent"] = intent.value
        
        # Perform filtered semantic search
        if filters.artifact_type or filters.temporal_filter or filters.topic_tags:
            search_results = self.store.filtered_search(
                query=semantic_query,
                filters=filters,
                k=k_stage1
            )
        else:
            search_results = self.store.semantic_search(
                query=semantic_query,
                k=k_stage1
            )
        
        metadata["stages"].append({
            "stage": 1,
            "candidates": len(search_results)
        })
        
        # Handle multi-hop queries
        if intent == QueryIntent.MULTI_HOP:
            search_results, hop_metadata = self._handle_multi_hop(
                user_query=user_query,
                initial_results=search_results,
                k=k_stage1
            )
            metadata["hops"] = hop_metadata
        
        # Stage 2: Context selection
        candidates = []
        for artifact_id, score in search_results:
            artifact = self.store.get_artifact(artifact_id)
            if artifact:
                candidates.append((artifact_id, artifact.get_summary(), score))
        
        if not candidates:
            return [], metadata
        
        selected_ids, selection_reasoning = self.context_selector(
            user_query=user_query,
            candidates=candidates
        )
        
        metadata["stages"].append({
            "stage": 2,
            "selected": len(selected_ids),
            "reasoning": selection_reasoning
        })
        
        # Gather selected artifacts
        selected_artifacts = []
        for aid in selected_ids[:k_stage2]:
            artifact = self.store.get_artifact(aid)
            if artifact:
                selected_artifacts.append(artifact)
        
        return selected_artifacts, metadata
    
    def _handle_multi_hop(
        self,
        user_query: str,
        initial_results: List[Tuple[str, float]],
        k: int
    ) -> Tuple[List[Tuple[str, float]], List[Dict]]:
        """
        Handle multi-hop retrieval.
        
        Iteratively retrieves more context until the planner determines
        we have enough information to answer the question.
        """
        all_results = list(initial_results)
        seen_ids = {aid for aid, _ in all_results}
        hop_metadata = []
        
        # Build context from initial results
        context_parts = []
        for artifact_id, _ in initial_results[:10]:
            artifact = self.store.get_artifact(artifact_id)
            if artifact:
                context_parts.append(artifact.to_context_string())
        current_context = "\n\n".join(context_parts)
        
        for hop in range(self.max_hops):
            needs_more, next_query, bridge_reasoning = self.multi_hop_planner(
                user_query=user_query,
                available_context=current_context
            )
            
            hop_metadata.append({
                "hop": hop + 1,
                "needs_more": needs_more,
                "next_query": next_query if needs_more else None,
                "bridge_reasoning": bridge_reasoning
            })
            
            if not needs_more:
                break
            
            # Perform additional retrieval
            hop_results = self.store.semantic_search(
                query=next_query,
                k=k // 2
            )
            
            # Add new results
            for artifact_id, score in hop_results:
                if artifact_id not in seen_ids:
                    all_results.append((artifact_id, score))
                    seen_ids.add(artifact_id)
                    
                    artifact = self.store.get_artifact(artifact_id)
                    if artifact:
                        context_parts.append(artifact.to_context_string())
            
            current_context = "\n\n".join(context_parts[-20:])  # Keep context manageable
        
        return all_results, hop_metadata
    
    def retrieve_for_reasoning(
        self,
        goal_category: str,
        user_query: str,
        min_rating: int = 3
    ) -> List[Artifact]:
        """
        Retrieve relevant reasoning strategies for a goal.
        
        Args:
            goal_category: The category of reasoning goal
            user_query: The query to match strategies against
            min_rating: Minimum outcome rating
            
        Returns:
            List of relevant ReasoningArtifacts
        """
        # Get all reasoning strategies
        strategies = self.store.find_reasoning_strategies(
            goal_category=goal_category,
            min_rating=min_rating
        )
        
        if not strategies:
            # Fall back to semantic search
            results = self.store.semantic_search(
                query=f"{goal_category}: {user_query}",
                k=5,
                artifact_type="ReasoningArtifact"
            )
            
            strategies = []
            for aid, _ in results:
                artifact = self.store.get_artifact(aid)
                if artifact:
                    strategies.append(artifact)
        
        return strategies


# ============================================================================
# Convenience Functions
# ============================================================================

def create_retrieval_engine(
    store: ArtifactStore,
    lm: Optional[dspy.LM] = None
) -> HybridRetrievalEngine:
    """
    Create and configure a retrieval engine.
    
    Args:
        store: The artifact store
        lm: Optional DSPy language model (uses default if not provided)
        
    Returns:
        Configured HybridRetrievalEngine
    """
    if lm:
        dspy.configure(lm=lm)
    
    return HybridRetrievalEngine(store=store)

