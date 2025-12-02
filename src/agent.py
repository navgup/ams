"""
AMS Agent - Main Agent Loop

The AMSAgent orchestrates the entire system:
1. Input Processing via QueryRouter
2. Two-stage Retrieval
3. CoT Generation with structured reasoning
4. Async Lifecycle processing for artifact extraction

This is the main entry point for the AMS chatbot.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

import dspy
from pydantic import BaseModel, Field

from schemas import (
    Artifact,
    EventArtifact,
    FactArtifact,
    ReasoningArtifact,
    SummaryArtifact,
    ConversationTurnArtifact,
    QueryIntent,
)
from storage import ArtifactStore
from retrieval import HybridRetrievalEngine, create_retrieval_engine
from lifecycle import LifecycleManager


# ============================================================================
# DSPy Signatures for Generation
# ============================================================================

class AnswerWithReasoningSignature(dspy.Signature):
    """
    Generate an answer with explicit reasoning in <thinking> tags.
    
    The thinking trace is crucial - it will be processed by the Lifecycle
    Manager to extract reusable reasoning patterns.
    """
    
    user_query: str = dspy.InputField(
        desc="The user's question to answer"
    )
    context: str = dspy.InputField(
        desc="Retrieved artifacts and context to use for answering"
    )
    reasoning_strategies: str = dspy.InputField(
        desc="Relevant reasoning strategies from past successful problem-solving"
    )
    
    thinking: str = dspy.OutputField(
        desc="Step-by-step reasoning process - be explicit about your logic chain"
    )
    answer: str = dspy.OutputField(
        desc="ONLY the direct answer - extremely concise, no explanations. For names give just the name. For dates use natural format like '7 May 2023'. Match the expected answer format."
    )


class MultiHopAnswerSignature(dspy.Signature):
    """
    Generate answer for multi-hop questions with bridge reasoning.
    
    Multi-hop questions require chaining multiple facts together.
    Example: "What is Taylor Hawkins' wife's occupation?"
    Requires: Find wife → Find wife's occupation
    """
    
    user_query: str = dspy.InputField(
        desc="The multi-hop question requiring chain reasoning"
    )
    context: str = dspy.InputField(
        desc="Retrieved context with entities and their relationships"
    )
    bridge_hints: str = dspy.InputField(
        desc="Hints about intermediate entities to bridge"
    )
    
    bridge_reasoning: str = dspy.OutputField(
        desc="Explicit chain of reasoning: 'A relates to B via X, B relates to C via Y'"
    )
    intermediate_facts: str = dspy.OutputField(
        desc="Key intermediate facts discovered during reasoning"
    )
    answer: str = dspy.OutputField(
        desc="ONLY the final answer - just the name, place, date, or fact. No sentences or explanations."
    )


class TemporalAnswerSignature(dspy.Signature):
    """
    Generate answer for temporal questions with time-aware reasoning.
    
    Temporal questions require understanding when events occurred
    and potentially comparing across time periods.
    """
    
    user_query: str = dspy.InputField(
        desc="The temporal question about events in time"
    )
    context: str = dspy.InputField(
        desc="Retrieved events and facts with timestamps"
    )
    current_date: str = dspy.InputField(
        desc="Current date for reference"
    )
    
    temporal_reasoning: str = dspy.OutputField(
        desc="Analysis of the temporal aspects and timeline"
    )
    answer: str = dspy.OutputField(
        desc="ONLY the date/time answer in natural format like '7 May 2023' or '2022'. No full sentences, just the temporal answer."
    )


class AdversarialAnswerSignature(dspy.Signature):
    """
    Generate answer for potentially adversarial questions.
    
    Some questions may have false premises or ask about things
    not mentioned in the conversation. The agent must recognize
    and appropriately handle these cases.
    """
    
    user_query: str = dspy.InputField(
        desc="The question which may have false premises"
    )
    context: str = dspy.InputField(
        desc="Retrieved context to verify against"
    )
    
    premise_check: str = dspy.OutputField(
        desc="Analysis of whether the question's premises are supported by context"
    )
    is_answerable: str = dspy.OutputField(
        desc="'yes' if the question can be answered from context, 'no' if not mentioned"
    )
    answer: str = dspy.OutputField(
        desc="ONLY the direct answer - concise, no explanations. If unanswerable, say ONLY 'Not mentioned in the conversation'"
    )


# ============================================================================
# Generation Modules
# ============================================================================

class ReasoningGenerator(dspy.Module):
    """Generates answers with explicit reasoning."""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(AnswerWithReasoningSignature)
    
    def forward(
        self,
        user_query: str,
        context: str,
        reasoning_strategies: str = ""
    ) -> Tuple[str, str]:
        """
        Generate an answer with reasoning.
        
        Returns:
            Tuple of (thinking_trace, answer)
        """
        result = self.generator(
            user_query=user_query,
            context=context,
            reasoning_strategies=reasoning_strategies or "No prior strategies available"
        )
        return result.thinking, result.answer


class MultiHopGenerator(dspy.Module):
    """Generates answers for multi-hop questions."""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(MultiHopAnswerSignature)
    
    def forward(
        self,
        user_query: str,
        context: str,
        bridge_hints: str = ""
    ) -> Tuple[str, str, str]:
        """
        Generate multi-hop answer.
        
        Returns:
            Tuple of (bridge_reasoning, intermediate_facts, answer)
        """
        result = self.generator(
            user_query=user_query,
            context=context,
            bridge_hints=bridge_hints or "No bridge hints"
        )
        return result.bridge_reasoning, result.intermediate_facts, result.answer


class TemporalGenerator(dspy.Module):
    """Generates answers for temporal questions."""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(TemporalAnswerSignature)
    
    def forward(
        self,
        user_query: str,
        context: str,
        current_date: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """
        Generate temporal answer.
        
        Returns:
            Tuple of (temporal_reasoning, answer)
        """
        date_str = (current_date or datetime.utcnow()).strftime("%Y-%m-%d")
        
        result = self.generator(
            user_query=user_query,
            context=context,
            current_date=date_str
        )
        return result.temporal_reasoning, result.answer


class AdversarialGenerator(dspy.Module):
    """Generates answers for potentially adversarial questions."""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(AdversarialAnswerSignature)
    
    def forward(
        self,
        user_query: str,
        context: str
    ) -> Tuple[str, bool, str]:
        """
        Generate adversarial-aware answer.
        
        Returns:
            Tuple of (premise_check, is_answerable, answer)
        """
        result = self.generator(
            user_query=user_query,
            context=context
        )
        
        is_answerable = result.is_answerable.lower().strip() == "yes"
        return result.premise_check, is_answerable, result.answer


# ============================================================================
# AMS Agent
# ============================================================================

class AMSAgentResponse(BaseModel):
    """Structured response from the AMS agent."""
    
    answer: str = Field(..., description="The final answer")
    thinking: str = Field(default="", description="The reasoning trace")
    intent: QueryIntent = Field(default=QueryIntent.FACTUAL, description="Detected query intent")
    retrieved_artifacts: int = Field(default=0, description="Number of artifacts retrieved")
    reasoning_applied: bool = Field(default=False, description="Whether a stored strategy was applied")
    artifact_summaries: List[str] = Field(default_factory=list, description="Summaries of retrieved artifacts used for answering")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AMSAgent(dspy.Module):
    """
    Agent Memory Scaffolding - Main Agent Class
    
    Orchestrates the full agent loop:
    1. Input Processing via QueryRouter
    2. Retrieval Stage 1: Filtered semantic search (~50 candidates)
    3. Retrieval Stage 2: Context selection (~5-10 relevant)
    4. Generation with CoT and <thinking> tags
    5. Async Lifecycle: Extract reasoning artifacts
    
    Key Innovation: Treats reasoning traces and intermediate work as
    structured, versioned artifacts - not just raw text logs.
    """
    
    def __init__(
        self,
        store: Optional[ArtifactStore] = None,
        storage_path: Optional[Path] = None,
        k_stage1: int = 50,
        k_stage2: int = 10,
    ):
        """
        Initialize the AMS Agent.
        
        Args:
            store: Optional pre-configured artifact store
            storage_path: Path for persistent storage
            k_stage1: Number of candidates in stage 1 retrieval
            k_stage2: Number of final context items
        """
        super().__init__()
        
        # Initialize store
        self.store = store or ArtifactStore(storage_path=storage_path)
        self.storage_path = storage_path
        
        # Retrieval configuration
        self.k_stage1 = k_stage1
        self.k_stage2 = k_stage2
        
        # Initialize components
        self.retrieval_engine = HybridRetrievalEngine(self.store)
        self.lifecycle_manager = LifecycleManager(self.store)
        
        # Generators for different query types
        self.reasoning_generator = ReasoningGenerator()
        self.multi_hop_generator = MultiHopGenerator()
        self.temporal_generator = TemporalGenerator()
        self.adversarial_generator = AdversarialGenerator()
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
    
    def forward(
        self,
        user_input: str,
        category: Optional[int] = None
    ) -> AMSAgentResponse:
        """
        Main agent loop - process user input and generate response.
        
        Args:
            user_input: The user's question or message
            category: Optional LoCoMo category (1-5) for specialized handling
            
        Returns:
            AMSAgentResponse with answer and metadata
        """
        # Build conversation context
        conversation_context = self._build_conversation_context()
        
        # =========================================
        # Step 1: Input Processing via QueryRouter
        # =========================================
        filters, semantic_query, intent = self.retrieval_engine.query_router(
            user_query=user_input,
            conversation_context=conversation_context
        )
        
        # Override intent based on LoCoMo category if provided
        if category:
            intent = self._map_category_to_intent(category)
        
        # =========================================
        # Step 2: Retrieval Stage 1 and 2 (semantic search [ + optional filter] -> agent chooses most relevant artifacts)
        # =========================================
        artifacts, retrieval_metadata = self.retrieval_engine.retrieve(
            user_query=user_input,
            conversation_context=conversation_context,
            k_stage1=self.k_stage1,
            k_stage2=self.k_stage2
        )
        
        # Build context from retrieved artifacts
        context = self._build_context(artifacts)
        artifact_summaries = [artifact.get_summary() for artifact in artifacts]
        
        # =========================================
        # Step 3: Retrieve reasoning strategies
        # =========================================
        goal_category = self._intent_to_goal_category(intent)
        strategies = self.retrieval_engine.retrieve_for_reasoning(
            goal_category=goal_category,
            user_query=user_input,
            min_rating=3
        )
        strategies_context = self._build_strategies_context(strategies)
        reasoning_applied = len(strategies) > 0
        
        # =========================================
        # Step 4: Generation with CoT
        # =========================================
        thinking, answer = self._generate_answer(
            user_query=user_input,
            context=context,
            intent=intent,
            strategies=strategies_context,
            category=category
        )
        
        # =========================================
        # Step 5: Async Lifecycle - Extract artifacts
        # =========================================
        if thinking:
            # Process thinking trace asynchronously
            self.lifecycle_manager.process_thinking_async(
                user_query=user_input,
                thinking_trace=thinking,
                final_answer=answer,
            )
        
        # Update conversation history
        self._update_history(user_input, answer)
        
        return AMSAgentResponse(
            answer=answer,
            thinking=thinking,
            intent=intent,
            retrieved_artifacts=len(artifacts),
            reasoning_applied=reasoning_applied,
            artifact_summaries=artifact_summaries,
            metadata={
                "filters": filters.model_dump() if filters else {},
                "semantic_query": semantic_query,
                "retrieval": retrieval_metadata,
                "artifact_summaries": artifact_summaries,
            }
        )
    
    def _generate_answer(
        self,
        user_query: str,
        context: str,
        intent: QueryIntent,
        strategies: str,
        category: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Generate answer based on intent type.
        
        Returns:
            Tuple of (thinking_trace, answer)
        """
        # Handle adversarial (category 5)
        if category == 5:
            premise_check, is_answerable, answer = self.adversarial_generator(
                user_query=user_query,
                context=context
            )
            if not is_answerable:
                return premise_check, "Not mentioned in the conversation"
            return premise_check, answer
        
        # Handle multi-hop
        if intent == QueryIntent.MULTI_HOP or category == 3:
            bridge_reasoning, intermediate, answer = self.multi_hop_generator(
                user_query=user_query,
                context=context,
                bridge_hints=strategies
            )
            thinking = f"Bridge reasoning: {bridge_reasoning}\nIntermediate facts: {intermediate}"
            return thinking, answer
        
        # Handle temporal
        if intent == QueryIntent.TEMPORAL or category == 2:
            temporal_reasoning, answer = self.temporal_generator(
                user_query=user_query,
                context=context
            )
            return temporal_reasoning, answer
        
        # Default: factual/reasoning
        thinking, answer = self.reasoning_generator(
            user_query=user_query,
            context=context,
            reasoning_strategies=strategies
        )
        return thinking, answer
    
    def _build_conversation_context(self) -> str:
        """Build context string from conversation history."""
        if not self.conversation_history:
            return ""
        
        lines = []
        for turn in self.conversation_history[-self.max_history:]:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(lines)
    
    def _build_context(self, artifacts: List[Artifact]) -> str:
        """Build context string from retrieved artifacts."""
        if not artifacts:
            return "No relevant context found."
        
        context_parts = []
        for artifact in artifacts:
            context_parts.append(artifact.to_context_string())
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_strategies_context(self, strategies: List[Artifact]) -> str:
        """Build context from reasoning strategies."""
        if not strategies:
            return ""
        
        parts = ["Relevant reasoning strategies from past successes:"]
        for strategy in strategies[:3]:  # Limit to top 3
            if isinstance(strategy, ReasoningArtifact):
                parts.append(f"\n{strategy.to_context_string()}")
        
        return "\n".join(parts)
    
    def _update_history(self, user_input: str, answer: str):
        """Update conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "assistant": answer
        })
        
        # Trim history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def _map_category_to_intent(self, category: int) -> QueryIntent:
        """Map LoCoMo category to QueryIntent."""
        mapping = {
            1: QueryIntent.FACTUAL,      # Single-hop factual
            2: QueryIntent.TEMPORAL,      # Temporal reasoning
            3: QueryIntent.MULTI_HOP,     # Multi-hop
            4: QueryIntent.FACTUAL,       # Open-domain
            5: QueryIntent.FACTUAL,       # Adversarial (handled separately)
        }
        return mapping.get(category, QueryIntent.FACTUAL)
    
    def _intent_to_goal_category(self, intent: QueryIntent) -> str:
        """Map QueryIntent to goal category for strategy lookup."""
        mapping = {
            QueryIntent.FACTUAL: "fact-verification",
            QueryIntent.TEMPORAL: "temporal-reasoning",
            QueryIntent.REASONING: "general-reasoning",
            QueryIntent.MULTI_HOP: "multi-hop-retrieval",
        }
        return mapping.get(intent, "general")
    
    def ingest_conversation(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Ingest a conversation turn into memory.
        
        This extracts entities, events, and facts from the turn
        and stores them as structured artifacts.
        
        Args:
            speaker: Who spoke
            content: What was said
            timestamp: When it was said
            session_id: Session identifier
            
        Returns:
            Tuple of (turn_id, number_of_extracted_artifacts)
        """
        turn, extracted = self.lifecycle_manager.process_conversation_turn(
            speaker=speaker,
            content=content,
            turn_timestamp=timestamp,
            session_id=session_id
        )
        
        return turn.id, len(extracted)
    
    def save(self, path: Optional[Path] = None):
        """Save agent state to disk."""
        save_path = path or self.storage_path
        if save_path:
            self.store.save_to_disk(save_path)
    
    def load(self, path: Optional[Path] = None):
        """Load agent state from disk."""
        load_path = path or self.storage_path
        if load_path and Path(load_path).exists():
            self.store.load_from_disk(load_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        store_stats = self.store.get_stats()
        return {
            "store": store_stats,
            "conversation_history_length": len(self.conversation_history),
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_ams_agent(
    model: str = "gpt-5-mini",
    storage_path: Optional[str] = None,
    k_stage1: int = 50,
    k_stage2: int = 10,
    temperature: float = 0.7, #1 for reasoning models 
    max_tokens: int = 1000, #>= 16000 for reasoning
) -> AMSAgent:
    """
    Create and configure an AMS agent.
    
    Args:
        model: The LLM model to use (OpenAI model name)
        storage_path: Optional path for persistent storage
        k_stage1: Number of candidates in first retrieval stage
        k_stage2: Number of final context items
        temperature: LLM temperature
        max_tokens: Max tokens for LLM responses
        
    Returns:
        Configured AMSAgent
    """
    # Configure DSPy with the LLM (explicit params to avoid reasoning model detection)
    lm = dspy.LM(f"openai/{model}", temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    
    # Create storage path
    path = Path(storage_path) if storage_path else None
    
    # Create and return agent
    return AMSAgent(
        storage_path=path,
        k_stage1=k_stage1,
        k_stage2=k_stage2,
    )


def create_ollama_agent(
    model: str = "llama3.2",
    storage_path: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    k_stage1: int = 50,
    k_stage2: int = 10,
) -> AMSAgent:
    """
    Create an AMS agent using Ollama backend.
    
    Args:
        model: The Ollama model name
        storage_path: Optional path for persistent storage
        base_url: Ollama API base URL
        k_stage1: Number of candidates in first retrieval stage
        k_stage2: Number of final context items
        
    Returns:
        Configured AMSAgent
    """
    # Configure DSPy with Ollama
    lm = dspy.LM(f"ollama_chat/{model}", api_base=base_url, temperature=0.7, max_tokens=4096)
    dspy.configure(lm=lm)
    
    # Create storage path
    path = Path(storage_path) if storage_path else None
    
    return AMSAgent(
        storage_path=path,
        k_stage1=k_stage1,
        k_stage2=k_stage2,
    )

