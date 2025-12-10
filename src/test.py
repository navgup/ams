#!/usr/bin/env python3
"""
AMS Test Runner - Evaluate AMS on LoCoMo Benchmark

This script runs the AMS (Agent Memory Scaffolding) system on the LoCoMo dataset
with support for:
- Variable models (OpenAI, Ollama, etc.)
- Variable dataset size (N random questions or ratio-based sampling)
- Optional comparison against A-MEM baseline
- Comprehensive metrics reporting

Usage:
    # Run AMS on 100 random questions
    python test.py --n_questions 100 --model gpt-4o-mini
    
    # Run with A-MEM comparison
    python test.py --n_questions 50 --compare_amem
    
    # Run on specific categories
    python test.py --categories 1 2 3 --ratio 0.5


    A-MEM (Baseline) (gpt-4o-mini)
------------------------------------------------------------
Total questions: 199
Artifact / Memory Stats:
  Total generated: 419
  Total retrieved/context items: 0
  Avg retrieved per question: 0.00

Overall Metrics (199 questions):
  exact_match         : 0.0452
  f1                  : 0.4800
  rouge1_f            : 0.4902
  rougeL_f            : 0.4798
  bleu1               : 0.3892
  bert_f1             : 0.9138
  meteor              : 0.3438
  sbert_similarity    : 0.5906
  llm_judge           : 0.5975 

By Category (F1 / BLEU-1 / BERT-F1 / SBERT):
  Cat 1: 0.153 / 0.103 / 0.865 / 0.360 (n=32)
  Cat 2: 0.492 / 0.370 / 0.911 / 0.690 (n=37)
  Cat 3: 0.210 / 0.167 / 0.875 / 0.322 (n=13)
  Cat 4: 0.437 / 0.368 / 0.909 / 0.549 (n=70)
  Cat 5 (adv): 0.832 / 0.693 / 0.966 / 0.805 (n=47)
"""

import os
import sys
import json
import argparse
import logging
import pickle
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ast

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

import dspy
from dateutil import parser as dateutil_parser
from tqdm import tqdm
from openai import OpenAI

# Suppress transformers warnings about uninitialized pooler weights
# (harmless when using models for feature extraction like BERT-score)
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message=".*pooler.*were not initialized")
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference")

sys.path.insert(0, str(Path(__file__).parent.parent / "a-mem"))

from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation, LoCoMoSample, EventSummary, Observation

# Import metrics from a-mem utils (comprehensive: ROUGE, BLEU, BERT, METEOR, SBERT)
from utils import calculate_metrics, aggregate_metrics as amem_aggregate_metrics

# Import AMS components
from schemas import QueryIntent
from storage import ArtifactStore
from agent import AMSAgent, create_ams_agent, create_ollama_agent
import nltk


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO or free-form datetime strings and return naive UTC."""
    if not value:
        return None
    try:
        dt = dateutil_parser.isoparse(value)
    except (ValueError, TypeError):
        try:
            dt = dateutil_parser.parse(value)
        except (ValueError, TypeError):
            return None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def create_dummy_dataset() -> List[LoCoMoSample]:
    """
    Create a tiny synthetic dataset for end-to-end testing.
    
    This creates a minimal dataset with:
    - 1 sample
    - 1 session with 5 short turns (~200 tokens total)
    - 2 simple QA questions
    
    Used with --dummy flag to verify the entire pipeline works
    without burning tokens/time on the full LoCoMo dataset.
    """
    # Create 5 simple conversation turns
    turns = [
        Turn(speaker="Alice", dia_id="s1_t1", text="Hey Bob! I just got back from my trip to Paris."),
        Turn(speaker="Bob", dia_id="s1_t2", text="That's awesome Alice! How was it?"),
        Turn(speaker="Alice", dia_id="s1_t3", text="It was amazing. I visited the Eiffel Tower and ate so many croissants."),
        Turn(speaker="Bob", dia_id="s1_t4", text="Nice! I'm planning to go to Tokyo next month."),
        Turn(speaker="Alice", dia_id="s1_t5", text="Oh Tokyo is great! You should try the ramen there."),
    ]
    
    # Create session
    session = Session(
        session_id=1,
        date_time="2024-06-15T10:00:00Z",
        turns=turns
    )
    
    # Create conversation
    conversation = Conversation(
        speaker_a="Alice",
        speaker_b="Bob",
        sessions={1: session}
    )
    
    # Create 2 simple QA questions
    qa_list = [
        QA(
            question="Where did Alice travel to recently?",
            answer="Paris",
            evidence=["s1:s1_t1"],
            category=1,  # Single-hop
            adversarial_answer=None
        ),
        QA(
            question="Where is Bob planning to go next month?",
            answer="Tokyo",
            evidence=["s1:s1_t4"],
            category=1,  # Single-hop
            adversarial_answer=None
        ),
    ]
    
    # Create minimal event summary and observation (required fields)
    event_summary = EventSummary(events={
        "session_1": {
            "Alice": ["traveled to Paris", "visited Eiffel Tower"],
            "Bob": ["planning Tokyo trip"]
        }
    })
    
    observation = Observation(observations={
        "session_1": {
            "Alice": [["loves travel", "s1_t1"]],
            "Bob": [["interested in Japan", "s1_t4"]]
        }
    })
    
    # Create sample
    sample = LoCoMoSample(
        sample_id="dummy_0",
        qa=qa_list,
        conversation=conversation,
        event_summary=event_summary,
        observation=observation,
        session_summary={"session_1": "Alice and Bob discussed travel plans."}
    )
    
    return [sample]


# ============================================================================
# Result Storage
# ============================================================================

@dataclass
class QuestionResult:
    """Result for a single question."""
    sample_id: int
    question: str
    category: int
    reference: str
    prediction: str
    thinking: str
    metrics: Dict[str, float]
    retrieved_artifacts: int
    intent: str
    artifact_summaries: List[str] = field(default_factory=list)
    retrieval_path: Optional[str] = None  # "fast" or "slow" - whether ContextSelector was skipped
    reasoning_applied: bool = False  # Whether a stored reasoning strategy was applied
    
    
@dataclass 
class EvaluationResults:
    """Full evaluation results."""
    system: str  # "ams" or "amem"
    model: str
    timestamp: str
    total_questions: int
    category_counts: Dict[int, int]
    results: List[QuestionResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMJudge:
    """
    LLM-as-a-judge using Together AI (OpenAI-compatible API).
    
    Adds a semantic correctness score (0.0-1.0) as an extra metric: 'llm_judge'.
    """
    
    def __init__(
        self,
        model: str,
        max_workers: int = 10,
        timeout: float = 15.0,
        base_url: str = "https://api.together.xyz/v1",
    ):
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            logging.warning("TOGETHER_API_KEY not set; LLM judge will be disabled.")
            self.client: Optional[OpenAI] = None
        else:
            # Use OpenAI client pointed at Together's API
            self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def _score_single(self, question: str, reference: str, prediction: str) -> Optional[float]:
        """Score a single QA pair, returning a float in [0.0, 1.0]."""
        if not self.client:
            return None
        
        system_prompt = (
            "You are a strict automatic judge for a QA benchmark.\n"
            "Given a question, the ground truth answer, and a model's answer, "
            "assign a numerical score between 0.0 and 1.0 indicating how "
            "correct the model's answer is relative to the ground truth answer.\n"
            "1.0 = perfectly correct and equivalent; 0.0 = completely wrong.\n"
            "Be robust to minor paraphrases or formatting differences. Focus only on the accuracy of the answer compared to the reference, not semantic differences. \n"
            "Do not penalize the model for being more verbose than the reference, as long as the additional information is correct."
            "Return ONLY the numeric score as a decimal number, with no explanation."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Gold answer: {reference}\n\n"
            f"Model answer: {prediction}\n\n"
            "Score (0.0-1.0):"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=16,
                temperature=0.0,
                timeout=self.timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                return None
            first_token = content.split()[0]
            score = float(first_token)
            # Clamp to [0, 1]
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            return score
        except Exception as e:
            logging.warning(f"LLM judge error: {e}")
            return None
    
    def score_results(self, eval_results: EvaluationResults):
        """
        Attach 'llm_judge' scores to each QuestionResult in-place.
        
        Uses a small thread pool to avoid slowing evaluation too much.
        """
        if not self.client:
            return
        if not eval_results.results:
            return
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_result = {}
            for r in eval_results.results:
                future = executor.submit(
                    self._score_single,
                    r.question,
                    r.reference,
                    r.prediction,
                )
                future_to_result[future] = r
            
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                score = future.result()
                if score is not None:
                    result.metrics["llm_judge"] = score


# ============================================================================
# AMS Evaluator
# ============================================================================

class AMSEvaluator:
    """Evaluates the AMS system on LoCoMo."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: str = "openai",
        storage_path: Optional[str] = None,
        k_stage1: int = 50,
        k_stage2: int = 10,
        ollama_url: str = "http://localhost:11434",
        debug: bool = False,
        max_turns_per_session: int = 0,  # 0 = no limit
        max_content_chars: int = 0,  # 0 = no limit
    ):
        self.model = model
        self.backend = backend
        self.storage_path = storage_path
        self.k_stage1 = k_stage1
        self.k_stage2 = k_stage2
        self.ollama_url = ollama_url
        self.debug = debug
        self.max_turns_per_session = max_turns_per_session
        self.max_content_chars = max_content_chars
        
        # Will be initialized per sample
        self.agent: Optional[AMSAgent] = None
        
    def _create_agent(self) -> AMSAgent:
        """Create a new AMS agent."""
        if self.backend == "openai":
            return create_ams_agent(
                model=self.model,
                storage_path=self.storage_path,
                k_stage1=self.k_stage1,
                k_stage2=self.k_stage2,
            )
        elif self.backend == "ollama":
            return create_ollama_agent(
                model=self.model,
                storage_path=self.storage_path,
                base_url=self.ollama_url,
                k_stage1=self.k_stage1,
                k_stage2=self.k_stage2,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def ingest_sample(self, sample: LoCoMoSample, cache_path: Optional[Path] = None, preload_path: Optional[Path] = None) -> int:
        """
        Ingest all conversation turns from a sample into the agent.
        
        Args:
            sample: The LoCoMo sample
            cache_path: Optional path to cache/load from
            preload_path: Optional path to preloaded artifacts (artifacts.json, indices, embeddings)
            
        Returns:
            Number of artifacts extracted
        """
        # Try to load from preload dir (artifacts saved earlier) or cache
        if preload_path and preload_path.exists():
            self.agent.load(preload_path)
            return self.agent.store.get_stats()["total_artifacts"]
        if cache_path and cache_path.exists():
            self.agent.load(cache_path)
            return self.agent.store.get_stats()["total_artifacts"]
        
        total_extracted = 0
        total_turns = 0
        skipped_turns = 0
        
        for session_id, session in sample.conversation.sessions.items():
            turn_count = 0
            session_timestamp = parse_datetime(session.date_time)
            for turn in session.turns:
                # Debug mode: limit turns per session
                if self.max_turns_per_session > 0 and turn_count >= self.max_turns_per_session:
                    skipped_turns += 1
                    continue
                
                try:
                    # Skip turns with empty or None text
                    content = turn.text or ""
                    if not content.strip():
                        skipped_turns += 1
                        continue
                    
                    # Debug mode: truncate content
                    if self.max_content_chars > 0:
                        content = content[:self.max_content_chars]
                    
                    timestamp = session_timestamp
                    
                    _, extracted = self.agent.ingest_conversation(
                        speaker=turn.speaker,
                        content=content,
                        timestamp=timestamp,
                        session_id=str(session_id)
                    )
                    total_extracted += extracted
                    total_turns += 1
                    turn_count += 1
                    
                    if self.debug and total_turns % 10 == 0:
                        logging.info(f"  [DEBUG] Ingested {total_turns} turns, {total_extracted} artifacts so far...")
                        
                except Exception as e:
                    logging.warning(f"Error ingesting turn: {e}")
                    skipped_turns += 1
        
        if self.debug:
            logging.info(f"  [DEBUG] Ingestion complete: {total_turns} turns, {skipped_turns} skipped, {total_extracted} artifacts")
        
        # Save to cache if path provided
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
            self.agent.save(cache_path)
        
        return total_extracted
    
    def answer_question(self, qa: QA) -> Tuple[str, str, int, str, List[str], Optional[str], bool]:
        """
        Answer a question using the AMS agent.
        
        Returns:
            Tuple of (prediction, thinking, retrieved_count, intent, artifact_summaries, retrieval_path, reasoning_applied)
        """
        try:
            response = self.agent(qa.question, category=qa.category)
            # Extract retrieval_path from metadata if available
            retrieval_path = None
            if hasattr(response, "metadata") and response.metadata:
                retrieval_meta = response.metadata.get("retrieval", {})
                retrieval_path = retrieval_meta.get("retrieval_path")
            
            return (
                response.answer,
                response.thinking,
                response.retrieved_artifacts,
                response.intent.value,
                response.artifact_summaries if hasattr(response, "artifact_summaries") else [],
                retrieval_path,
                response.reasoning_applied if hasattr(response, "reasoning_applied") else False
            )
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            return str(e), "", 0, "error", [], None, False
    
    def evaluate(
        self,
        samples: List[LoCoMoSample],
        questions: Optional[List[Tuple[int, QA]]] = None,
        cache_dir: Optional[Path] = None,
        artifact_dir: Optional[Path] = None,
        load_artifacts_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> EvaluationResults:
        """
        Evaluate AMS on the given samples.
        
        Args:
            samples: List of LoCoMo samples
            questions: Optional list of (sample_idx, QA) to evaluate specific questions
            cache_dir: Directory to cache sample ingestion
            logger: Logger instance
            
        Returns:
            EvaluationResults with all metrics
        """
        log = logger or logging.getLogger(__name__)
        
        results = EvaluationResults(
            system="ams",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            total_questions=0,
            category_counts=defaultdict(int),
        )
        total_generated_artifacts = 0
        total_artifacts_retrieved = 0
        
        # If specific questions provided, group by sample
        if questions:
            sample_questions = defaultdict(list)
            for sample_idx, qa in questions:
                sample_questions[sample_idx].append(qa)
        else:
            # Evaluate all questions from all samples
            sample_questions = {
                i: sample.qa for i, sample in enumerate(samples)
            }
        
        # Process each sample
        for sample_idx in tqdm(sorted(sample_questions.keys()), desc="Processing samples"):
            sample = samples[sample_idx]
            sample_qas = sample_questions[sample_idx]
            
            # Create new agent for this sample
            self.agent = self._create_agent()
            
            # Ingest sample (or load pre-saved artifacts)
            cache_path = cache_dir / f"sample_{sample_idx}" if cache_dir else None
            preload_path = load_artifacts_dir / f"sample_{sample_idx}" if load_artifacts_dir else None
            try:
                extracted = self.ingest_sample(sample, cache_path, preload_path)
                log.info(f"Sample {sample_idx}: ingested {extracted} artifacts")
                total_generated_artifacts += extracted
            except Exception as e:
                log.error(f"Error ingesting sample {sample_idx}: {e}")
                continue
            
            # Answer questions
            log.info(f"Sample {sample_idx}: Answering {len(sample_qas)} questions...")
            for q_idx, qa in enumerate(tqdm(sample_qas, desc=f"Sample {sample_idx}", leave=False)):
                
                prediction, thinking, retrieved, intent, artifact_summaries, retrieval_path, reasoning_applied = self.answer_question(qa)
                
                # For category 5 (adversarial), the ground truth is "Not mentioned in the conversation"
                # The adversarial_answer field is the TRAP answer, not the correct one!
                if qa.category == 5:
                    reference = "Not mentioned in the conversation"
                else:
                    reference = qa.final_answer or qa.answer or ""
                metrics = calculate_metrics(prediction, reference)
                total_artifacts_retrieved += retrieved
                
                result = QuestionResult(
                    sample_id=sample_idx,
                    question=qa.question,
                    category=qa.category or 1,
                    reference=reference,
                    prediction=prediction,
                    thinking=thinking,
                    metrics=metrics,
                    retrieved_artifacts=retrieved,
                    intent=intent,
                    artifact_summaries=artifact_summaries,
                    retrieval_path=retrieval_path,
                    reasoning_applied=reasoning_applied,
                )
                
                results.results.append(result)
                results.total_questions += 1
                results.category_counts[qa.category or 1] += 1
                
                log.debug(f"Q: {qa.question[:50]}... -> {prediction[:50]}... (F1: {metrics['f1']:.2f})")

            # Optionally save full artifact store for this sample
            if artifact_dir:
                sample_artifact_path = artifact_dir / f"sample_{sample_idx}"
                sample_artifact_path.mkdir(parents=True, exist_ok=True)
                try:
                    self.agent.save(sample_artifact_path)
                    log.info(f"Sample {sample_idx}: saved artifacts to {sample_artifact_path}")
                except Exception as e:
                    log.warning(f"Failed to save artifacts for sample {sample_idx}: {e}")
        
        # Calculate aggregate metrics
        results.aggregate_metrics = self._aggregate_metrics(results.results)
        if results.total_questions > 0:
            # Count fast vs slow retrieval paths
            fast_path_count = sum(1 for r in results.results if r.retrieval_path == "fast")
            slow_path_count = sum(1 for r in results.results if r.retrieval_path == "slow")
            
            results.metadata["artifact_stats"] = {
                "total_generated": total_generated_artifacts,
                "total_retrieved": total_artifacts_retrieved,
                "avg_retrieved_per_question": total_artifacts_retrieved / results.total_questions if results.total_questions else 0,
            }
            results.metadata["retrieval_stats"] = {
                "fast_path_count": fast_path_count,
                "slow_path_count": slow_path_count,
                "fast_path_percentage": (fast_path_count / results.total_questions * 100) if results.total_questions > 0 else 0,
            }
        
        return results
    
    def _aggregate_metrics(self, results: List[QuestionResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across all results using a-mem's aggregate_metrics."""
        if not results:
            return {}
        
        # Extract metrics and categories for a-mem's aggregate function
        all_metrics = [r.metrics for r in results]
        all_categories = [r.category for r in results]
        
        return amem_aggregate_metrics(all_metrics, all_categories)


# ============================================================================
# A-MEM Baseline Evaluator
# ============================================================================

class AMEMEvaluator:
    """Evaluates the A-MEM baseline on LoCoMo."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: str = "openai",
        retrieve_k: int = 10,
        temperature_c5: float = 0.5,
    ):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        
        # Import A-MEM components
        try:
            if not hasattr(ast, "Str"):
                setattr(ast, "Str", str)
            sys.path.insert(0, str(Path(__file__).parent.parent / "a-mem"))
            from memory_layer import AgenticMemorySystem, LLMController
            self.AgenticMemorySystem = AgenticMemorySystem
            self.LLMController = LLMController
            self.available = True
        except ImportError as e:
            logging.warning(f"A-MEM not available: {e}")
            self.available = False
    
    def evaluate(
        self,
        samples: List[LoCoMoSample],
        questions: Optional[List[Tuple[int, QA]]] = None,
        cache_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ) -> Optional[EvaluationResults]:
        """Evaluate A-MEM on samples."""
        if not self.available:
            return None
        
        log = logger or logging.getLogger(__name__)
        
        results = EvaluationResults(
            system="amem",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            total_questions=0,
            category_counts=defaultdict(int),
        )
        total_generated_memories = 0
        total_context_retrieved = 0
        
        # Group questions by sample
        if questions:
            sample_questions = defaultdict(list)
            for sample_idx, qa in questions:
                sample_questions[sample_idx].append(qa)
        else:
            sample_questions = {i: s.qa for i, s in enumerate(samples)}
        
        for sample_idx in tqdm(sorted(sample_questions.keys()), desc="A-MEM Samples"):
            sample = samples[sample_idx]
            sample_qas = sample_questions[sample_idx]
            
            # Create A-MEM agent
            memory_system = self.AgenticMemorySystem(
                model_name='all-MiniLM-L6-v2',
                llm_backend=self.backend,
                llm_model=self.model,
            )
            
            # Check for cache
            cache_path = cache_dir / f"amem_sample_{sample_idx}.pkl" if cache_dir else None
            
            if cache_path and cache_path.exists():
                with open(cache_path, 'rb') as f:
                    memory_system.memories = pickle.load(f)
                memory_system.retriever = memory_system.retriever.load_from_local_memory(
                    memory_system.memories, 'all-MiniLM-L6-v2'
                )
            else:
                # Ingest conversation
                for _, session in sample.conversation.sessions.items():
                    for turn in session.turns:
                        content = f"Speaker {turn.speaker} says: {turn.text}"
                        memory_system.add_note(content, time=session.date_time)
                
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(memory_system.memories, f)
            
            # Track generated memories (notes)
            try:
                total_generated_memories += len(memory_system.memories)
            except Exception:
                pass
            
            # Answer questions
            for qa in tqdm(sample_qas, desc=f"A-MEM {sample_idx}", leave=False):
                try:
                    prediction, context = self._answer_amem(memory_system, qa)
                except Exception as e:
                    log.warning(f"A-MEM error: {e}")
                    prediction = str(e)
                    context = []
                
                # For category 5 (adversarial), the ground truth is "Not mentioned in the conversation"
                # The adversarial_answer field is the TRAP answer, not the correct one!
                if qa.category == 5:
                    reference = "Not mentioned in the conversation"
                else:
                    reference = qa.final_answer or qa.answer or ""
                metrics = calculate_metrics(prediction, reference)
                
                context_count = len(context) if isinstance(context, list) else 0
                total_context_retrieved += context_count
                result = QuestionResult(
                    sample_id=sample_idx,
                    question=qa.question,
                    category=qa.category or 1,
                    reference=reference,
                    prediction=prediction,
                    thinking="",  # A-MEM doesn't capture thinking
                    metrics=metrics,
                    retrieved_artifacts=context_count,
                    intent="unknown",
                    artifact_summaries=context if isinstance(context, list) else [],
                )
                
                results.results.append(result)
                results.total_questions += 1
                results.category_counts[qa.category or 1] += 1
        
        # Aggregate
        results.aggregate_metrics = self._aggregate_metrics(results.results)
        if results.total_questions > 0:
            results.metadata["artifact_stats"] = {
                "total_generated": total_generated_memories,
                "total_retrieved": total_context_retrieved,
                "avg_retrieved_per_question": total_context_retrieved / results.total_questions if results.total_questions else 0,
            }
        
        return results
    
    def _answer_amem(self, memory_system, qa: QA) -> Tuple[str, List[str]]:
        """Answer using A-MEM. Returns tuple of (answer, retrieved_context)."""
        # Retrieve context
        context = memory_system.find_related_memories_raw(qa.question, k=self.retrieve_k)
        
        # Category-specific prompting (mirrors AMS fairness)
        if qa.category == 5:
            prompt = f"""You must determine whether the question can be answered from the provided conversation context.
            
            Context:
            {context}
            
            Question: {qa.question}
            
            If the context clearly supports the answer, return ONLY the direct answer (no explanations, no extra words).
            If the context does NOT contain the necessary information or contradicts the question's premise, return ONLY 'Not mentioned in the conversation'.
            
            Answer:"""
        elif qa.category == 2:
            prompt = f"""
            Based on the context: {context}, answer the following question. Use the dates/phrasing from the context; keep the answer short and avoid adding subjects.
            
            Question: {qa.question}
            
            Answer:"""
        elif qa.category == 3:
            prompt = f"""
            Based on the context: {context}, write an answer as a short phrase. Use exact words from the context whenever possible.
            
            Question: {qa.question}
            
            Answer:"""
        else:
            prompt = f"""Based on the context: {context}, write an answer as a short phrase. Use exact words from the context whenever possible.
            
            Question: {qa.question}
            
            Answer:"""
        
        response = memory_system.llm_controller.llm.get_completion(
            prompt,
            response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }}
        )
        
        try:
            return json.loads(response)["answer"], context
        except:
            return response, context
    
    def _aggregate_metrics(self, results: List[QuestionResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics using a-mem's aggregate_metrics."""
        if not results:
            return {}
        
        all_metrics = [r.metrics for r in results]
        all_categories = [r.category for r in results]
        
        return amem_aggregate_metrics(all_metrics, all_categories)


# ============================================================================
# Main Evaluation Runner
# ============================================================================

def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger("ams_test")
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def select_questions(
    samples: List[LoCoMoSample],
    n_samples: int = 1,
    n_questions: Optional[int] = None,
    ratio: Optional[float] = None,
    categories: Optional[List[int]] = None,
    sample_indices: Optional[List[int]] = None,
    seed: int = 42
) -> List[Tuple[int, QA]]:
    """
    Select questions to evaluate.
    
    Args:
        samples: All LoCoMo samples
        n_samples: Number of samples to use (limits which samples we draw from)
        n_questions: Number of random questions to select (within selected samples)
        ratio: Ratio of questions to select (alternative to n_questions)
        categories: Filter to specific categories
        seed: Random seed for reproducibility
        
    Returns:
        List of (sample_idx, QA) tuples
    """
    random.seed(seed)
    
    # Determine which sample indices to include
    if sample_indices is not None:
        selected_indices = sorted(set(i for i in sample_indices if 0 <= i < len(samples)))
    else:
        selected_indices = list(range(min(n_samples, len(samples))))
    
    # Collect all questions from the selected samples
    all_questions = []
    for sample_idx in selected_indices:
        sample = samples[sample_idx]
        for qa in sample.qa:
            if categories is None or qa.category in categories:
                all_questions.append((sample_idx, qa))
    
    # Further filter by n_questions or ratio if specified
    if n_questions:
        n = min(n_questions, len(all_questions))
        return random.sample(all_questions, n)
    elif ratio:
        n = max(1, int(len(all_questions) * ratio))
        return random.sample(all_questions, n)
    else:
        return all_questions


def print_comparison(ams_results: EvaluationResults, amem_results: Optional[EvaluationResults]):
    """Print comparison between AMS and A-MEM results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Key metrics to display (from a-mem utils.py)
    # Includes optional LLM-as-a-judge metric if present.
    KEY_METRICS = ["exact_match", "f1", "rouge1_f", "rougeL_f", "bleu1", "bert_f1", "meteor", "sbert_similarity", "llm_judge"]
    
    def print_system_results(results: EvaluationResults, name: str):
        print(f"\n{name} ({results.model})")
        print("-" * 60)
        print(f"Total questions: {results.total_questions}")
        
        artifact_stats = results.metadata.get("artifact_stats") if results.metadata else None
        retrieval_stats = results.metadata.get("retrieval_stats") if results.metadata else None
        
        if artifact_stats:
            print("Artifact / Memory Stats:")
            print(f"  Total generated: {artifact_stats.get('total_generated', 0)}")
            print(f"  Total retrieved/context items: {artifact_stats.get('total_retrieved', 0)}")
            print(f"  Avg retrieved per question: {artifact_stats.get('avg_retrieved_per_question', 0):.2f}")
        
        if retrieval_stats:
            print("Retrieval Path Stats:")
            print(f"  Fast path (ContextSelector skipped): {retrieval_stats.get('fast_path_count', 0)} ({retrieval_stats.get('fast_path_percentage', 0):.1f}%)")
            print(f"  Slow path (ContextSelector used): {retrieval_stats.get('slow_path_count', 0)}")
        
        if results.aggregate_metrics:
            overall = results.aggregate_metrics.get("overall", {})
            
            # Print all key metrics
            print(f"\nOverall Metrics ({results.total_questions} questions):")
            for metric in KEY_METRICS:
                if metric in overall:
                    val = overall[metric].get("mean", 0)
                    print(f"  {metric:20s}: {val:.4f}")
            
            # Print by category (with F1 and BLEU-1 first)
            print("\nBy Category (F1 / BLEU-1 / BERT-F1 / SBERT / LLM-Judge):")
            for cat in sorted(results.category_counts.keys()):
                cat_key = f"category_{cat}"
                if cat_key in results.aggregate_metrics:
                    cat_metrics = results.aggregate_metrics[cat_key]
                    cat_f1 = cat_metrics.get("f1", {}).get("mean", 0)
                    cat_bleu1 = cat_metrics.get("bleu1", {}).get("mean", 0)
                    cat_bert = cat_metrics.get("bert_f1", {}).get("mean", 0)
                    cat_sbert = cat_metrics.get("sbert_similarity", {}).get("mean", 0)
                    cat_llm = cat_metrics.get("llm_judge", {}).get("mean")
                    cat_llm_display = f"{cat_llm:.3f}" if cat_llm is not None else "N/A"
                    count = results.category_counts[cat]
                    cat_label = f"Cat {cat}" if cat != 5 else "Cat 5 (adv)"
                    print(f"  {cat_label}: {cat_f1:.3f} / {cat_bleu1:.3f} / {cat_bert:.3f} / {cat_sbert:.3f} / {cat_llm_display} (n={count})")
    
    print_system_results(ams_results, "AMS (Agent Memory Scaffolding)")
    
    if amem_results:
        print_system_results(amem_results, "A-MEM (Baseline)")
        
        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON: AMS vs A-MEM")
        print("=" * 70)
        
        ams_overall = ams_results.aggregate_metrics.get("overall", {})
        amem_overall = amem_results.aggregate_metrics.get("overall", {})
        
        print("\nMetric Deltas (AMS - A-MEM):")
        for metric in KEY_METRICS:
            ams_val = ams_overall.get(metric, {}).get("mean", 0)
            amem_val = amem_overall.get(metric, {}).get("mean", 0)
            delta = ams_val - amem_val
            delta_pct = (delta / amem_val * 100) if amem_val > 0 else 0
            winner = "✅" if delta > 0.001 else ("❌" if delta < -0.001 else "➖")
            print(f"  {metric:20s}: {delta:+.4f} ({delta_pct:+.1f}%) {winner}")
        
        # Overall verdict based on F1
        ams_f1 = ams_overall.get("f1", {}).get("mean", 0)
        amem_f1 = amem_overall.get("f1", {}).get("mean", 0)
        if ams_f1 > amem_f1:
            print("\n✅ Overall: AMS outperforms A-MEM")
        elif ams_f1 < amem_f1:
            print("\n❌ Overall: A-MEM outperforms AMS")
        else:
            print("\n➖ Overall: Equal performance")


def save_results(
    ams_results: EvaluationResults,
    amem_results: Optional[EvaluationResults],
    output_path: Path
):
    """Save results to JSON."""
    output = {
        "ams": {
            "model": ams_results.model,
            "timestamp": ams_results.timestamp,
            "total_questions": ams_results.total_questions,
            "category_counts": dict(ams_results.category_counts),
            "aggregate_metrics": ams_results.aggregate_metrics,
            "results": [asdict(r) for r in ams_results.results],
        }
    }
    
    if amem_results:
        output["amem"] = {
            "model": amem_results.model,
            "timestamp": amem_results.timestamp,
            "total_questions": amem_results.total_questions,
            "category_counts": dict(amem_results.category_counts),
            "aggregate_metrics": amem_results.aggregate_metrics,
            "results": [asdict(r) for r in amem_results.results],
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


def main():
    nltk.download('punkt_tab')

    parser = argparse.ArgumentParser(
        description="Evaluate AMS on LoCoMo benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # END-TO-END TEST (recommended first!): Tiny synthetic data, ~5 min total
  python test.py --dummy
  
  # E2E test with A-MEM comparison
  python test.py --dummy --compare_amem
  
  # Run on 1 sample (default) - all questions from first conversation
  python test.py
  
  # Run on 3 samples
  python test.py --n_samples 3
  
  # Compare AMS vs A-MEM on 1 sample
  python test.py --n_samples 1 --compare_amem
  
  # Run on specific categories (temporal + multi-hop) from 2 samples
  python test.py --n_samples 2 --categories 2 3
  
  # Use a different model
  python test.py --model gpt-4o-mini --backend openai --n_samples 1
        """
    )
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default="../a-mem/data/locomo10.json",
                        help="Path to LoCoMo dataset")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of LoCoMo samples (conversations) to use (default: 1)")
    parser.add_argument("--n_questions", type=int, default=None,
                        help="Number of random questions to evaluate (within selected samples)")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Ratio of questions to evaluate (0.0 to 1.0)")
    parser.add_argument("--categories", type=int, nargs="+", default=None,
                        help="Filter to specific LoCoMo categories (1-5)")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None,
                        help="Explicit sample indices to run (overrides n_samples if provided)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question selection")
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "ollama"],
                        help="LLM backend")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    
    # AMS options
    parser.add_argument("--k_stage1", type=int, default=50,
                        help="Number of candidates in retrieval stage 1")
    parser.add_argument("--k_stage2", type=int, default=10,
                        help="Number of final context items")
    
    # Comparison options
    parser.add_argument("--compare_amem", action="store_true",
                        help="Also run A-MEM baseline for comparison")
    
    # Output options
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache ingested samples (default: <script_dir>/cache)")
    parser.add_argument("--artifact_dir", type=str, default=None,
                        help="Directory to save AMS artifacts per sample (default: <script_dir>/artifacts)")
    parser.add_argument("--load_artifacts_dir", type=str, default=None,
                        help="Directory of pre-saved AMS artifacts per sample (artifacts.json, indices, embeddings). If provided, ingestion is skipped for samples found here.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    
    # Debug options (for quick sanity checks with minimal tokens)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode with extra logging and optional truncation")
    parser.add_argument("--max_turns", type=int, default=0,
                        help="Debug: max turns per session to ingest (0=unlimited)")
    parser.add_argument("--max_chars", type=int, default=0,
                        help="Debug: max chars per turn content (0=unlimited)")
    
    # Dummy mode for end-to-end testing without burning tokens/time
    parser.add_argument("--dummy", action="store_true",
                        help="Use tiny synthetic dataset for end-to-end testing (5 turns, 2 QAs, ~5 min total)")
    
    # LLM-as-a-judge options (Together AI)
    parser.add_argument("--llm_judge_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
                        help="Together AI model name for LLM-as-a-judge (e.g. meta-llama/Meta-Llama-3-70B-Instruct-Turbo). "
                             "If not set, LLM-as-a-judge is disabled.")
    parser.add_argument("--llm_judge_max_workers", type=int, default=10,
                        help="Max parallel requests for LLM judge (Together API).")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_file, args.verbose)
    
    # Resolve paths (skip dataset check if using dummy mode)
    dataset_path = None
    if not args.dummy:
        dataset_path = Path(__file__).parent / args.dataset
        if not dataset_path.exists():
            # Try absolute path
            dataset_path = Path(args.dataset)
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)
    
    base_dir = Path(__file__).parent
    cache_dir = Path(args.cache_dir) if args.cache_dir else (base_dir / "cache")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else (base_dir / "artifacts")
    load_artifacts_dir = Path(args.load_artifacts_dir) if args.load_artifacts_dir else None
    output_path = Path(args.output) if args.output else None
    
    # Load dataset (or use dummy for end-to-end testing)
    if args.dummy:
        logger.info("=" * 50)
        logger.info("DUMMY MODE: Using tiny synthetic dataset for E2E testing")
        logger.info("=" * 50)
        samples = create_dummy_dataset()
        logger.info(f"Created dummy dataset: 1 sample, 5 turns, 2 QA questions")
        # In dummy mode, use all questions from the dummy sample
        questions = [(0, qa) for qa in samples[0].qa]
        logger.info(f"Using all {len(questions)} dummy questions")
    else:
        logger.info(f"Loading dataset from {dataset_path}")
        samples = load_locomo_dataset(str(dataset_path))
        logger.info(f"Loaded {len(samples)} samples")
        
        # Select questions
        questions = select_questions(
            samples,
            n_samples=args.n_samples,
            n_questions=args.n_questions,
            ratio=args.ratio,
            categories=args.categories,
            sample_indices=args.sample_indices,
            seed=args.seed
        )
        logger.info(f"Using {args.n_samples} sample(s), selected {len(questions)} questions for evaluation")
    
    if args.categories:
        logger.info(f"Filtering to categories: {args.categories}")
    
    # Category distribution
    cat_dist = defaultdict(int)
    for _, qa in questions:
        cat_dist[qa.category or 1] += 1
    logger.info(f"Category distribution: {dict(cat_dist)}")
    
    # Run AMS evaluation
    logger.info("=" * 50)
    logger.info("Running AMS Evaluation")
    logger.info("=" * 50)
    
    # Debug mode info
    if args.debug:
        logger.info("[DEBUG MODE ENABLED]")
        if args.max_turns > 0:
            logger.info(f"  Max turns per session: {args.max_turns}")
        if args.max_chars > 0:
            logger.info(f"  Max chars per turn: {args.max_chars}")
    
    ams_evaluator = AMSEvaluator(
        model=args.model,
        backend=args.backend,
        k_stage1=args.k_stage1,
        k_stage2=args.k_stage2,
        ollama_url=args.ollama_url,
        debug=args.debug,
        max_turns_per_session=args.max_turns,
        max_content_chars=args.max_chars,
    )
    
    ams_results = ams_evaluator.evaluate(
        samples=samples,
        questions=questions,
        cache_dir=cache_dir / "ams" if cache_dir else None,
        artifact_dir=artifact_dir,
        load_artifacts_dir=load_artifacts_dir,
        logger=logger,
    )
    
    # Run A-MEM comparison if requested
    amem_results = None
    if args.compare_amem:
        logger.info("=" * 50)
        logger.info("Running A-MEM Baseline Evaluation")
        logger.info("=" * 50)
        
        amem_evaluator = AMEMEvaluator(
            model=args.model,
            backend=args.backend,
        )
        
        if amem_evaluator.available:
            amem_results = amem_evaluator.evaluate(
                samples=samples,
                questions=questions,
                cache_dir=cache_dir / "amem" if cache_dir else None,
                logger=logger,
            )
        else:
            logger.warning("A-MEM not available for comparison")
    
    # Optional: run LLM-as-a-judge over results (Together AI)
    if args.llm_judge_model:
        logger.info("=" * 50)
        logger.info("Running LLM-as-a-judge evaluation (Together AI)")
        logger.info("=" * 50)
        
        judge = LLMJudge(
            model=args.llm_judge_model,
            max_workers=args.llm_judge_max_workers,
        )
        
        # Score AMS results
        judge.score_results(ams_results)
        # Recompute aggregate metrics to include 'llm_judge'
        if ams_results.results:
            ams_results.aggregate_metrics = amem_aggregate_metrics(
                [r.metrics for r in ams_results.results],
                [r.category for r in ams_results.results],
            )
        
        # Score A-MEM results if present
        if amem_results:
            judge.score_results(amem_results)
            if amem_results.results:
                amem_results.aggregate_metrics = amem_aggregate_metrics(
                    [r.metrics for r in amem_results.results],
                    [r.category for r in amem_results.results],
                )
    
    # Print results
    print_comparison(ams_results, amem_results)
    
    # Save results
    if output_path:
        save_results(ams_results, amem_results, output_path)
    else:
        # Auto-generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = Path(__file__).parent / "results" / f"eval_{args.model}_{timestamp}.json"
        default_output.parent.mkdir(exist_ok=True)
        save_results(ams_results, amem_results, default_output)


if __name__ == "__main__":
    main()

