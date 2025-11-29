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
"""

import os
import sys
import json
import argparse
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import dspy
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "a-mem"))

from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation, LoCoMoSample

# Import AMS components
from schemas import QueryIntent
from storage import ArtifactStore
from agent import AMSAgent, create_ams_agent, create_ollama_agent


# ============================================================================
# Metrics (simplified version - can use a-mem/utils.py for full metrics)
# ============================================================================

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    text = str(text).lower()
    for char in '.,:;!?"\'':
        text = text.replace(char, ' ')
    return text.split()


def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 score."""
    # Ensure strings
    prediction = str(prediction) if prediction else ""
    reference = str(reference) if reference else ""
    
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, reference: str) -> int:
    """Calculate exact match (case-insensitive)."""
    # Ensure strings
    prediction = str(prediction) if prediction else ""
    reference = str(reference) if reference else ""
    return int(prediction.strip().lower() == reference.strip().lower())


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Convert None to empty string
    prediction = str(prediction) if prediction else ""
    reference = str(reference) if reference else ""
    
    if not prediction.strip() or not reference.strip():
        return {"exact_match": 0, "f1": 0.0}
    
    return {
        "exact_match": calculate_exact_match(prediction, reference),
        "f1": calculate_f1(prediction, reference),
    }


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


# ============================================================================
# AMS Evaluator
# ============================================================================

class AMSEvaluator:
    """Evaluates the AMS system on LoCoMo."""
    
    def __init__(
        self,
        model: str = "gpt-5-mini",
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
    
    def ingest_sample(self, sample: LoCoMoSample, cache_path: Optional[Path] = None) -> int:
        """
        Ingest all conversation turns from a sample into the agent.
        
        Args:
            sample: The LoCoMo sample
            cache_path: Optional path to cache/load from
            
        Returns:
            Number of artifacts extracted
        """
        # Try to load from cache
        if cache_path and cache_path.exists():
            self.agent.load(cache_path)
            return self.agent.store.get_stats()["total_artifacts"]
        
        total_extracted = 0
        total_turns = 0
        skipped_turns = 0
        
        for session_id, session in sample.conversation.sessions.items():
            turn_count = 0
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
                    
                    # Parse timestamp
                    timestamp = None
                    if session.date_time:
                        try:
                            timestamp = datetime.strptime(session.date_time, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            try:
                                timestamp = datetime.strptime(session.date_time, "%Y-%m-%d")
                            except ValueError:
                                pass
                    
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
    
    def answer_question(self, qa: QA) -> Tuple[str, str, int, str]:
        """
        Answer a question using the AMS agent.
        
        Returns:
            Tuple of (prediction, thinking, retrieved_count, intent)
        """
        try:
            response = self.agent(qa.question, category=qa.category)
            return (
                response.answer,
                response.thinking,
                response.retrieved_artifacts,
                response.intent.value
            )
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            return str(e), "", 0, "error"
    
    def evaluate(
        self,
        samples: List[LoCoMoSample],
        questions: Optional[List[Tuple[int, QA]]] = None,
        cache_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
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
            
            # Ingest sample
            cache_path = cache_dir / f"sample_{sample_idx}" if cache_dir else None
            try:
                extracted = self.ingest_sample(sample, cache_path)
                log.info(f"Sample {sample_idx}: ingested {extracted} artifacts")
            except Exception as e:
                log.error(f"Error ingesting sample {sample_idx}: {e}")
                continue
            
            # Answer questions
            log.info(f"Sample {sample_idx}: Answering {len(sample_qas)} questions...")
            for q_idx, qa in enumerate(tqdm(sample_qas, desc=f"Sample {sample_idx}", leave=False)):
                if self.debug:
                    log.info(f"  [DEBUG] Q{q_idx+1}/{len(sample_qas)}: {qa.question[:80]}...")
                
                prediction, thinking, retrieved, intent = self.answer_question(qa)
                
                reference = qa.final_answer or qa.answer or ""
                metrics = calculate_metrics(prediction, reference)
                
                if self.debug:
                    log.info(f"  [DEBUG] Answer: {str(prediction)[:100]}...")
                    log.info(f"  [DEBUG] Reference: {str(reference)[:100]}...")
                    log.info(f"  [DEBUG] F1={metrics['f1']:.3f}, EM={metrics['exact_match']}, Retrieved={retrieved}")
                
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
                )
                
                results.results.append(result)
                results.total_questions += 1
                results.category_counts[qa.category or 1] += 1
                
                log.debug(f"Q: {qa.question[:50]}... -> {prediction[:50]}... (F1: {metrics['f1']:.2f})")
        
        # Calculate aggregate metrics
        results.aggregate_metrics = self._aggregate_metrics(results.results)
        
        return results
    
    def _aggregate_metrics(self, results: List[QuestionResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across all results."""
        if not results:
            return {}
        
        aggregates = {"overall": defaultdict(list)}
        category_aggregates = defaultdict(lambda: defaultdict(list))
        
        for r in results:
            for metric, value in r.metrics.items():
                aggregates["overall"][metric].append(value)
                category_aggregates[r.category][metric].append(value)
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0, "count": 0}
            return {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
        
        result = {
            "overall": {k: compute_stats(v) for k, v in aggregates["overall"].items()}
        }
        
        for cat, metrics in category_aggregates.items():
            result[f"category_{cat}"] = {k: compute_stats(v) for k, v in metrics.items()}
        
        return result


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
            
            # Answer questions
            for qa in tqdm(sample_qas, desc=f"A-MEM {sample_idx}", leave=False):
                try:
                    prediction = self._answer_amem(memory_system, qa)
                except Exception as e:
                    log.warning(f"A-MEM error: {e}")
                    prediction = str(e)
                
                reference = qa.final_answer or qa.answer or ""
                metrics = calculate_metrics(prediction, reference)
                
                result = QuestionResult(
                    sample_id=sample_idx,
                    question=qa.question,
                    category=qa.category or 1,
                    reference=reference,
                    prediction=prediction,
                    thinking="",  # A-MEM doesn't capture thinking
                    metrics=metrics,
                    retrieved_artifacts=0,
                    intent="unknown",
                )
                
                results.results.append(result)
                results.total_questions += 1
                results.category_counts[qa.category or 1] += 1
        
        # Aggregate
        results.aggregate_metrics = self._aggregate_metrics(results.results)
        
        return results
    
    def _answer_amem(self, memory_system, qa: QA) -> str:
        """Answer using A-MEM."""
        # Retrieve context
        context = memory_system.find_related_memories_raw(qa.question, k=self.retrieve_k)
        
        # Generate answer (simplified - using the memory system's LLM)
        prompt = f"""Based on the context: {context}
        
        Answer the following question briefly.
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
            return json.loads(response)["answer"]
        except:
            return response
    
    def _aggregate_metrics(self, results: List[QuestionResult]) -> Dict[str, Dict[str, float]]:
        """Same aggregation as AMS."""
        if not results:
            return {}
        
        aggregates = {"overall": defaultdict(list)}
        category_aggregates = defaultdict(lambda: defaultdict(list))
        
        for r in results:
            for metric, value in r.metrics.items():
                aggregates["overall"][metric].append(value)
                category_aggregates[r.category][metric].append(value)
        
        def compute_stats(values):
            return {"mean": sum(values) / len(values), "count": len(values)} if values else {"mean": 0, "count": 0}
        
        result = {"overall": {k: compute_stats(v) for k, v in aggregates["overall"].items()}}
        for cat, metrics in category_aggregates.items():
            result[f"category_{cat}"] = {k: compute_stats(v) for k, v in metrics.items()}
        
        return result


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
    
    # Limit to first n_samples
    n_samples = min(n_samples, len(samples))
    
    # Collect all questions from the selected samples
    all_questions = []
    for sample_idx in range(n_samples):
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
    
    def print_system_results(results: EvaluationResults, name: str):
        print(f"\n{name} ({results.model})")
        print("-" * 40)
        print(f"Total questions: {results.total_questions}")
        
        if results.aggregate_metrics:
            overall = results.aggregate_metrics.get("overall", {})
            f1 = overall.get("f1", {}).get("mean", 0)
            em = overall.get("exact_match", {}).get("mean", 0)
            print(f"Overall F1: {f1:.4f}")
            print(f"Overall EM: {em:.4f}")
            
            print("\nBy Category:")
            for cat in sorted(results.category_counts.keys()):
                cat_key = f"category_{cat}"
                if cat_key in results.aggregate_metrics:
                    cat_f1 = results.aggregate_metrics[cat_key].get("f1", {}).get("mean", 0)
                    cat_em = results.aggregate_metrics[cat_key].get("exact_match", {}).get("mean", 0)
                    count = results.category_counts[cat]
                    print(f"  Cat {cat}: F1={cat_f1:.4f}, EM={cat_em:.4f} (n={count})")
    
    print_system_results(ams_results, "AMS (Agent Memory Scaffolding)")
    
    if amem_results:
        print_system_results(amem_results, "A-MEM (Baseline)")
        
        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON: AMS vs A-MEM")
        print("=" * 70)
        
        ams_overall = ams_results.aggregate_metrics.get("overall", {})
        amem_overall = amem_results.aggregate_metrics.get("overall", {})
        
        ams_f1 = ams_overall.get("f1", {}).get("mean", 0)
        amem_f1 = amem_overall.get("f1", {}).get("mean", 0)
        
        delta = ams_f1 - amem_f1
        delta_pct = (delta / amem_f1 * 100) if amem_f1 > 0 else 0
        
        print(f"\nOverall F1 Delta: {delta:+.4f} ({delta_pct:+.1f}%)")
        
        if delta > 0:
            print("✅ AMS outperforms A-MEM")
        elif delta < 0:
            print("❌ A-MEM outperforms AMS")
        else:
            print("➖ Equal performance")


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
    parser = argparse.ArgumentParser(
        description="Evaluate AMS on LoCoMo benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 1 sample (default) - all questions from first conversation
  python test.py
  
  # Run on 3 samples
  python test.py --n_samples 3
  
  # QUICK SANITY CHECK: 1 sample, 5 turns/session, 100 chars/turn, 1 question
  python test.py --debug --max_turns 5 --max_chars 100 --n_questions 1
  
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
                        help="Directory to cache ingested samples")
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
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_file, args.verbose)
    
    # Resolve paths
    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        # Try absolute path
        dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    output_path = Path(args.output) if args.output else None
    
    # Load dataset
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

