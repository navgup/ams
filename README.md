# AMS - Agent Memory Scaffolding

A standalone chatbot agent that achieves State-of-the-Art performance on the LoCoMo benchmark by treating **Reasoning Traces** and **Intermediate Work** as structured, versioned "Artifacts" in a database.

## Key Innovation

Unlike standard RAG or A-MEM (which uses generic JSON notes), AMS:

1. **Schema-First Approach**: All artifacts have strict Pydantic schemas
2. **Versioned Artifacts**: Non-destructive updates preserve full history
3. **Meta-Cognitive Layer**: Captures reusable reasoning patterns
4. **Structured Retrieval**: DSPy modules for intelligent query routing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AMSAgent                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ QueryRouter │→ │ Hybrid Retrieval │→ │ CoT Generation    │  │
│  │ (DSPy)      │  │ (2-stage)        │  │ with <thinking>   │  │
│  └─────────────┘  └──────────────────┘  └───────────────────┘  │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ArtifactStore (Versioned)                   │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────────────────┐ │   │
│  │  │Event   │ │Fact    │ │Summary │ │ReasoningArtifact  │ │   │
│  │  │Artifact│ │Artifact│ │Artifact│ │(Meta-Cognitive)   │ │   │
│  │  └────────┘ └────────┘ └────────┘ └───────────────────┘ │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │    ConversationTurnArtifact (Raw Input)          │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ▲                                                       │
│         │              ┌────────────────────┐                  │
│         └──────────────│ LifecycleManager   │                  │
│                        │ (Observer Pattern) │                  │
│                        └────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Schemas (`schemas.py`)

Strict Pydantic models for all artifact types:

- **`EventArtifact`**: Time-bound events (crucial for temporal reasoning)
- **`FactArtifact`**: Atomic claims with validity status and conflict tracking
- **`SummaryArtifact`**: Consolidated facts and summaries
- **`ReasoningArtifact`**: Reusable reasoning patterns (meta-cognitive layer)
- **`ConversationTurnArtifact`**: Raw conversation turns for reference

### 2. Storage (`storage.py`)

Versioned artifact store with:
- Non-destructive updates (creates new versions, preserves history)
- Semantic search via embeddings
- Structured filtering based on artifact schemas

### 3. Retrieval (`retrieval.py`)

Two-stage hybrid retrieval:
- **Stage 1: QueryRouter** (DSPy) - Generates structured filters + semantic query
- **Stage 2: ContextSelector** (DSPy) - Filters ~50 candidates to ~10 relevant

### 4. Lifecycle (`lifecycle.py`)

The "Observer" that runs after responses:
- **ReasoningExtractor**: Extracts reusable strategies from CoT traces
- **FactConsolidator**: Merges agreeing facts, flags conflicts

### 5. Agent (`agent.py`)

Main orchestration class with the full agent loop:
1. Input Processing via QueryRouter
2. Retrieval Stage 1 (~50 candidates)
3. Retrieval Stage 2 (~10 selected)
4. CoT Generation with `<thinking>` tags
5. Async Lifecycle processing

## LoCoMo Benchmark Optimization

| Category | Challenge | AMS Solution |
|----------|-----------|--------------|
| 1. Single-hop | Direct fact retrieval | `FactArtifact` with semantic search |
| 2. Temporal | Time-based reasoning | `EventArtifact` with `timestamp` filtering |
| 3. Multi-hop | Chain reasoning | `ReasoningArtifact` captures bridge strategies |
| 4. Open-domain | Broad knowledge | Full semantic search across all types |
| 5. Adversarial | False premises | `AdversarialGenerator` with premise checking |

## Installation

```bash
cd src
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from agent import create_ams_agent
from datetime import datetime

# Create agent
agent = create_ams_agent(
    model="gpt-4o-mini",
    storage_path="./memory_store"
)

# Ingest conversation
agent.ingest_conversation(
    speaker="Alice",
    content="I just started working at TechCorp last month.",
    timestamp=datetime(2024, 3, 1)
)

# Answer questions
response = agent("Where does Alice work?")
print(response.answer)  # "TechCorp"
print(response.thinking)  # Shows reasoning trace
```

### Evaluation on LoCoMo Benchmark

```bash
# Run evaluation on LoCoMo dataset
cd src
python test.py --n_samples 1

# Compare against A-MEM baseline
python test.py --n_samples 1 --compare_amem

# With LLM-as-a-judge (Together AI)
export TOGETHER_API_KEY="your-key"
python test.py --n_samples 1 --llm_judge_model meta-llama/Meta-Llama-3-70B-Instruct-Turbo

# Artifacts are automatically saved to ./artifacts/ and ./cache/
```

## Evaluation & Metrics

The `test.py` script provides comprehensive evaluation on the LoCoMo benchmark:

- **Standard Metrics**: Exact Match, F1, ROUGE, BLEU, BERT-F1, METEOR, SBERT
- **LLM-as-a-Judge**: Optional semantic correctness scoring via Together AI
- **A-MEM Comparison**: Side-by-side comparison with baseline
- **Artifact Persistence**: Automatic saving of artifacts and caches

Results are saved as JSON with per-question and aggregate metrics, broken down by LoCoMo category.

## Key Differences from A-MEM

| Aspect | A-MEM | AMS |
|--------|-------|-----|
| Data Model | Generic JSON notes | Strict Pydantic schemas |
| Updates | Overwrites | Versioned (non-destructive) |
| Reasoning | Not captured | `ReasoningArtifact` stores patterns |
| Temporal | String timestamps | `datetime` with structured filtering |
| Retrieval | Simple embedding | Two-stage with QueryRouter |
| Framework | Raw prompts | DSPy Signatures & Modules |
| Entities | Explicit entity tracking | Implicit via Facts/Events |

## Requirements

- Python 3.10+
- DSPy 2.5+
- Pydantic 2.0+
- sentence-transformers
- OpenAI API key (or Ollama for local)
- Together AI API key (optional, for LLM-as-a-judge evaluation)

## File Structure

```
ams/
├── src/
│   ├── agent.py          # Main AMSAgent class
│   ├── schemas.py        # Artifact type definitions
│   ├── storage.py         # ArtifactStore with versioning
│   ├── retrieval.py      # Hybrid retrieval engine
│   ├── lifecycle.py      # Artifact extraction & consolidation
│   ├── test.py           # LoCoMo evaluation script
│   └── requirements.txt
├── cache/                # Default cache directory (auto-created)
├── artifacts/            # Default artifact storage (auto-created)
└── results/              # Evaluation results JSON files
```

## License

MIT
