# AMS - Agent Memory Scaffolding

A standalone chatbot agent that achieves State-of-the-Art performance on the LoCoMo benchmark by treating **Reasoning Traces** and **Intermediate Work** as structured, versioned "Artifacts" in a database.


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

## To USe

```bash
cd src
pip install -r requirements.txt
python test.py --compare_amem 
```

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
