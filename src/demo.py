#!/usr/bin/env python3
"""
AMS Demo - Example usage of the Agent Memory Scaffolding system.

This demonstrates:
1. Creating and configuring the AMS agent
2. Ingesting conversation data
3. Answering questions with structured reasoning
4. Viewing extracted artifacts
"""

import os
from datetime import datetime
from pathlib import Path

import dspy

from schemas import (
    EntityArtifact,
    EventArtifact,
    FactArtifact,
    ReasoningArtifact,
)
from storage import ArtifactStore
from agent import AMSAgent, create_ams_agent


def demo_basic_usage():
    """Demonstrate basic AMS agent usage."""
    print("=" * 60)
    print("AMS Demo: Basic Usage")
    print("=" * 60)
    
    # Configure DSPy (using OpenAI - set OPENAI_API_KEY env var)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create the agent
    agent = create_ams_agent(
        model="gpt-5-mini",
        storage_path="./demo_storage",
        k_stage1=50,
        k_stage2=10
    )
    
    # Ingest some conversation data
    print("\n Ingesting conversation...")
    
    conversations = [
        {
            "speaker": "Alice",
            "content": "I just got back from my trip to Japan last week. I visited Tokyo and Kyoto.",
            "timestamp": datetime(2024, 3, 15, 10, 0)
        },
        {
            "speaker": "Bob", 
            "content": "That's amazing! How long were you there?",
            "timestamp": datetime(2024, 3, 15, 10, 1)
        },
        {
            "speaker": "Alice",
            "content": "I was there for two weeks. I started working at TechCorp as a software engineer last month.",
            "timestamp": datetime(2024, 3, 15, 10, 2)
        },
        {
            "speaker": "Bob",
            "content": "Congratulations on the new job! My sister Sarah also works in tech, she's a data scientist at DataLabs.",
            "timestamp": datetime(2024, 3, 15, 10, 3)
        },
    ]
    
    for conv in conversations:
        turn_id, extracted = agent.ingest_conversation(
            speaker=conv["speaker"],
            content=conv["content"],
            timestamp=conv["timestamp"],
            session_id="demo-session"
        )
        print(f" Ingested turn from {conv['speaker']}: {extracted} artifacts extracted")
    
    # Show storage stats
    stats = agent.get_stats()
    print(f"\n Storage Stats: {stats['store']['total_artifacts']} total artifacts")
    for type_name, count in stats['store']['type_counts'].items():
        if count > 0:
            print(f"  - {type_name}: {count}")
    
    # Ask questions
    print("\n? Asking questions...")
    
    questions = [
        ("Where did Alice travel to?", 1),  # Factual
        ("When did Alice visit Japan?", 2),  # Temporal
        ("What does Bob's sister do?", 3),  # Multi-hop
    ]
    
    for question, category in questions:
        print(f"\nQ: {question}")
        response = agent(question, category=category)
        print(f"A: {response.answer}")
        print(f"   (Intent: {response.intent.value}, Artifacts used: {response.retrieved_artifacts})")
    
    # Save state
    agent.save()
    print("\n Agent state saved to ./demo_storage")

"""
LoCoMo Categories and AMS Handling:

1. Single-hop Factual (Category 1)
   → Uses: FactArtifact retrieval
   → Example: "What is Alice's job?"
   
2. Temporal Reasoning (Category 2)  
   → Uses: EventArtifact with timestamp filtering
   → QueryRouter generates: {"type": "EventArtifact", "timestamp": {"$gt": "2022"}}
   → Example: "When did Alice start her new job?"
   
3. Multi-hop Reasoning (Category 3)
   → Uses: MultiHopPlanner for bridge strategies
   → ReasoningArtifact captures patterns like "To find X's wife's job, first find X's wife"
   → Example: "What does Bob's sister's company specialize in?"
   
4. Open-domain (Category 4)
   → Uses: Full semantic search across all artifacts
   → Example: General knowledge questions
   
5. Adversarial (Category 5)
   → Uses: AdversarialGenerator with premise checking
   → Detects false premises and "not mentioned" cases
   → Example: "What color is Alice's car?" (never mentioned)

Key Innovation:
Unlike A-MEM which stores raw JSON notes, AMS:
- Captures reasoning traces as ReasoningArtifact (meta-cognitive layer)
- Enforces schema on all data (Pydantic models)
- Versions artifacts instead of overwriting
- Uses structured filters for temporal queries
"""


if __name__ == "__main__":
    demo_basic_usage()

