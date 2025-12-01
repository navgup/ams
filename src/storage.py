"""
AMS Storage Layer - Artifact Store with Versioning Support

Provides in-memory and persistent storage for artifacts with:
- Automatic versioning (non-destructive updates)
- Semantic search via embeddings
- Structured filtering based on artifact schemas
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4
import json
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize datetime to naive UTC for consistent comparison.
    
    This fixes "can't compare offset-naive and offset-aware datetimes" errors.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC then strip timezone
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

from schemas import (
    Artifact,
    EntityArtifact,
    EventArtifact,
    FactArtifact,
    ReasoningArtifact,
    SummaryArtifact,
    ConversationTurnArtifact,
    StructuredFilter,
    ValidityStatus,
    ArtifactType,
    ARTIFACT_TYPES,
    deserialize_artifact,
)


class ArtifactStore:
    """
    Storage backend for AMS artifacts with versioning and retrieval.
    
    Features:
    - Non-destructive versioning: updates create new versions, preserving history
    - Semantic search: embeddings-based similarity search
    - Structured filtering: type-based and temporal filtering
    - Persistence: save/load to disk
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the artifact store.
        
        Args:
            embedding_model: SentenceTransformer model name for embeddings
            storage_path: Optional path for persistent storage
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Primary storage: id -> artifact
        self.artifacts: Dict[str, Artifact] = {}
        
        # Index by type for efficient filtering
        self.type_index: Dict[str, List[str]] = {
            artifact_type: [] for artifact_type in ARTIFACT_TYPES.keys()
        }
        
        # Version chains: original_id -> list of version ids (oldest to newest)
        self.version_chains: Dict[str, List[str]] = {}
        
        # Embedding matrix for semantic search
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_ids: List[str] = []  # Maps embedding index to artifact id
        
        # Entity name index for deduplication
        self.entity_name_index: Dict[str, str] = {}  # normalized_name -> artifact_id
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedding_model.encode([text])[0]
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return name.lower().strip()
    
    def _find_existing_entity(self, entity: EntityArtifact) -> Optional[str]:
        """
        Find an existing entity that matches this one.
        
        Checks:
        1. Exact name match
        2. Alias match
        
        Returns:
            ID of existing entity if found, None otherwise
        """
        # Check primary name
        normalized = self._normalize_name(entity.name)
        if normalized in self.entity_name_index:
            return self.entity_name_index[normalized]
        
        # Check aliases
        for alias in entity.aliases:
            normalized_alias = self._normalize_name(alias)
            if normalized_alias in self.entity_name_index:
                return self.entity_name_index[normalized_alias]
        
        return None
    
    def save_artifact(self, artifact: Artifact, update_of: Optional[str] = None) -> Artifact:
        """
        Save an artifact with proper versioning.
        
        Versioning Logic:
        1. Creates a new row with a new UUID
        2. If updating existing, sets previous_version_id to old UUID
        3. Updates latest_version_id pointer on all versions
        4. Marks old version as is_current=False
        
        Args:
            artifact: The artifact to save
            update_of: ID of artifact this is updating (for explicit versioning)
            
        Returns:
            The saved artifact with updated fields
        """
        # Handle entity deduplication
        if isinstance(artifact, EntityArtifact):
            existing_id = self._find_existing_entity(artifact)
            if existing_id and not update_of:
                update_of = existing_id
        
        # Generate new ID for this version
        new_id = str(uuid4())
        artifact.id = new_id
        artifact.created_at = datetime.utcnow()
        
        # Handle versioning if this updates an existing artifact
        if update_of and update_of in self.artifacts:
            old_artifact = self.artifacts[update_of]
            
            # Set version chain links
            artifact.previous_version_id = update_of
            artifact.is_current = True
            
            # Mark old version as not current
            old_artifact.is_current = False
            old_artifact.latest_version_id = new_id
            
            # Update entire version chain
            if update_of in self.version_chains:
                chain_root = self._get_chain_root(update_of)
                for old_id in self.version_chains[chain_root]:
                    if old_id in self.artifacts:
                        self.artifacts[old_id].latest_version_id = new_id
                self.version_chains[chain_root].append(new_id)
            else:
                # Start new chain
                self.version_chains[update_of] = [update_of, new_id]
        else:
            artifact.is_current = True
            artifact.latest_version_id = None
        
        # Generate and store embedding
        embedding_text = artifact.get_embedding_text()
        embedding = self._generate_embedding(embedding_text)
        artifact.embedding = embedding.tolist()
        
        # Store artifact
        self.artifacts[new_id] = artifact
        
        # Update type index
        artifact_type = artifact.__class__.__name__
        if artifact_type in self.type_index:
            self.type_index[artifact_type].append(new_id)
        
        # Update entity name index
        if isinstance(artifact, EntityArtifact):
            self.entity_name_index[self._normalize_name(artifact.name)] = new_id
            for alias in artifact.aliases:
                self.entity_name_index[self._normalize_name(alias)] = new_id
        
        # Update embedding matrix
        self._update_embeddings(new_id, embedding)
        
        return artifact
    
    def _get_chain_root(self, artifact_id: str) -> str:
        """Get the root ID of a version chain."""
        for root, chain in self.version_chains.items():
            if artifact_id in chain:
                return root
        return artifact_id
    
    def _update_embeddings(self, artifact_id: str, embedding: np.ndarray):
        """Update the embedding matrix with a new artifact."""
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            self.embedding_ids = [artifact_id]
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.embedding_ids.append(artifact_id)
    
    def get_artifact(self, artifact_id: str, include_history: bool = False) -> Optional[Union[Artifact, Tuple[Artifact, List[Artifact]]]]:
        """
        Retrieve an artifact by ID.
        
        Args:
            artifact_id: The artifact ID
            include_history: If True, also return version history
            
        Returns:
            The artifact, or (artifact, history) if include_history=True
        """
        artifact = self.artifacts.get(artifact_id)
        
        if not artifact:
            return None
        
        if not include_history:
            return artifact
        
        # Get version history
        history = []
        chain_root = self._get_chain_root(artifact_id)
        if chain_root in self.version_chains:
            for vid in self.version_chains[chain_root]:
                if vid != artifact_id and vid in self.artifacts:
                    history.append(self.artifacts[vid])
        
        return artifact, history
    
    def get_current_version(self, artifact_id: str) -> Optional[Artifact]:
        """Get the current (latest) version of an artifact."""
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            return None
        
        if artifact.is_current:
            return artifact
        
        if artifact.latest_version_id:
            return self.artifacts.get(artifact.latest_version_id)
        
        return artifact
    
    def semantic_search(
        self,
        query: str,
        k: int = 50,
        artifact_type: Optional[str] = None,
        current_only: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Perform semantic search over artifacts.
        
        Args:
            query: Search query
            k: Number of results to return
            artifact_type: Optional filter by artifact type
            current_only: Only return current versions
            
        Returns:
            List of (artifact_id, score) tuples, sorted by score descending
        """
        if self.embeddings is None or len(self.embedding_ids) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Create (id, score) pairs with filtering
        results = []
        for idx, (aid, score) in enumerate(zip(self.embedding_ids, similarities)):
            artifact = self.artifacts.get(aid)
            if not artifact:
                continue
            
            # Apply filters
            if current_only and not artifact.is_current:
                continue
            
            if artifact_type and artifact.__class__.__name__ != artifact_type:
                continue
            
            results.append((aid, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def filtered_search(
        self,
        query: str,
        filters: StructuredFilter,
        k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Perform filtered semantic search.
        
        Args:
            query: Semantic search query
            filters: Structured filters to apply
            k: Number of results
            
        Returns:
            List of (artifact_id, score) tuples
        """
        # Get candidates from semantic search
        artifact_type = filters.artifact_type.value if filters.artifact_type else None
        candidates = self.semantic_search(query, k=k*2, artifact_type=artifact_type)
        
        # Apply additional filters
        filtered = []
        for aid, score in candidates:
            artifact = self.artifacts.get(aid)
            if not artifact:
                continue
            
            # Temporal filter
            if filters.temporal_filter:
                if not hasattr(artifact, 'timestamp') or artifact.timestamp is None:
                    continue
                
                # Normalize both datetimes to naive UTC for consistent comparison
                artifact_time = _normalize_datetime(artifact.timestamp)
                try:
                    filter_time = _normalize_datetime(datetime.fromisoformat(filters.temporal_filter.value))
                except (ValueError, TypeError):
                    # Skip this filter if we can't parse the value
                    continue
                
                if artifact_time is None or filter_time is None:
                    continue
                
                op = filters.temporal_filter.operator
                
                if op == "$gt" and not artifact_time > filter_time:
                    continue
                elif op == "$lt" and not artifact_time < filter_time:
                    continue
                elif op == "$gte" and not artifact_time >= filter_time:
                    continue
                elif op == "$lte" and not artifact_time <= filter_time:
                    continue
                elif op == "$eq" and artifact_time.date() != filter_time.date():
                    continue
            
            # Topic tags filter
            if filters.topic_tags and hasattr(artifact, 'topic_tags'):
                if not any(tag in artifact.topic_tags for tag in filters.topic_tags):
                    continue
            
            # Entity name filter
            if filters.entity_name and hasattr(artifact, 'name'):
                if filters.entity_name.lower() not in artifact.name.lower():
                    continue
            
            # Validity status filter
            if filters.validity_status and hasattr(artifact, 'validity_status'):
                if artifact.validity_status != filters.validity_status:
                    continue
            
            filtered.append((aid, score))
        
        return filtered[:k]
    
    def get_all_by_type(self, artifact_type: str, current_only: bool = True) -> List[Artifact]:
        """Get all artifacts of a specific type."""
        ids = self.type_index.get(artifact_type, [])
        artifacts = []
        for aid in ids:
            artifact = self.artifacts.get(aid)
            if artifact:
                if current_only and not artifact.is_current:
                    continue
                artifacts.append(artifact)
        return artifacts
    
    def find_similar_facts(self, fact: FactArtifact, threshold: float = 0.8) -> List[Tuple[FactArtifact, float]]:
        """
        Find facts similar to the given fact.
        
        Used by FactConsolidator to find candidates for merging.
        
        Args:
            fact: The fact to find similar facts for
            threshold: Minimum similarity threshold
            
        Returns:
            List of (fact, similarity) tuples
        """
        results = self.semantic_search(
            fact.claim,
            k=20,
            artifact_type="FactArtifact",
            current_only=True
        )
        
        similar = []
        for aid, score in results:
            if aid == fact.id:
                continue
            if score < threshold:
                continue
            
            artifact = self.artifacts.get(aid)
            if isinstance(artifact, FactArtifact):
                similar.append((artifact, score))
        
        return similar
    
    def find_reasoning_strategies(
        self,
        goal_category: Optional[str] = None,
        min_rating: int = 3
    ) -> List[ReasoningArtifact]:
        """
        Find relevant reasoning strategies.
        
        Args:
            goal_category: Optional filter by goal category
            min_rating: Minimum outcome rating
            
        Returns:
            List of reasoning artifacts
        """
        strategies = []
        for artifact in self.get_all_by_type("ReasoningArtifact"):
            if not isinstance(artifact, ReasoningArtifact):
                continue
            
            if artifact.outcome_rating < min_rating:
                continue
            
            if goal_category and artifact.goal_category != goal_category:
                continue
            
            strategies.append(artifact)
        
        # Sort by rating descending
        strategies.sort(key=lambda x: x.outcome_rating, reverse=True)
        return strategies
    
    def save_to_disk(self, path: Optional[Path] = None):
        """Save the store to disk."""
        save_path = path or self.storage_path
        if not save_path:
            raise ValueError("No storage path specified")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts
        artifacts_data = {}
        for aid, artifact in self.artifacts.items():
            data = artifact.model_dump()
            data['_class'] = artifact.__class__.__name__
            artifacts_data[aid] = data
        
        with open(save_path / "artifacts.json", "w") as f:
            json.dump(artifacts_data, f, default=str, indent=2)
        
        # Save indices
        indices = {
            'type_index': self.type_index,
            'version_chains': self.version_chains,
            'entity_name_index': self.entity_name_index,
            'embedding_ids': self.embedding_ids,
        }
        with open(save_path / "indices.pkl", "wb") as f:
            pickle.dump(indices, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(save_path / "embeddings.npy", self.embeddings)
    
    def load_from_disk(self, path: Optional[Path] = None):
        """Load the store from disk."""
        load_path = path or self.storage_path
        if not load_path:
            raise ValueError("No storage path specified")
        
        load_path = Path(load_path)
        
        # Load artifacts
        with open(load_path / "artifacts.json", "r") as f:
            artifacts_data = json.load(f)
        
        for aid, data in artifacts_data.items():
            class_name = data.pop('_class')
            artifact_class = ARTIFACT_TYPES.get(class_name)
            if artifact_class:
                # Handle datetime fields
                for field in ['created_at', 'timestamp', 'turn_timestamp', 'consolidated_at']:
                    if field in data and data[field]:
                        data[field] = datetime.fromisoformat(data[field])
                # Handle enum fields
                if 'validity_status' in data and data['validity_status']:
                    data['validity_status'] = ValidityStatus(data['validity_status'])
                
                self.artifacts[aid] = artifact_class(**data)
        
        # Load indices
        with open(load_path / "indices.pkl", "rb") as f:
            indices = pickle.load(f)
        
        self.type_index = indices['type_index']
        self.version_chains = indices['version_chains']
        self.entity_name_index = indices['entity_name_index']
        self.embedding_ids = indices['embedding_ids']
        
        # Load embeddings
        embeddings_path = load_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        type_counts = {t: len(ids) for t, ids in self.type_index.items()}
        current_count = sum(1 for a in self.artifacts.values() if a.is_current)
        
        return {
            "total_artifacts": len(self.artifacts),
            "current_artifacts": current_count,
            "type_counts": type_counts,
            "version_chains": len(self.version_chains),
            "entities_indexed": len(self.entity_name_index),
        }

