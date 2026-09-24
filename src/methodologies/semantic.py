"""
Semantic Deduplication Methodology using Vector Embeddings.
Uses sentence-transformers and Milvus-lite for similarity search.
"""

import logging
from typing import Set, List, Dict, Any
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

from src.config import Config

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Semantic deduplication using vector embeddings.
    Uses sentence-transformers for encoding and Milvus-lite for storage/search.
    """

    def __init__(
        self,
        similarity_threshold: float = None,
        embedding_model: str = None
    ):
        """
        Initialize Semantic deduplicator.

        Args:
            similarity_threshold: Threshold for cosine similarity
            embedding_model: Name of sentence-transformers model
        """
        self.threshold = similarity_threshold or Config.SEMANTIC_SIMILARITY_THRESHOLD
        self.model_name = embedding_model or Config.SEMANTIC_EMBEDDING_MODEL

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize Milvus client (using milvus-lite)
        self.milvus_client = None
        self.collection_name = Config.MILVUS_COLLECTION_NAME

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform semantic deduplication on the knowledge graph.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        logger.info("Running Semantic (Embedding-based) deduplication")

        nodes = list(graph.nodes())
        n = len(nodes)

        logger.info(f"Processing {n} entities")

        # Generate embeddings for all entities
        embeddings, node_ids = self._generate_embeddings(graph)

        if len(embeddings) < 2:
            logger.warning("Not enough entities for clustering")
            return []

        # Store in Milvus for efficient similarity search
        self._store_in_milvus(embeddings, node_ids, graph)

        # Find duplicates using similarity search
        clusters = self._find_duplicates_with_milvus(embeddings, node_ids)

        # Clean up
        self._cleanup_milvus()

        logger.info(f"Found {len(clusters)} duplicate clusters using semantic analysis")

        return clusters

    def _generate_embeddings(
        self,
        graph: nx.DiGraph
    ) -> tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for all entities in the graph.

        Args:
            graph: Knowledge graph

        Returns:
            Tuple of (embeddings array, node IDs list)
        """
        logger.info("Generating embeddings...")

        texts = []
        node_ids = []

        for node, attrs in graph.nodes(data=True):
            # Create text representation of entity
            text = self._entity_to_text(attrs)
            texts.append(text)
            node_ids.append(node)

        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Generated {len(embeddings)} embeddings")

        return embeddings, node_ids

    def _entity_to_text(self, attrs: Dict) -> str:
        """
        Convert entity attributes to text for embedding.

        Args:
            attrs: Entity attributes

        Returns:
            Text representation
        """
        # Concatenate important attributes
        parts = []

        # Entity type
        entity_type = attrs.get('type', '')
        if entity_type:
            parts.append(f"Type: {entity_type}")

        # Name
        name = attrs.get('name', '')
        if name:
            parts.append(f"Name: {name}")

        # Other attributes
        for key in ['role', 'title', 'position', 'company', 'organization', 'address', 'email']:
            value = attrs.get(key, '')
            if value:
                parts.append(f"{key.title()}: {value}")

        return " | ".join(parts) if parts else "Unknown Entity"

    def _store_in_milvus(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        graph: nx.DiGraph
    ):
        """
        Store embeddings in Milvus-lite.

        Args:
            embeddings: Embedding vectors
            node_ids: Corresponding node IDs
            graph: Knowledge graph (for metadata)
        """
        logger.info("Storing embeddings in Milvus-lite...")

        # Initialize Milvus client (using local file)
        self.milvus_client = MilvusClient(str(Config.OUTPUTS_DIR / "milvus_lite.db"))

        # Drop collection if exists
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        # Create collection with schema
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",  # Cosine similarity
        )

        # Prepare data for insertion
        data = []
        for i, (node_id, embedding) in enumerate(zip(node_ids, embeddings)):
            attrs = graph.nodes[node_id]

            data.append({
                "id": i,
                "node_id": node_id,
                "vector": embedding.tolist(),
                "entity_type": attrs.get('type', 'Unknown'),
                "name": attrs.get('name', '')[:500],  # Limit length
            })

        # Insert data
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data
        )

        logger.info(f"Stored {len(data)} embeddings in Milvus")

    def _find_duplicates_with_milvus(
        self,
        embeddings: np.ndarray,
        node_ids: List[str]
    ) -> List[Set[str]]:
        """
        Find duplicates using Milvus similarity search.

        Args:
            embeddings: Embedding vectors
            node_ids: Corresponding node IDs

        Returns:
            List of duplicate clusters
        """
        logger.info("Finding duplicates with similarity search...")

        # Build similarity graph
        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(node_ids)

        # For each entity, find similar entities
        for i, (embedding, node_id) in enumerate(zip(embeddings, node_ids)):
            # Search for similar entities
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[embedding.tolist()],
                limit=20,  # Top 20 similar entities
                output_fields=["node_id"]
            )

            # Add edges for similar entities
            for hit in results[0]:
                similar_node_id = hit['entity']['node_id']
                distance = hit['distance']

                # For the COSINE metric Milvus returns the cosine similarity itself
                # in the 'distance' field (1.0 = identical), not 1 - similarity
                similarity = distance

                if similarity >= self.threshold and similar_node_id != node_id:
                    similarity_graph.add_edge(node_id, similar_node_id, weight=similarity)

        # Extract connected components as clusters
        clusters = []
        for component in nx.connected_components(similarity_graph):
            if len(component) > 1:
                clusters.append(set(component))

        return clusters

    def _find_duplicates_with_clustering(
        self,
        embeddings: np.ndarray,
        node_ids: List[str]
    ) -> List[Set[str]]:
        """
        Alternative approach: Find duplicates using agglomerative clustering.

        Args:
            embeddings: Embedding vectors
            node_ids: Corresponding node IDs

        Returns:
            List of duplicate clusters
        """
        logger.info("Finding duplicates with clustering...")

        # Use agglomerative clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.threshold,  # Convert similarity to distance
            metric='cosine',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(embeddings)

        # Group nodes by cluster
        clusters_dict = {}
        for node_id, label in zip(node_ids, cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = set()
            clusters_dict[label].add(node_id)

        # Filter out singleton clusters
        clusters = [cluster for cluster in clusters_dict.values() if len(cluster) > 1]

        return clusters

    def _cleanup_milvus(self):
        """Clean up Milvus resources."""
        if self.milvus_client:
            # Optionally drop collection
            # self.milvus_client.drop_collection(self.collection_name)
            self.milvus_client.close()

    def get_methodology_name(self) -> str:
        """Get the name of this methodology."""
        return "Semantic (Embedding-based)"

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used by this methodology."""
        return {
            "methodology": "Semantic",
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "similarity_threshold": self.threshold,
            "vector_db": "Milvus-lite",
            "metric": "cosine_similarity"
        }


def create_semantic_deduplicator() -> SemanticDeduplicator:
    """
    Create and return a SemanticDeduplicator instance.

    Returns:
        SemanticDeduplicator instance
    """
    return SemanticDeduplicator()
