"""
R-Swoosh (Recursive Swoosh) Deduplication Methodology.
Implements an iterative merge-closure algorithm for entity resolution.
"""

import logging
from typing import Set, List, Dict, Tuple, Any
import networkx as nx
from rapidfuzz import fuzz

from src.config import Config

logger = logging.getLogger(__name__)


class RSwooshDeduplicator:
    """
    R-Swoosh deduplication using iterative merge-closure.
    Mechanistic approach based on similarity functions.
    """

    def __init__(self, similarity_threshold: float = None):
        """
        Initialize R-Swoosh deduplicator.

        Args:
            similarity_threshold: Threshold for considering entities as matches
        """
        self.threshold = similarity_threshold or Config.RSWOOSH_SIMILARITY_THRESHOLD

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform R-Swoosh deduplication on the knowledge graph.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        logger.info("Running R-Swoosh deduplication")

        # Get all nodes
        nodes = list(graph.nodes())
        n = len(nodes)

        logger.info(f"Processing {n} entities")

        # Initialize: each entity in its own cluster
        clusters = {node: {node} for node in nodes}

        # Iterative swoosh
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        changed = True

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            logger.debug(f"R-Swoosh iteration {iterations}")

            # Compare all pairs of clusters
            cluster_ids = list(clusters.keys())

            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    cluster_i = cluster_ids[i]
                    cluster_j = cluster_ids[j]

                    # Skip if clusters have been merged
                    if cluster_i not in clusters or cluster_j not in clusters:
                        continue

                    # Check if clusters should be merged
                    if self._should_merge_clusters(
                        clusters[cluster_i],
                        clusters[cluster_j],
                        graph
                    ):
                        # Merge cluster_j into cluster_i
                        clusters[cluster_i] = clusters[cluster_i].union(clusters[cluster_j])
                        del clusters[cluster_j]
                        changed = True

        logger.info(
            f"R-Swoosh completed in {iterations} iterations. "
            f"Reduced {n} entities to {len(clusters)} clusters"
        )

        # Return list of clusters (filter out singleton clusters)
        result_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]

        logger.info(f"Found {len(result_clusters)} duplicate clusters")

        return result_clusters

    def _should_merge_clusters(
        self,
        cluster1: Set[str],
        cluster2: Set[str],
        graph: nx.DiGraph
    ) -> bool:
        """
        Determine if two clusters should be merged.
        Uses pairwise similarity between entities in clusters.

        Args:
            cluster1: First cluster of node IDs
            cluster2: Second cluster of node IDs
            graph: Knowledge graph

        Returns:
            True if clusters should be merged
        """
        # Compare all pairs between clusters
        max_similarity = 0.0

        for node1 in cluster1:
            for node2 in cluster2:
                similarity = self._compute_similarity(
                    graph.nodes[node1],
                    graph.nodes[node2]
                )
                max_similarity = max(max_similarity, similarity)

                # Early exit if threshold exceeded
                if max_similarity >= self.threshold:
                    return True

        return max_similarity >= self.threshold

    def _compute_similarity(self, entity1: Dict, entity2: Dict) -> float:
        """
        Compute similarity between two entities.
        Combines multiple attribute comparisons.

        Args:
            entity1: First entity attributes
            entity2: Second entity attributes

        Returns:
            Similarity score [0, 1]
        """
        # Must be same type to be duplicates
        if entity1.get('type') != entity2.get('type'):
            return 0.0

        # Compare attributes
        scores = []

        # Name comparison (most important)
        name1 = entity1.get('name', '')
        name2 = entity2.get('name', '')

        if name1 and name2:
            name_sim = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
            scores.append(name_sim * 2.0)  # Weight name more heavily

        # Compare other text attributes
        text_attrs = ['role', 'title', 'position', 'company', 'organization']

        for attr in text_attrs:
            val1 = str(entity1.get(attr, '')).lower()
            val2 = str(entity2.get(attr, '')).lower()

            if val1 and val2:
                attr_sim = fuzz.ratio(val1, val2) / 100.0
                scores.append(attr_sim)

        # Compare exact match attributes (IDs, emails, etc.)
        exact_attrs = ['id', 'email', 'phone', 'din', 'pan', 'registration_id']

        for attr in exact_attrs:
            val1 = str(entity1.get(attr, '')).lower()
            val2 = str(entity2.get(attr, '')).lower()

            if val1 and val2:
                if val1 == val2:
                    scores.append(1.0)
                else:
                    scores.append(0.0)

        # Compute average similarity
        if scores:
            return sum(scores) / len(scores)
        else:
            # No comparable attributes
            return 0.0

    def get_methodology_name(self) -> str:
        """Get the name of this methodology."""
        return "R-Swoosh"

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used by this methodology."""
        return {
            "methodology": "R-Swoosh",
            "similarity_threshold": self.threshold,
            "algorithm": "iterative_merge_closure"
        }


def create_rswoosh_deduplicator() -> RSwooshDeduplicator:
    """
    Create and return an RSwooshDeduplicator instance.

    Returns:
        RSwooshDeduplicator instance
    """
    return RSwooshDeduplicator()
