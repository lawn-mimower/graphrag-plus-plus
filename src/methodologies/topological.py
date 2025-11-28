"""
Topological Deduplication Methodology using Graph Structure.
Uses Jaccard similarity on node neighborhoods to identify duplicates.
"""

import logging
from typing import Set, List, Dict, Any, Tuple
import networkx as nx
from itertools import combinations

from src.config import Config

logger = logging.getLogger(__name__)


class TopologicalDeduplicator:
    """
    Topological deduplication using graph structure.
    Entities with similar connections (neighbors) are likely duplicates.
    """

    def __init__(self, jaccard_threshold: float = None):
        """
        Initialize Topological deduplicator.

        Args:
            jaccard_threshold: Threshold for Jaccard similarity
        """
        self.threshold = jaccard_threshold or Config.TOPOLOGICAL_JACCARD_THRESHOLD

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform topological deduplication on the knowledge graph.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        logger.info("Running Topological (Graph-based) deduplication")

        nodes = list(graph.nodes())
        n = len(nodes)

        logger.info(f"Processing {n} entities")

        # Build similarity matrix based on neighbor similarity
        similarity_scores = {}

        # Only compare nodes of the same type
        nodes_by_type = self._group_nodes_by_type(graph)

        for entity_type, type_nodes in nodes_by_type.items():
            logger.debug(f"Processing {len(type_nodes)} entities of type '{entity_type}'")

            # Compare all pairs within the same type
            for node1, node2 in combinations(type_nodes, 2):
                similarity = self._compute_neighbor_similarity(node1, node2, graph)

                if similarity >= self.threshold:
                    similarity_scores[(node1, node2)] = similarity

        logger.info(f"Found {len(similarity_scores)} similar pairs")

        # Build similarity graph
        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(nodes)

        for (node1, node2), score in similarity_scores.items():
            similarity_graph.add_edge(node1, node2, weight=score)

        # Extract connected components as clusters
        clusters = []
        for component in nx.connected_components(similarity_graph):
            if len(component) > 1:
                clusters.append(set(component))

        logger.info(f"Found {len(clusters)} duplicate clusters using topological analysis")

        return clusters

    def _group_nodes_by_type(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Group nodes by entity type.

        Args:
            graph: Knowledge graph

        Returns:
            Dictionary mapping type -> list of node IDs
        """
        nodes_by_type = {}

        for node, attrs in graph.nodes(data=True):
            entity_type = attrs.get('type', 'Unknown')

            if entity_type not in nodes_by_type:
                nodes_by_type[entity_type] = []

            nodes_by_type[entity_type].append(node)

        return nodes_by_type

    def _compute_neighbor_similarity(
        self,
        node1: str,
        node2: str,
        graph: nx.DiGraph
    ) -> float:
        """
        Compute Jaccard similarity between neighbors of two nodes.

        Args:
            node1: First node ID
            node2: Second node ID
            graph: Knowledge graph

        Returns:
            Jaccard similarity score [0, 1]
        """
        # Get neighbors (both predecessors and successors)
        neighbors1 = set(graph.predecessors(node1)).union(set(graph.successors(node1)))
        neighbors2 = set(graph.predecessors(node2)).union(set(graph.successors(node2)))

        # Compute Jaccard similarity
        if len(neighbors1) == 0 and len(neighbors2) == 0:
            # Both nodes are isolated - check attribute similarity as fallback
            return self._compute_attribute_similarity(
                graph.nodes[node1],
                graph.nodes[node2]
            )

        intersection = neighbors1.intersection(neighbors2)
        union = neighbors1.union(neighbors2)

        if len(union) == 0:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Combine with attribute similarity
        attr_sim = self._compute_attribute_similarity(
            graph.nodes[node1],
            graph.nodes[node2]
        )

        # Weighted combination: 70% topological, 30% attribute
        combined = 0.7 * jaccard + 0.3 * attr_sim

        return combined

    def _compute_attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """
        Compute simple attribute similarity as fallback.

        Args:
            attrs1: First entity attributes
            attrs2: Second entity attributes

        Returns:
            Similarity score [0, 1]
        """
        # Compare name if available
        name1 = str(attrs1.get('name', '')).lower()
        name2 = str(attrs2.get('name', '')).lower()

        if name1 and name2:
            # Simple token overlap
            tokens1 = set(name1.split())
            tokens2 = set(name2.split())

            if len(tokens1.union(tokens2)) > 0:
                return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

        return 0.0

    def deduplicate_with_advanced_metrics(
        self,
        graph: nx.DiGraph
    ) -> Tuple[List[Set[str]], Dict[str, Any]]:
        """
        Perform deduplication with additional graph metrics.

        Args:
            graph: Input knowledge graph

        Returns:
            Tuple of (clusters, metrics_dict)
        """
        logger.info("Running advanced topological deduplication")

        nodes = list(graph.nodes())

        # Compute additional graph metrics
        metrics = {
            'betweenness_centrality': nx.betweenness_centrality(graph),
            'closeness_centrality': nx.closeness_centrality(graph),
            'pagerank': nx.pagerank(graph),
        }

        # Group by type
        nodes_by_type = self._group_nodes_by_type(graph)

        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(nodes)

        for entity_type, type_nodes in nodes_by_type.items():
            for node1, node2 in combinations(type_nodes, 2):
                # Combine multiple similarity measures
                neighbor_sim = self._compute_neighbor_similarity(node1, node2, graph)

                # Centrality similarity
                centrality_sim = self._compute_centrality_similarity(
                    node1, node2, metrics
                )

                # Combined score
                combined_sim = 0.7 * neighbor_sim + 0.3 * centrality_sim

                if combined_sim >= self.threshold:
                    similarity_graph.add_edge(node1, node2, weight=combined_sim)

        # Extract clusters
        clusters = []
        for component in nx.connected_components(similarity_graph):
            if len(component) > 1:
                clusters.append(set(component))

        logger.info(f"Advanced method found {len(clusters)} clusters")

        return clusters, metrics

    def _compute_centrality_similarity(
        self,
        node1: str,
        node2: str,
        metrics: Dict[str, Dict]
    ) -> float:
        """
        Compute similarity based on centrality measures.

        Args:
            node1: First node ID
            node2: Second node ID
            metrics: Dictionary of centrality metrics

        Returns:
            Similarity score [0, 1]
        """
        scores = []

        for metric_name, metric_values in metrics.items():
            val1 = metric_values.get(node1, 0)
            val2 = metric_values.get(node2, 0)

            # Normalize difference to similarity
            max_val = max(val1, val2)
            if max_val > 0:
                sim = 1 - abs(val1 - val2) / max_val
                scores.append(sim)

        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0

    def get_methodology_name(self) -> str:
        """Get the name of this methodology."""
        return "Topological (Graph-based)"

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used by this methodology."""
        return {
            "methodology": "Topological",
            "similarity_measure": "Jaccard",
            "threshold": self.threshold,
            "neighbor_weight": 0.7,
            "attribute_weight": 0.3
        }


def create_topological_deduplicator() -> TopologicalDeduplicator:
    """
    Create and return a TopologicalDeduplicator instance.

    Returns:
        TopologicalDeduplicator instance
    """
    return TopologicalDeduplicator()
