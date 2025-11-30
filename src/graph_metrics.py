"""
Graph Metrics Calculator for Dynamic Leiden Resolution Tuning.

Analyzes graph structure to auto-tune Leiden resolution parameter (γ) based on:
- Graph density
- Average degree
- Clustering coefficient
- Entity-to-relationship ratios
"""

import sqlite3
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


class GraphMetricsCalculator:
    """
    Calculates graph metrics to dynamically determine optimal Leiden resolutions.

    Auto-tunes γ (gamma) for 3-level hierarchy:
    - Micro: Fine-grained communities (teams, 5-10 nodes)
    - Meso: Mid-level communities (departments, 50-100 nodes)
    - Macro: Coarse-grained communities (divisions, 1000+ nodes)
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        """
        Initialize the metrics calculator.

        Args:
            db_path: Path to SQLite knowledge graph database
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        logger.info(f"GraphMetricsCalculator initialized with DB: {self.db_path}")

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive graph metrics from SQL database.

        Returns:
            Dictionary with metrics:
            - node_count: Number of entities
            - edge_count: Number of relationships
            - density: Graph density [0, 1]
            - avg_degree: Average number of connections per node
            - clustering_coefficient: Measure of local clustering
            - connected_components: Number of disconnected subgraphs
            - entity_to_relationship_ratio: Ratio of nodes to edges
        """
        logger.info("Calculating graph metrics...")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM entities")
            node_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM relationships")
            edge_count = cursor.fetchone()[0]

            logger.info(f"Graph size: {node_count} nodes, {edge_count} edges")

            # Build NetworkX graph for advanced metrics
            G = self._build_networkx_graph(cursor)

            # Calculate metrics
            metrics = {
                'node_count': node_count,
                'edge_count': edge_count,
                'density': self._calculate_density(node_count, edge_count),
                'avg_degree': self._calculate_avg_degree(node_count, edge_count),
                'clustering_coefficient': nx.average_clustering(G.to_undirected()) if node_count > 0 else 0.0,
                'connected_components': nx.number_weakly_connected_components(G),
                'entity_to_relationship_ratio': node_count / edge_count if edge_count > 0 else float('inf')
            }

            logger.info(f"Metrics calculated: density={metrics['density']:.4f}, "
                       f"avg_degree={metrics['avg_degree']:.2f}, "
                       f"clustering={metrics['clustering_coefficient']:.4f}")

            return metrics

        finally:
            conn.close()

    def _build_networkx_graph(self, cursor: sqlite3.Cursor) -> nx.DiGraph:
        """
        Build NetworkX DiGraph from SQL database.

        Args:
            cursor: Database cursor

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes
        cursor.execute("SELECT unique_entity_id, entity_type, canonical_name FROM entities")
        for entity_id, entity_type, name in cursor.fetchall():
            G.add_node(entity_id, type=entity_type, name=name)

        # Add edges
        cursor.execute("""
            SELECT from_entity_id, to_entity_id, relationship_type
            FROM relationships
        """)
        for from_id, to_id, rel_type in cursor.fetchall():
            G.add_edge(from_id, to_id, relation=rel_type)

        return G

    def _calculate_density(self, node_count: int, edge_count: int) -> float:
        """
        Calculate graph density.

        Density = edges / (nodes × (nodes - 1))
        For directed graphs: range [0, 1]

        Args:
            node_count: Number of nodes
            edge_count: Number of edges

        Returns:
            Density value [0, 1]
        """
        if node_count <= 1:
            return 0.0

        max_possible_edges = node_count * (node_count - 1)
        density = edge_count / max_possible_edges if max_possible_edges > 0 else 0.0

        return density

    def _calculate_avg_degree(self, node_count: int, edge_count: int) -> float:
        """
        Calculate average node degree.

        For directed graphs: avg_degree = total_edges / nodes

        Args:
            node_count: Number of nodes
            edge_count: Number of edges

        Returns:
            Average degree
        """
        if node_count == 0:
            return 0.0

        return edge_count / node_count

    def calculate_optimal_resolutions(
        self,
        metrics: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Auto-tune Leiden resolution parameters based on graph metrics.

        Logic:
        - High Density (>0.1) or High Avg Degree (>20): Use HIGH γ to force fine-grained splits
        - Medium Density (0.01-0.1) or Medium Avg Degree (5-20): Use STANDARD γ
        - Low Density (<0.01) or Low Avg Degree (<5): Use LOW γ to find loose groups

        Args:
            metrics: Pre-computed metrics (optional, will calculate if not provided)

        Returns:
            Dictionary with resolution parameters:
            - micro: γ for fine-grained communities
            - meso: γ for mid-level communities
            - macro: γ for coarse communities
        """
        if metrics is None:
            metrics = self.calculate_all_metrics()

        density = metrics['density']
        avg_degree = metrics['avg_degree']
        clustering = metrics['clustering_coefficient']

        logger.info(f"Auto-tuning resolutions based on: density={density:.4f}, "
                   f"avg_degree={avg_degree:.2f}, clustering={clustering:.4f}")

        # Determine graph complexity category
        if density > 0.1 or avg_degree > 20:
            # DENSE GRAPH: Need high resolution to break apart
            category = "DENSE"
            resolutions = {
                "micro": 2.0,    # Very fine-grained
                "meso": 1.2,     # Moderately fine
                "macro": 0.5     # Still somewhat fine
            }
        elif density > 0.01 or avg_degree > 5:
            # MEDIUM GRAPH: Standard resolution works well
            category = "MEDIUM"
            resolutions = {
                "micro": 1.5,    # Fine-grained
                "meso": 1.0,     # Standard (modularity)
                "macro": 0.3     # Coarse
            }
        else:
            # SPARSE GRAPH: Need low resolution to find meaningful groups
            category = "SPARSE"
            resolutions = {
                "micro": 1.0,    # Moderate
                "meso": 0.7,     # Slightly coarse
                "macro": 0.1     # Very coarse
            }

        # Fine-tune based on clustering coefficient
        # High clustering = tight-knit groups already exist, can use higher γ
        if clustering > 0.5:
            resolutions["micro"] *= 1.2
            resolutions["meso"] *= 1.1
            logger.info("High clustering detected, increasing resolutions by 10-20%")

        logger.info(f"Graph category: {category}")
        logger.info(f"Optimal resolutions: micro={resolutions['micro']:.2f}, "
                   f"meso={resolutions['meso']:.2f}, macro={resolutions['macro']:.2f}")

        return resolutions

    def should_recompute_leiden(self) -> Tuple[bool, str]:
        """
        Determine if Leiden should be recomputed based on graph changes.

        Checks leiden_metadata table and compares current graph state:
        - If >10% change in entity count: recompute
        - If >15% change in relationship count: recompute
        - If never computed before: recompute

        Returns:
            Tuple of (should_recompute: bool, reason: str)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Check if Leiden has been computed before
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='leiden_metadata'")
            if cursor.fetchone()[0] == 0:
                return True, "Leiden never computed (metadata table doesn't exist)"

            cursor.execute("SELECT COUNT(*) FROM leiden_metadata")
            if cursor.fetchone()[0] == 0:
                return True, "Leiden never computed (metadata table empty)"

            # Get last computation state
            cursor.execute("""
                SELECT total_entities, total_relationships, last_computed
                FROM leiden_metadata
                WHERE id = 1
            """)
            row = cursor.fetchone()

            if row is None:
                return True, "No metadata found"

            last_entities, last_relationships, last_computed = row

            # Get current state
            cursor.execute("SELECT COUNT(*) FROM entities")
            current_entities = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM relationships")
            current_relationships = cursor.fetchone()[0]

            # Calculate change percentages
            entity_change_pct = abs(current_entities - last_entities) / last_entities if last_entities > 0 else 1.0
            rel_change_pct = abs(current_relationships - last_relationships) / last_relationships if last_relationships > 0 else 1.0

            logger.info(f"Graph changes since last Leiden: entities={entity_change_pct*100:.1f}%, "
                       f"relationships={rel_change_pct*100:.1f}%")

            # Thresholds
            ENTITY_THRESHOLD = 0.10  # 10%
            RELATIONSHIP_THRESHOLD = 0.15  # 15%

            if entity_change_pct > ENTITY_THRESHOLD:
                return True, f"Entity count changed by {entity_change_pct*100:.1f}% (threshold: 10%)"

            if rel_change_pct > RELATIONSHIP_THRESHOLD:
                return True, f"Relationship count changed by {rel_change_pct*100:.1f}% (threshold: 15%)"

            return False, f"Graph stable (last computed: {last_computed})"

        finally:
            conn.close()

    def get_summary_statistics(self) -> str:
        """
        Get human-readable summary of graph metrics.

        Returns:
            Formatted string with key statistics
        """
        metrics = self.calculate_all_metrics()
        resolutions = self.calculate_optimal_resolutions(metrics)

        summary = f"""
Graph Metrics Summary
=====================
Nodes: {metrics['node_count']}
Edges: {metrics['edge_count']}
Density: {metrics['density']:.6f}
Average Degree: {metrics['avg_degree']:.2f}
Clustering Coefficient: {metrics['clustering_coefficient']:.4f}
Connected Components: {metrics['connected_components']}
Entity/Relationship Ratio: {metrics['entity_to_relationship_ratio']:.2f}

Recommended Leiden Resolutions
===============================
Micro (Teams): γ = {resolutions['micro']:.2f}
Meso (Departments): γ = {resolutions['meso']:.2f}
Macro (Divisions): γ = {resolutions['macro']:.2f}
"""
        return summary


def calculate_metrics(db_path: str = "knowledge_graph.db") -> Dict[str, Any]:
    """
    Convenience function to calculate graph metrics.

    Args:
        db_path: Path to database

    Returns:
        Dictionary of metrics
    """
    calculator = GraphMetricsCalculator(db_path)
    return calculator.calculate_all_metrics()


def get_optimal_resolutions(db_path: str = "knowledge_graph.db") -> Dict[str, float]:
    """
    Convenience function to get optimal Leiden resolutions.

    Args:
        db_path: Path to database

    Returns:
        Dictionary of resolutions for micro/meso/macro levels
    """
    calculator = GraphMetricsCalculator(db_path)
    return calculator.calculate_optimal_resolutions()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Calculate and display metrics
    calculator = GraphMetricsCalculator()
    print(calculator.get_summary_statistics())

    # Check if recomputation needed
    should_recompute, reason = calculator.should_recompute_leiden()
    print(f"\nShould recompute Leiden? {should_recompute}")
    print(f"Reason: {reason}")
