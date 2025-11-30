"""
Leiden Community Detection Builder.

Runs Leiden algorithm at multiple hierarchical levels (micro/meso/macro)
and stores results in SQL database for use by the query orchestrator.
"""

import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import networkx as nx

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("leidenalg not installed. Run: pip install leidenalg python-igraph")

from src.graph_metrics import GraphMetricsCalculator

logger = logging.getLogger(__name__)


class LeidenCommunityBuilder:
    """
    Builds hierarchical Leiden communities from knowledge graph.

    3-Level Hierarchy:
    - Micro: Fine-grained communities (teams, ~5-10 nodes)
    - Meso: Mid-level communities (departments, ~50-100 nodes)
    - Macro: Coarse communities (divisions, ~1000+ nodes)

    Resolution parameters (γ) are auto-tuned based on graph density.
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        """
        Initialize the Leiden builder.

        Args:
            db_path: Path to SQLite knowledge graph database

        Raises:
            ImportError: If leidenalg is not installed
            FileNotFoundError: If database doesn't exist
        """
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "leidenalg is required for community detection. "
                "Install with: pip install leidenalg python-igraph"
            )

        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.metrics_calculator = GraphMetricsCalculator(str(self.db_path))

        logger.info(f"LeidenCommunityBuilder initialized with DB: {self.db_path}")

    def build_communities(
        self,
        resolutions: Dict[str, float] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Build Leiden communities at all 3 hierarchical levels.

        Args:
            resolutions: Optional dict with 'micro', 'meso', 'macro' γ values
                        If None, will auto-calculate based on graph metrics
            force: If True, rebuild even if communities already exist

        Returns:
            Dictionary with statistics:
            - levels: Dict of level -> community count
            - modularity: Dict of level -> modularity score
            - total_entities: Number of entities processed
            - computation_time: Time taken in seconds
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting Leiden Community Detection")
        logger.info("=" * 80)

        # Check if recomputation needed
        if not force:
            should_recompute, reason = self.metrics_calculator.should_recompute_leiden()
            if not should_recompute:
                logger.info(f"Skipping Leiden computation: {reason}")
                return self._get_existing_statistics()

        # Step 1: Calculate metrics and optimal resolutions
        logger.info("\nStep 1: Calculating graph metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics()

        if resolutions is None:
            resolutions = self.metrics_calculator.calculate_optimal_resolutions(metrics)
            logger.info(f"Auto-tuned resolutions: {resolutions}")

        # Step 2: Build NetworkX graph from SQL
        logger.info("\nStep 2: Loading graph from database...")
        nx_graph = self._load_graph_from_sql()

        # Step 3: Convert to igraph (required for leidenalg)
        logger.info("\nStep 3: Converting to igraph...")
        ig_graph, node_mapping = self._convert_to_igraph(nx_graph)

        # Step 4: Run Leiden at each level
        logger.info("\nStep 4: Running Leiden algorithm...")
        community_results = self._run_leiden_hierarchy(ig_graph, resolutions, node_mapping)

        # Step 5: Store results in database
        logger.info("\nStep 5: Storing communities in database...")
        self._store_communities(community_results, resolutions, metrics)

        # Step 6: Calculate statistics
        computation_time = time.time() - start_time

        stats = {
            'levels': {
                level: len(set(communities.values()))
                for level, communities in community_results.items()
            },
            'modularity': {
                level: self._calculate_modularity(ig_graph, list(communities.values()), resolutions[level])
                for level, communities in community_results.items()
            },
            'total_entities': len(nx_graph.nodes()),
            'computation_time': computation_time
        }

        logger.info("\n" + "=" * 80)
        logger.info("Leiden Community Detection Complete")
        logger.info("=" * 80)
        logger.info(f"Total entities: {stats['total_entities']}")
        logger.info(f"Computation time: {computation_time:.2f} seconds")
        logger.info("\nCommunities per level:")
        for level in ['micro', 'meso', 'macro']:
            logger.info(f"  {level.capitalize()}: {stats['levels'][level]} communities "
                       f"(modularity: {stats['modularity'][level]:.4f})")

        return stats

    def _load_graph_from_sql(self) -> nx.DiGraph:
        """
        Load graph from SQL database into NetworkX.

        Returns:
            NetworkX directed graph with all entities and relationships
        """
        G = nx.DiGraph()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Add nodes
            cursor.execute("""
                SELECT unique_entity_id, entity_type, canonical_name
                FROM entities
            """)
            for entity_id, entity_type, name in cursor.fetchall():
                G.add_node(entity_id, type=entity_type, name=name)

            # Add edges
            cursor.execute("""
                SELECT from_entity_id, to_entity_id, relationship_type
                FROM relationships
            """)
            for from_id, to_id, rel_type in cursor.fetchall():
                if from_id in G and to_id in G:  # Ensure both nodes exist
                    G.add_edge(from_id, to_id, relation=rel_type)

            logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            return G

        finally:
            conn.close()

    def _convert_to_igraph(self, nx_graph: nx.DiGraph) -> Tuple[ig.Graph, Dict]:
        """
        Convert NetworkX graph to igraph (required for leidenalg).

        Args:
            nx_graph: NetworkX directed graph

        Returns:
            Tuple of (igraph.Graph, node_mapping)
            node_mapping: Dict mapping integer index -> entity_id
        """
        # Treat as undirected for community detection
        nx_undirected = nx_graph.to_undirected()

        # Create mapping: entity_id -> integer index
        nodes = list(nx_undirected.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        # Build edge list with integer indices
        edges = [
            (node_to_idx[u], node_to_idx[v])
            for u, v in nx_undirected.edges()
        ]

        # Create igraph
        ig_graph = ig.Graph(len(nodes))
        ig_graph.add_edges(edges)

        # Add node attributes
        ig_graph.vs['entity_id'] = nodes
        ig_graph.vs['name'] = [nx_graph.nodes[node].get('name', node) for node in nodes]
        ig_graph.vs['type'] = [nx_graph.nodes[node].get('type', 'Unknown') for node in nodes]

        logger.info(f"Converted to igraph: {ig_graph.vcount()} vertices, {ig_graph.ecount()} edges")

        return ig_graph, idx_to_node

    def _run_leiden_hierarchy(
        self,
        graph: ig.Graph,
        resolutions: Dict[str, float],
        node_mapping: Dict[int, str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Run Leiden algorithm at multiple resolution levels.

        Args:
            graph: igraph.Graph
            resolutions: Dict with 'micro', 'meso', 'macro' γ values
            node_mapping: Dict mapping index -> entity_id

        Returns:
            Dict of level -> {entity_id: community_id}
        """
        results = {}

        for level_name in ['micro', 'meso', 'macro']:
            gamma = resolutions[level_name]

            logger.info(f"Running Leiden for {level_name} level (γ={gamma:.2f})...")

            # Run Leiden with RBConfigurationVertexPartition (supports resolution parameter)
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=gamma,
                n_iterations=-1  # Run until convergence
            )

            # Map back to entity IDs
            entity_communities = {}
            for idx, community_id in enumerate(partition.membership):
                entity_id = node_mapping[idx]
                entity_communities[entity_id] = community_id

            num_communities = len(set(partition.membership))
            modularity = partition.modularity

            logger.info(f"  {level_name.capitalize()}: {num_communities} communities, "
                       f"modularity={modularity:.4f}")

            results[level_name] = entity_communities

        return results

    def _calculate_modularity(
        self,
        graph: ig.Graph,
        membership: List[int],
        resolution: float
    ) -> float:
        """
        Calculate modularity for a given partition.

        Args:
            graph: igraph.Graph
            membership: List of community IDs
            resolution: Resolution parameter used

        Returns:
            Modularity score
        """
        try:
            partition = leidenalg.RBConfigurationVertexPartition(
                graph,
                membership,
                resolution_parameter=resolution
            )
            return partition.modularity
        except Exception as e:
            logger.warning(f"Failed to calculate modularity: {e}")
            return 0.0

    def _store_communities(
        self,
        community_results: Dict[str, Dict[str, int]],
        resolutions: Dict[str, float],
        metrics: Dict[str, Any]
    ):
        """
        Store community detection results in database.

        Args:
            community_results: Dict of level -> {entity_id: community_id}
            resolutions: Dict of level -> γ value
            metrics: Graph metrics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Clear existing communities
            cursor.execute("DELETE FROM leiden_communities")

            # Insert new communities
            for level_name, entity_communities in community_results.items():
                resolution = resolutions[level_name]

                for entity_id, community_id in entity_communities.items():
                    cursor.execute("""
                        INSERT INTO leiden_communities
                        (entity_id, level, resolution, community_id)
                        VALUES (?, ?, ?, ?)
                    """, (entity_id, level_name, resolution, community_id))

            # Update metadata
            cursor.execute("DELETE FROM leiden_metadata WHERE id = 1")
            cursor.execute("""
                INSERT INTO leiden_metadata
                (id, total_entities, total_relationships, graph_density, avg_degree,
                 clustering_coefficient, connected_components, resolutions, computation_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                1,
                metrics['node_count'],
                metrics['edge_count'],
                metrics['density'],
                metrics['avg_degree'],
                metrics['clustering_coefficient'],
                metrics['connected_components'],
                json.dumps(resolutions),
                time.time()
            ))

            conn.commit()
            logger.info("Communities stored successfully")

        finally:
            conn.close()

    def _get_existing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from existing Leiden computation.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get community counts per level
            cursor.execute("""
                SELECT level, COUNT(DISTINCT community_id) as count
                FROM leiden_communities
                GROUP BY level
            """)
            levels = {row[0]: row[1] for row in cursor.fetchall()}

            # Get metadata
            cursor.execute("""
                SELECT total_entities, computation_time_seconds
                FROM leiden_metadata
                WHERE id = 1
            """)
            row = cursor.fetchone()

            return {
                'levels': levels,
                'total_entities': row[0] if row else 0,
                'computation_time': row[1] if row else 0,
                'note': 'Using existing Leiden communities (not recomputed)'
            }

        finally:
            conn.close()

    def get_entity_community(
        self,
        entity_id: str,
        level: str = 'meso'
    ) -> int:
        """
        Get community ID for a specific entity at a given level.

        Args:
            entity_id: Entity unique ID
            level: 'micro', 'meso', or 'macro'

        Returns:
            Community ID (integer)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT community_id
                FROM leiden_communities
                WHERE entity_id = ? AND level = ?
            """, (entity_id, level))

            row = cursor.fetchone()
            return row[0] if row else None

        finally:
            conn.close()

    def get_community_members(
        self,
        community_id: int,
        level: str = 'meso'
    ) -> List[str]:
        """
        Get all entity IDs in a specific community.

        Args:
            community_id: Community ID
            level: 'micro', 'meso', or 'macro'

        Returns:
            List of entity IDs
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entity_id
                FROM leiden_communities
                WHERE community_id = ? AND level = ?
            """, (community_id, level))

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()


def build_leiden_communities(
    db_path: str = "knowledge_graph.db",
    force: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to build Leiden communities.

    Args:
        db_path: Path to database
        force: Force rebuild even if communities exist

    Returns:
        Dictionary with statistics
    """
    builder = LeidenCommunityBuilder(db_path)
    return builder.build_communities(force=force)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Build communities
    stats = build_leiden_communities(force=True)

    print("\n" + "=" * 80)
    print("LEIDEN COMMUNITY DETECTION - SUMMARY")
    print("=" * 80)
    print(json.dumps(stats, indent=2))
