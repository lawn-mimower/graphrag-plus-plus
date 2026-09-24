"""
Knowledge Graph Builder using NetworkX.
Constructs non-deduplicated and deduplicated knowledge graphs from extracted entities.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import networkx as nx
from collections import Counter

from src.config import Config

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and manages knowledge graphs using NetworkX.
    Supports both non-deduplicated and deduplicated graph construction.
    """

    # Node/edge metadata set by the builder. Extracted attributes with the same
    # key are kept under an "attr_" prefix instead of overwriting them.
    RESERVED_NODE_KEYS = {
        'original_id', 'type', 'source_doc', 'source_docs',
        'page_numbers', 'cluster_size', 'merge_reasoning'
    }
    RESERVED_EDGE_KEYS = {'relationship_type', 'source_doc', 'source_docs'}

    def __init__(self):
        """Initialize the Knowledge Graph Builder."""
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.entity_counter = 0  # For generating unique internal node IDs

    def add_entities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        source_doc: str,
        page_numbers: List[int] = None
    ):
        """
        Add entities and relationships to the knowledge graph.
        Creates a new node for EVERY entity mention (non-deduplicated).

        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            source_doc: Source document name
            page_numbers: List of page numbers where entities appear
        """
        logger.debug(f"Adding {len(entities)} entities from {source_doc}")

        # Map original entity IDs to internal node IDs
        id_mapping = {}

        # Add nodes
        for entity in entities:
            original_id = entity.get('id', f'ENTITY_{self.entity_counter}')
            self.entity_counter += 1

            # Create unique internal node ID
            internal_id = f"node_{self.entity_counter}_{original_id}"
            id_mapping[original_id] = internal_id

            # Add node with all attributes
            attributes = dict(entity.get('attributes') or {})

            # Pages reported for this entity take precedence over document-level pages
            entity_pages = attributes.pop('page_numbers', None)
            if isinstance(entity_pages, int):
                entity_pages = [entity_pages]
            if not isinstance(entity_pages, list) or not entity_pages:
                entity_pages = page_numbers or []

            node_attrs = {
                (f"attr_{k}" if k in self.RESERVED_NODE_KEYS else k): v
                for k, v in attributes.items()
            }
            node_attrs.update({
                'original_id': original_id,
                'type': entity.get('type', 'Unknown'),
                'source_doc': source_doc,
                'page_numbers': entity_pages,
            })

            self.graph.add_node(internal_id, **node_attrs)

        # Add edges
        for rel in relationships:
            from_id = rel.get('from_id')
            to_id = rel.get('to_id')
            rel_type = rel.get('type', 'RELATED_TO')

            # Map to internal IDs
            from_internal = id_mapping.get(from_id)
            to_internal = id_mapping.get(to_id)

            if from_internal and to_internal:
                edge_attrs = {
                    (f"attr_{k}" if k in self.RESERVED_EDGE_KEYS else k): v
                    for k, v in (rel.get('attributes') or {}).items()
                }
                edge_attrs.update({
                    'relationship_type': rel_type,
                    'source_doc': source_doc,
                })
                self.graph.add_edge(from_internal, to_internal, **edge_attrs)

        logger.info(
            f"Graph now has {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

    def get_graph(self) -> nx.DiGraph:
        """
        Get the knowledge graph.

        Returns:
            NetworkX DiGraph
        """
        return self.graph

    def save_graph(self, save_path: Path, format: str = 'gpickle'):
        """
        Save the knowledge graph to disk.

        Args:
            save_path: Path to save file
            format: Format ('gpickle', 'graphml', 'gexf')
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'gpickle':
            with open(save_path, 'wb') as f:
                pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == 'graphml':
            # Convert all attributes to strings for GraphML compatibility
            graph_copy = self._prepare_for_graphml()
            nx.write_graphml(graph_copy, save_path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, save_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved graph to {save_path} (format: {format})")

    def load_graph(self, load_path: Path, format: str = 'gpickle'):
        """
        Load a knowledge graph from disk.

        Args:
            load_path: Path to load from
            format: Format ('gpickle', 'graphml', 'gexf')
        """
        if format == 'gpickle':
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)
        elif format == 'graphml':
            self.graph = nx.read_graphml(load_path)
        elif format == 'gexf':
            self.graph = nx.read_gexf(load_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded graph from {load_path}")

    def _prepare_for_graphml(self) -> nx.DiGraph:
        """
        Prepare graph for GraphML export by converting all attributes to strings.

        Returns:
            Graph copy with string attributes
        """
        graph_copy = nx.DiGraph()

        for node, attrs in self.graph.nodes(data=True):
            str_attrs = {k: str(v) for k, v in attrs.items()}
            graph_copy.add_node(node, **str_attrs)

        for u, v, attrs in self.graph.edges(data=True):
            str_attrs = {k: str(v) for k, v in attrs.items()}
            graph_copy.add_edge(u, v, **str_attrs)

        return graph_copy

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'entity_types': self._count_entity_types(),
            'relationship_types': self._count_relationship_types(),
            'source_documents': self._count_source_documents(),
        }

        # Compute degree statistics
        if self.graph.number_of_nodes() > 0:
            degrees = [d for _, d in self.graph.degree()]
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
        else:
            stats['avg_degree'] = 0
            stats['max_degree'] = 0

        return stats

    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type."""
        types = [attrs.get('type', 'Unknown')
                for _, attrs in self.graph.nodes(data=True)]
        return dict(Counter(types))

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count relationships by type."""
        types = [attrs.get('relationship_type', 'Unknown')
                for _, _, attrs in self.graph.edges(data=True)]
        return dict(Counter(types))

    def _count_source_documents(self) -> Dict[str, int]:
        """Count entities by source document."""
        docs = [attrs.get('source_doc', 'Unknown')
               for _, attrs in self.graph.nodes(data=True)]
        return dict(Counter(docs))

    def get_entities_by_type(self, entity_type: str) -> List[Tuple[str, Dict]]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Entity type to filter by

        Returns:
            List of (node_id, attributes) tuples
        """
        entities = [
            (node, attrs)
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get('type') == entity_type
        ]
        return entities

    def get_entities_by_document(self, doc_name: str) -> List[Tuple[str, Dict]]:
        """
        Get all entities from a specific document.

        Args:
            doc_name: Document name

        Returns:
            List of (node_id, attributes) tuples
        """
        entities = [
            (node, attrs)
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get('source_doc') == doc_name
        ]
        return entities

    def create_deduplicated_graph(
        self,
        duplicate_clusters: List[Set[str]],
        cluster_reasoning: Dict[str, Any] = None
    ) -> nx.DiGraph:
        """
        Create a deduplicated version of the graph based on duplicate clusters.

        Args:
            duplicate_clusters: List of sets, where each set contains node IDs
                               that should be merged into a single entity
            cluster_reasoning: Optional dictionary mapping cluster info to LLM reasoning
                             (used by LLM-based methodologies for explainability)

        Returns:
            New deduplicated graph
        """
        logger.info(f"Creating deduplicated graph from {len(duplicate_clusters)} clusters")

        dedup_graph = nx.DiGraph()

        # Map old node IDs to new representative IDs
        node_mapping = {}
        for cluster in duplicate_clusters:
            if not cluster:
                continue

            # Choose representative node (first one, or most complete one)
            cluster_list = list(cluster)
            representative = self._choose_representative(cluster_list)

            # Map all nodes in cluster to representative
            for node in cluster_list:
                node_mapping[node] = representative

        # Add all nodes (unmapped nodes map to themselves)
        for node, attrs in self.graph.nodes(data=True):
            rep_node = node_mapping.get(node, node)

            if rep_node not in dedup_graph.nodes:
                # Merge attributes from all nodes in cluster
                merged_attrs = self._merge_node_attributes(
                    node_mapping, rep_node, node, attrs, cluster_reasoning
                )
                dedup_graph.add_node(rep_node, **merged_attrs)

        # Add edges (redirect to representative nodes)
        for u, v, attrs in self.graph.edges(data=True):
            rep_u = node_mapping.get(u, u)
            rep_v = node_mapping.get(v, v)

            # Avoid self-loops
            if rep_u != rep_v:
                if dedup_graph.has_edge(rep_u, rep_v):
                    # Merge edge attributes if edge already exists
                    existing_attrs = dedup_graph[rep_u][rep_v]
                    merged_attrs = self._merge_edge_attributes(existing_attrs, attrs)
                    dedup_graph[rep_u][rep_v].update(merged_attrs)
                else:
                    dedup_graph.add_edge(rep_u, rep_v, **attrs)

        logger.info(
            f"Deduplicated graph: {dedup_graph.number_of_nodes()} nodes, "
            f"{dedup_graph.number_of_edges()} edges "
            f"(reduced from {self.graph.number_of_nodes()} nodes)"
        )

        return dedup_graph

    def _choose_representative(self, cluster_nodes: List[str]) -> str:
        """
        Choose the representative node from a cluster.
        Prefers nodes with more attributes.

        Args:
            cluster_nodes: List of node IDs in the cluster

        Returns:
            Representative node ID
        """
        # Choose node with most attributes
        best_node = cluster_nodes[0]
        max_attrs = len(self.graph.nodes[best_node])

        for node in cluster_nodes[1:]:
            num_attrs = len(self.graph.nodes[node])
            if num_attrs > max_attrs:
                max_attrs = num_attrs
                best_node = node

        return best_node

    def _merge_node_attributes(
        self,
        node_mapping: Dict,
        rep_node: str,
        current_node: str,
        current_attrs: Dict,
        cluster_reasoning: Dict[str, Any] = None
    ) -> Dict:
        """
        Merge attributes from multiple nodes in a cluster.

        Args:
            node_mapping: Mapping of nodes to representatives
            rep_node: Representative node ID
            current_node: Current node being processed
            current_attrs: Attributes of current node
            cluster_reasoning: Optional reasoning data from LLM deduplication

        Returns:
            Merged attributes dictionary
        """
        # Find all nodes that map to this representative
        cluster_nodes = [n for n, r in node_mapping.items() if r == rep_node]

        # Collect attributes from all nodes in cluster
        merged = current_attrs.copy()

        # Merge source documents
        source_docs = set([current_attrs.get('source_doc', '')])
        page_nums = set(current_attrs.get('page_numbers', []))

        for node in cluster_nodes:
            if node != current_node and node in self.graph.nodes:
                attrs = self.graph.nodes[node]
                source_docs.add(attrs.get('source_doc', ''))
                page_nums.update(attrs.get('page_numbers', []))

        merged['source_docs'] = list(source_docs)
        merged['page_numbers'] = sorted(list(page_nums))
        # Nodes outside any duplicate cluster are not in node_mapping: size 1
        merged['cluster_size'] = max(len(cluster_nodes), 1)

        # Add merge reasoning if available (from LLM methodologies)
        if cluster_reasoning and len(cluster_nodes) > 1:
            # Find reasoning for this cluster
            cluster_set = set(cluster_nodes)
            for duplicate_info in cluster_reasoning.get('duplicates', []):
                entity_ids = set(duplicate_info.get('entities', []))
                # Check if this is the cluster we're looking for
                if entity_ids == cluster_set:
                    merged['merge_reasoning'] = {
                        'analysis': duplicate_info.get('analysis', ''),
                        'decision': duplicate_info.get('deduplication_decision', ''),
                        'merged_entities': list(cluster_set),
                        'cluster_id': duplicate_info.get('cluster_id')
                    }
                    break

        return merged

    def _merge_edge_attributes(self, attrs1: Dict, attrs2: Dict) -> Dict:
        """
        Merge attributes from two edges.

        Args:
            attrs1: First edge attributes
            attrs2: Second edge attributes

        Returns:
            Merged attributes
        """
        merged = attrs1.copy()

        # Combine source documents
        docs1 = set([attrs1.get('source_doc', '')])
        docs2 = set([attrs2.get('source_doc', '')])
        merged['source_docs'] = list(docs1.union(docs2))

        return merged

    def print_statistics(self):
        """Print graph statistics to console."""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 80)
        print(f"Nodes: {stats['num_nodes']:,}")
        print(f"Edges: {stats['num_edges']:,}")
        print(f"Connected Components: {stats['num_connected_components']}")
        print(f"Average Degree: {stats['avg_degree']:.2f}")
        print(f"Max Degree: {stats['max_degree']}")

        print("\nEntity Types:")
        for entity_type, count in sorted(stats['entity_types'].items()):
            print(f"  {entity_type}: {count:,}")

        print("\nRelationship Types:")
        for rel_type, count in sorted(stats['relationship_types'].items()):
            print(f"  {rel_type}: {count:,}")

        print("\nSource Documents:")
        for doc, count in sorted(stats['source_documents'].items()):
            print(f"  {doc}: {count:,} entities")

        print("=" * 80 + "\n")


# Utility functions
def create_knowledge_graph_builder() -> KnowledgeGraphBuilder:
    """
    Create and return a KnowledgeGraphBuilder instance.

    Returns:
        KnowledgeGraphBuilder instance
    """
    return KnowledgeGraphBuilder()


def merge_knowledge_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    """
    Merge multiple knowledge graphs into one.

    Args:
        graphs: List of NetworkX graphs

    Returns:
        Merged graph
    """
    merged = nx.DiGraph()

    for graph in graphs:
        merged = nx.compose(merged, graph)

    return merged
