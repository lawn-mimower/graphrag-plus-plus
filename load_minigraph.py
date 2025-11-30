#!/usr/bin/env python
"""
Utility script to load and inspect saved mini-graphs.
"""

import pickle
import json
import sys
from pathlib import Path
import networkx as nx


def load_graph(graph_path: str):
    """Load a mini-graph from pickle file."""
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G


def print_graph_info(G: nx.DiGraph):
    """Print information about the graph."""
    print("\n" + "=" * 80)
    print("MINI-GRAPH INFORMATION")
    print("=" * 80)

    # Metadata
    print(f"\nQuery: {G.graph.get('query', 'N/A')}")
    print(f"Timestamp: {G.graph.get('timestamp', 'N/A')}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Citations
    citations = G.graph.get('citations', {})
    if citations:
        print(f"\nCitations ({len(citations)} documents):")
        for doc, pages in citations.items():
            print(f"  - {doc}: pages {pages}")

    # Nodes
    print(f"\nNodes:")
    for node_id, attrs in G.nodes(data=True):
        print(f"  - {attrs.get('name', node_id)} ({attrs.get('type', 'Unknown')})")
        if attrs.get('attributes'):
            for key, val in list(attrs['attributes'].items())[:3]:
                print(f"    • {key}: {val}")

    # Edges
    print(f"\nEdges:")
    for u, v, attrs in G.edges(data=True):
        source_name = G.nodes[u].get('name', u)
        target_name = G.nodes[v].get('name', v)
        relation = attrs.get('relation', 'CONNECTED')
        print(f"  - {source_name} --[{relation}]--> {target_name}")

    print("=" * 80 + "\n")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python load_minigraph.py <path_to_graph_file>")
        print("\nAvailable graphs:")

        mini_graphs_dir = Path("./mini_graphs")
        if mini_graphs_dir.exists():
            graphs = sorted(mini_graphs_dir.glob("*.gpickle"))
            for graph_file in graphs:
                print(f"  - {graph_file}")
        else:
            print("  No mini_graphs directory found")

        sys.exit(1)

    graph_path = sys.argv[1]

    print(f"\nLoading graph from: {graph_path}")
    G = load_graph(graph_path)

    print_graph_info(G)

    # Check for JSON file
    json_path = Path(graph_path).with_suffix('.json')
    if json_path.exists():
        print(f"JSON version available at: {json_path}")


if __name__ == "__main__":
    main()
