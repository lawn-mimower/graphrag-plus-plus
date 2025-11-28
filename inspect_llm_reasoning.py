#!/usr/bin/env python3
"""
Interactive tool to inspect LLM deduplication reasoning.
Shows why the LLM Full Context methodology merged specific entities.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any

def load_reasoning(reasoning_file: Path) -> Dict[str, Any]:
    """Load reasoning JSON file."""
    with open(reasoning_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_graph(graph_file: Path):
    """Load deduplicated graph."""
    with open(graph_file, 'rb') as f:
        return pickle.load(f)

def print_reasoning_summary(reasoning: Dict[str, Any]):
    """Print summary of reasoning."""
    print("\n" + "=" * 80)
    print("LLM DEDUPLICATION REASONING SUMMARY")
    print("=" * 80)

    summary = reasoning.get('summary', {})
    print(f"\nTotal Entities: {summary.get('total_entities', 'N/A')}")
    print(f"Duplicate Clusters Found: {summary.get('total_clusters', 'N/A')}")
    print(f"Total Entities Merged: {summary.get('total_duplicates_found', 'N/A')}")
    print("\n" + "=" * 80)

def print_cluster_details(reasoning: Dict[str, Any], cluster_id: int = None):
    """Print detailed reasoning for clusters."""
    duplicates = reasoning.get('duplicates', [])

    if not duplicates:
        print("\nNo duplicate clusters found.")
        return

    if cluster_id is not None:
        # Print specific cluster
        for dup in duplicates:
            if dup.get('cluster_id') == cluster_id:
                print_single_cluster(dup)
                return
        print(f"\nCluster {cluster_id} not found.")
    else:
        # Print all clusters
        print(f"\nShowing {len(duplicates)} duplicate clusters:\n")
        for dup in duplicates:
            print_single_cluster(dup)
            print("\n" + "-" * 80 + "\n")

def print_single_cluster(duplicate_info: Dict[str, Any]):
    """Print details for a single cluster."""
    cluster_id = duplicate_info.get('cluster_id', 'N/A')
    entities = duplicate_info.get('entities', [])
    analysis = duplicate_info.get('analysis', 'No analysis provided')
    decision = duplicate_info.get('deduplication_decision', 'No decision recorded')

    print(f"Cluster ID: {cluster_id}")
    print(f"Entities Merged: {len(entities)}")
    print(f"Entity IDs: {', '.join(entities[:5])}{' ...' if len(entities) > 5 else ''}")
    print(f"\nAnalysis:")
    print(f"  {analysis}")
    print(f"\nDecision:")
    print(f"  {decision}")

def inspect_graph_reasoning(graph_file: Path):
    """Inspect reasoning stored in graph nodes."""
    graph = load_graph(graph_file)

    print("\n" + "=" * 80)
    print("REASONING EMBEDDED IN GRAPH NODES")
    print("=" * 80)

    nodes_with_reasoning = []
    for node, attrs in graph.nodes(data=True):
        if 'merge_reasoning' in attrs:
            nodes_with_reasoning.append((node, attrs))

    if not nodes_with_reasoning:
        print("\nNo nodes with merge reasoning found.")
        print("(Only merged clusters have reasoning attached)")
        return

    print(f"\nFound {len(nodes_with_reasoning)} merged nodes with reasoning:\n")

    for node, attrs in nodes_with_reasoning[:10]:  # Show first 10
        reasoning = attrs['merge_reasoning']
        print(f"Node: {node}")
        print(f"  Cluster Size: {attrs.get('cluster_size', 'N/A')}")
        print(f"  Merged Entities: {', '.join(reasoning.get('merged_entities', [])[:3])}")
        print(f"  Analysis: {reasoning.get('analysis', '')[:100]}...")
        print()

    if len(nodes_with_reasoning) > 10:
        print(f"... and {len(nodes_with_reasoning) - 10} more nodes with reasoning")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Inspect LLM deduplication reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary of reasoning
  python inspect_llm_reasoning.py outputs/reasoning/llm_full_context_reasoning_*.json

  # View specific cluster
  python inspect_llm_reasoning.py reasoning.json --cluster 5

  # View reasoning in graph
  python inspect_llm_reasoning.py --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        type=str,
        help='Path to reasoning JSON file or graph file'
    )

    parser.add_argument(
        '--cluster',
        type=int,
        help='Show details for specific cluster ID'
    )

    parser.add_argument(
        '--graph',
        type=str,
        help='Path to deduplicated graph file (.gpickle)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all clusters (not just summary)'
    )

    args = parser.parse_args()

    # Inspect graph if --graph flag is used
    if args.graph:
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"Error: Graph file not found: {graph_path}")
            return 1
        inspect_graph_reasoning(graph_path)
        return 0

    # Otherwise inspect reasoning JSON
    if not args.file:
        # Try to find latest reasoning file
        reasoning_dir = Path("outputs/reasoning")
        if reasoning_dir.exists():
            reasoning_files = sorted(reasoning_dir.glob("llm_full_context_reasoning_*.json"))
            if reasoning_files:
                reasoning_file = reasoning_files[-1]
                print(f"Using latest reasoning file: {reasoning_file.name}")
            else:
                print("Error: No reasoning files found in outputs/reasoning/")
                return 1
        else:
            print("Error: No reasoning file specified and outputs/reasoning/ not found")
            parser.print_help()
            return 1
    else:
        reasoning_file = Path(args.file)

    if not reasoning_file.exists():
        print(f"Error: File not found: {reasoning_file}")
        return 1

    # Load and display reasoning
    reasoning = load_reasoning(reasoning_file)

    print_reasoning_summary(reasoning)

    if args.cluster is not None:
        print_cluster_details(reasoning, cluster_id=args.cluster)
    elif args.all:
        print_cluster_details(reasoning)
    else:
        print("\nUse --all to see all clusters or --cluster N to see specific cluster")
        print(f"Total clusters: {len(reasoning.get('duplicates', []))}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
