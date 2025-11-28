#!/usr/bin/env python3
"""
Benchmark Script for Deduplication Methodologies
Compares 4 non-LLM methods + 1 LLM method on extracted entities.
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.kg_builder import KnowledgeGraphBuilder
from google import genai

# Import all deduplication methodologies
from src.methodologies.rswoosh import RSwooshDeduplicator
from src.methodologies.probabilistic import ProbabilisticDeduplicator
from src.methodologies.topological import TopologicalDeduplicator
from src.methodologies.semantic import SemanticDeduplicator
from src.methodologies.llm_full_context import LLMFullContextDeduplicator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class DeduplicationBenchmark:
    """
    Benchmark harness for comparing deduplication methodologies.
    """

    def __init__(self, entities_json_path: Path):
        """
        Initialize benchmark.

        Args:
            entities_json_path: Path to extracted entities JSON file
        """
        self.entities_json_path = entities_json_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Loading entities from: {entities_json_path}")

        # Load entities JSON
        with open(entities_json_path, 'r', encoding='utf-8') as f:
            self.entities_data = json.load(f)

        # Build knowledge graph from entities
        self.kg_builder = KnowledgeGraphBuilder()
        self._build_kg_from_json()

        # Initialize methodologies
        self.methodologies = self._initialize_methodologies()

        logger.info(f"Initialized {len(self.methodologies)} deduplication methodologies")

    def _build_kg_from_json(self):
        """Build knowledge graph from entities JSON."""
        entities = self.entities_data.get('entities', [])
        relationships = self.entities_data.get('relationships', [])

        logger.info(f"Building KG from {len(entities)} entities, {len(relationships)} relationships")

        # Extract source document name from file path
        source_doc = self.entities_json_path.stem.replace('_entities', '')

        # Extract page numbers from entities if available
        page_numbers = []
        for entity in entities:
            attrs = entity.get('attributes', {})
            if 'page_number' in attrs:
                page_numbers.append(attrs['page_number'])
            elif 'pages' in attrs:
                pages = attrs['pages']
                if isinstance(pages, list):
                    page_numbers.extend(pages)
            elif 'pages_mentioned' in attrs:
                pages = attrs['pages_mentioned']
                if isinstance(pages, list):
                    page_numbers.extend(pages)

        self.kg_builder.add_entities(
            entities,
            relationships,
            source_doc=source_doc,
            page_numbers=list(set(page_numbers)) if page_numbers else None
        )

        logger.info(f"KG built with {self.kg_builder.graph.number_of_nodes()} nodes, "
                   f"{self.kg_builder.graph.number_of_edges()} edges")

    def _initialize_methodologies(self) -> Dict[str, Any]:
        """
        Initialize all deduplication methodologies.

        Returns:
            Dictionary mapping method name to deduplicator instance
        """
        # Initialize Gemini client for LLM method
        client = genai.Client(api_key=Config.GOOGLE_API_KEY)

        return {
            'rswoosh': RSwooshDeduplicator(),
            'probabilistic': ProbabilisticDeduplicator(),
            'topological': TopologicalDeduplicator(),
            'semantic': SemanticDeduplicator(),
            'llm_full_context': LLMFullContextDeduplicator(client),
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run benchmark on all methodologies.

        Returns:
            Benchmark results dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING DEDUPLICATION BENCHMARK")
        logger.info("=" * 80)

        base_graph = self.kg_builder.get_graph()

        logger.info(f"Base graph: {base_graph.number_of_nodes()} nodes, "
                   f"{base_graph.number_of_edges()} edges")

        results = {}

        for method_name, deduplicator in self.methodologies.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {deduplicator.get_methodology_name()}")
            logger.info(f"{'='*60}")

            start_time = time.time()

            try:
                # Check if this is LLM Full Context method (has reasoning)
                reasoning_data = None
                if isinstance(deduplicator, LLMFullContextDeduplicator):
                    # Use deduplicate_with_details to get reasoning
                    logger.info("Capturing LLM reasoning for explainability...")
                    detailed_result = deduplicator.deduplicate_with_details(base_graph)
                    reasoning_data = detailed_result

                    # Extract clusters from detailed result
                    clusters = []
                    for duplicate_group in detailed_result.get("duplicates", []):
                        entity_ids = duplicate_group.get("entities", [])
                        if len(entity_ids) > 1:
                            clusters.append(set(entity_ids))

                    # Save reasoning to JSON file
                    self._save_reasoning(method_name, reasoning_data)
                else:
                    # Standard deduplication (no reasoning)
                    clusters = deduplicator.deduplicate(base_graph)

                # Create deduplicated graph (with reasoning if available)
                dedup_graph = self.kg_builder.create_deduplicated_graph(
                    clusters,
                    cluster_reasoning=reasoning_data
                )

                elapsed_time = time.time() - start_time

                # Calculate metrics
                num_nodes_before = base_graph.number_of_nodes()
                num_nodes_after = dedup_graph.number_of_nodes()
                reduction_pct = ((num_nodes_before - num_nodes_after) / num_nodes_before * 100) if num_nodes_before > 0 else 0

                # Store results
                results[method_name] = {
                    'methodology': deduplicator.get_methodology_name(),
                    'parameters': deduplicator.get_parameters(),
                    'num_clusters': len(clusters),
                    'nodes_before': num_nodes_before,
                    'nodes_after': num_nodes_after,
                    'reduction_percentage': reduction_pct,
                    'execution_time_seconds': elapsed_time,
                    'status': 'success'
                }

                logger.info(f"Results: {len(clusters)} clusters, "
                           f"{num_nodes_before} → {num_nodes_after} nodes "
                           f"({reduction_pct:.1f}% reduction) "
                           f"in {elapsed_time:.2f}s")

                # Save deduplicated graph
                self._save_dedup_graph(method_name, dedup_graph)

            except Exception as e:
                logger.error(f"Failed to run {method_name}: {e}", exc_info=True)

                results[method_name] = {
                    'methodology': deduplicator.get_methodology_name(),
                    'status': 'failed',
                    'error': str(e)
                }

        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK COMPLETED")
        logger.info("=" * 80)

        return results

    def _save_dedup_graph(self, method_name: str, graph):
        """
        Save deduplicated graph.

        Args:
            method_name: Name of the methodology
            graph: Deduplicated graph
        """
        output_dir = Config.KNOWLEDGE_GRAPHS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"dedup_{method_name}_{self.timestamp}.gpickle"

        try:
            temp_builder = KnowledgeGraphBuilder()
            temp_builder.graph = graph
            temp_builder.save_graph(save_path, format='gpickle')

            logger.debug(f"Saved graph to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def _save_reasoning(self, method_name: str, reasoning_data: Dict[str, Any]):
        """
        Save LLM reasoning to JSON file.

        Args:
            method_name: Name of the methodology
            reasoning_data: Reasoning data from LLM deduplication
        """
        output_dir = Config.REASONING_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"{method_name}_reasoning_{self.timestamp}.json"

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved reasoning to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save reasoning: {e}")

    def generate_report(self, results: Dict[str, Any]):
        """
        Generate and save benchmark report.

        Args:
            results: Benchmark results
        """
        report = {
            'timestamp': self.timestamp,
            'input_file': str(self.entities_json_path),
            'base_graph': {
                'num_nodes': self.kg_builder.graph.number_of_nodes(),
                'num_edges': self.kg_builder.graph.number_of_edges(),
            },
            'methodologies': results
        }

        # Save JSON report
        report_path = Config.OUTPUTS_DIR / f"dedup_benchmark_{self.timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\nReport saved to: {report_path}")

        # Print summary table
        self._print_summary_table(results)

    def _print_summary_table(self, results: Dict[str, Any]):
        """
        Print summary table to console.

        Args:
            results: Benchmark results
        """
        print("\n" + "=" * 100)
        print("DEDUPLICATION BENCHMARK RESULTS")
        print("=" * 100)

        print(f"\n{'Methodology':<25} {'Clusters':<10} {'Nodes After':<12} {'Reduction':<12} {'Time (s)':<10} {'Status':<10}")
        print("-" * 100)

        for method_name, result in results.items():
            if result['status'] == 'success':
                print(
                    f"{result['methodology']:<25} "
                    f"{result['num_clusters']:<10} "
                    f"{result['nodes_after']:<12} "
                    f"{result['reduction_percentage']:<11.1f}% "
                    f"{result['execution_time_seconds']:<10.2f} "
                    f"{result['status']:<10}"
                )
            else:
                print(f"{result['methodology']:<25} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} {result['status']:<10}")

        print("=" * 100 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark deduplication methodologies on extracted entities'
    )

    parser.add_argument(
        'entities_json',
        type=str,
        help='Path to entities JSON file (e.g., outputs/entities_extracted/XBRL CFS_entities.json)'
    )

    args = parser.parse_args()

    # Validate input file
    entities_path = Path(args.entities_json)

    if not entities_path.exists():
        logger.error(f"File not found: {entities_path}")
        return 1

    if not entities_path.is_file():
        logger.error(f"Not a file: {entities_path}")
        return 1

    try:
        # Create benchmark
        benchmark = DeduplicationBenchmark(entities_path)

        # Run benchmark
        results = benchmark.run_benchmark()

        # Generate report
        benchmark.generate_report(results)

        logger.info("\n✓ Benchmark completed successfully!")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
