"""
Benchmark Harness for Entity Resolution Deduplication Methodologies.
Orchestrates the entire pipeline from parsing to evaluation.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from google import genai

from src.config import Config
from src.rapidocr_parser import RapidOCRParser
from src.entity_extractor import MultimodalEntityExtractor
from src.token_manager import TokenManager
from src.kg_builder import KnowledgeGraphBuilder

# Import all deduplication methodologies
from src.methodologies.rswoosh import RSwooshDeduplicator
from src.methodologies.probabilistic import ProbabilisticDeduplicator
from src.methodologies.topological import TopologicalDeduplicator
from src.methodologies.semantic import SemanticDeduplicator
from src.methodologies.llm_full_context import LLMFullContextDeduplicator

logger = logging.getLogger(__name__)


class BenchmarkHarness:
    """
    Orchestrates the entire benchmarking pipeline.
    Processes documents, builds knowledge graphs, runs deduplication methodologies.
    """

    def __init__(self):
        """Initialize the benchmark harness."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

        # Initialize components
        logger.info("Initializing components...")

        self.parser = RapidOCRParser()

        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.token_manager = TokenManager(self.client)
        self.entity_extractor = MultimodalEntityExtractor(
            self.client,
            self.token_manager
        )

        self.kg_builder = KnowledgeGraphBuilder()

        # Initialize deduplication methodologies
        self.methodologies = {
            'rswoosh': RSwooshDeduplicator(),
            'probabilistic': ProbabilisticDeduplicator(),
            'topological': TopologicalDeduplicator(),
            'semantic': SemanticDeduplicator(),
            'llm_full_context': LLMFullContextDeduplicator(self.client),
        }

        logger.info("Components initialized successfully")

    def run(self, dataset_path: Path = None):
        """
        Run the complete benchmark pipeline.
        Processes files iteratively to avoid memory overflow.

        Args:
            dataset_path: Path to dataset directory (defaults to config)
        """
        dataset_path = dataset_path or Config.DATASET_DIR

        logger.info("=" * 80)
        logger.info("STARTING ENTITY RESOLUTION BENCHMARK")
        logger.info("=" * 80)

        start_time = time.time()

        # Get list of supported document files
        document_files = []
        for ext in Config.SUPPORTED_FORMATS:
            document_files.extend(dataset_path.glob(f"*{ext}"))

        if not document_files:
            raise ValueError(
                f"No supported documents found in {dataset_path}. "
                f"Supported formats: {Config.SUPPORTED_FORMATS}"
            )

        logger.info(f"Found {len(document_files)} document(s) to process")
        logger.info(f"Supported formats: {Config.SUPPORTED_FORMATS}")

        # Step 1-3: Process each file iteratively
        logger.info("\n[STEP 1-3] Processing files iteratively (parse -> extract -> add to KG)...")
        self._process_files_iteratively(document_files)

        # Step 4: Save non-deduplicated graph
        logger.info("\n[STEP 4] Saving non-deduplicated knowledge graph...")
        self._save_non_dedup_graph()

        # Step 5: Run all deduplication methodologies
        logger.info("\n[STEP 5] Running 5 deduplication methodologies...")
        dedup_results = self._run_all_methodologies()

        # Step 6: Save deduplicated graphs
        logger.info("\n[STEP 6] Saving deduplicated knowledge graphs...")
        self._save_dedup_graphs(dedup_results)

        # Step 7: Generate benchmark report
        logger.info("\n[STEP 7] Generating benchmark report...")
        self._generate_report(dedup_results, start_time)

        total_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"BENCHMARK COMPLETED in {total_time:.2f} seconds")
        logger.info("=" * 80)

    def _process_files_iteratively(self, document_files: List[Path]):
        """
        Process document files one at a time to avoid memory overflow.
        For each file: parse -> extract entities -> add to KG -> cleanup.

        Args:
            document_files: List of document file paths to process
        """
        import gc

        for idx, doc_path in enumerate(document_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {idx}/{len(document_files)}: {doc_path.name}")
            logger.info(f"{'='*60}")

            try:
                # Step 1: Parse this document (universal parser handles all formats)
                logger.info(f"[1/3] Parsing {doc_path.name}...")
                page_pairs = self.parser.parse(doc_path)
                logger.info(f"  → Extracted {len(page_pairs)} pages")

                if not page_pairs:
                    logger.warning(f"  → No pages extracted from {doc_path.name}, skipping")
                    continue

                # Step 2: Extract entities from this document
                logger.info(f"[2/3] Extracting entities from {doc_path.name}...")
                save_path = Config.ENTITIES_EXTRACTED_DIR / f"{doc_path.stem}_entities.json"

                result = self.entity_extractor.extract_entities(
                    page_pairs,
                    doc_path.stem,
                    save_path
                )

                logger.info(
                    f"  → Extracted {len(result['entities'])} entities, "
                    f"{len(result['relationships'])} relationships"
                )

                # Step 3: Add to knowledge graph
                logger.info(f"[3/3] Adding entities to knowledge graph...")
                entities = result.get('entities', [])
                relationships = result.get('relationships', [])

                if entities:
                    # Extract page numbers from entities
                    page_numbers = []
                    for entity in entities:
                        attrs = entity.get('attributes', {})
                        if 'page_number' in attrs:
                            page_numbers.append(attrs['page_number'])

                    self.kg_builder.add_entities(
                        entities,
                        relationships,
                        source_doc=doc_path.stem,
                        page_numbers=page_numbers if page_numbers else None
                    )

                    logger.info(f"  → Added {len(entities)} entities to KG")

                # Step 4: Cleanup memory
                del page_pairs
                del result
                del entities
                del relationships
                gc.collect()

                logger.info(f"✓ Completed processing {doc_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {doc_path.name}: {e}", exc_info=True)
                continue

        # Print final KG statistics
        logger.info(f"\n{'='*60}")
        logger.info("Knowledge Graph Statistics:")
        self.kg_builder.print_statistics()
        logger.info(f"{'='*60}")

    def _parse_documents(self, dataset_path: Path) -> Dict[str, List]:
        """
        Parse all documents in the dataset.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary mapping doc_name -> page_pairs
        """
        pdf_files = list(dataset_path.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {dataset_path}")

        logger.info(f"Found {len(pdf_files)} PDF files")

        parsed_documents = {}

        for pdf_path in pdf_files:
            logger.info(f"Parsing: {pdf_path.name}")

            try:
                page_pairs = self.parser.parse_pdf(pdf_path)
                parsed_documents[pdf_path.stem] = page_pairs

                logger.info(f"  → Extracted {len(page_pairs)} pages")

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path.name}: {e}")
                parsed_documents[pdf_path.stem] = []

        return parsed_documents

    def _extract_entities(self, parsed_documents: Dict) -> Dict[str, Dict]:
        """
        Extract entities from all parsed documents.

        Args:
            parsed_documents: Dictionary of parsed documents

        Returns:
            Dictionary mapping doc_name -> extraction_result
        """
        extraction_results = {}

        for doc_name, page_pairs in parsed_documents.items():
            logger.info(f"Extracting entities from: {doc_name}")

            if not page_pairs:
                logger.warning(f"  → No pages to process for {doc_name}")
                extraction_results[doc_name] = {"entities": [], "relationships": []}
                continue

            try:
                save_path = Config.ENTITIES_EXTRACTED_DIR / f"{doc_name}_entities.json"

                result = self.entity_extractor.extract_entities(
                    page_pairs,
                    doc_name,
                    save_path
                )

                extraction_results[doc_name] = result

                logger.info(
                    f"  → Extracted {len(result['entities'])} entities, "
                    f"{len(result['relationships'])} relationships"
                )

            except Exception as e:
                logger.error(f"Failed to extract entities from {doc_name}: {e}")
                extraction_results[doc_name] = {"entities": [], "relationships": []}

        return extraction_results

    def _build_knowledge_graph(self, extraction_results: Dict):
        """
        Build non-deduplicated knowledge graph from extraction results.

        Args:
            extraction_results: Dictionary of extraction results
        """
        total_entities = 0
        total_relationships = 0

        for doc_name, result in extraction_results.items():
            entities = result.get('entities', [])
            relationships = result.get('relationships', [])

            if not entities:
                continue

            logger.info(f"Adding {len(entities)} entities from {doc_name}")

            # Extract page numbers from entities (if available)
            page_numbers = []
            for entity in entities:
                attrs = entity.get('attributes', {})
                if 'page_number' in attrs:
                    page_numbers.append(attrs['page_number'])

            self.kg_builder.add_entities(
                entities,
                relationships,
                source_doc=doc_name,
                page_numbers=page_numbers if page_numbers else None
            )

            total_entities += len(entities)
            total_relationships += len(relationships)

        logger.info(f"Total entities added: {total_entities}")
        logger.info(f"Total relationships added: {total_relationships}")

        # Print statistics
        self.kg_builder.print_statistics()

    def _save_non_dedup_graph(self):
        """Save the non-deduplicated knowledge graph."""
        save_path_pickle = Config.KNOWLEDGE_GRAPHS_DIR / "non_dedup_kg.gpickle"
        save_path_graphml = Config.KNOWLEDGE_GRAPHS_DIR / "non_dedup_kg.graphml"

        self.kg_builder.save_graph(save_path_pickle, format='gpickle')
        self.kg_builder.save_graph(save_path_graphml, format='graphml')

        logger.info(f"Saved non-deduplicated graph to:")
        logger.info(f"  - {save_path_pickle}")
        logger.info(f"  - {save_path_graphml}")

    def _run_all_methodologies(self) -> Dict[str, Any]:
        """
        Run all deduplication methodologies.

        Returns:
            Dictionary mapping methodology_name -> results
        """
        results = {}
        base_graph = self.kg_builder.get_graph()

        for method_name, deduplicator in self.methodologies.items():
            logger.info(f"\n--- Running {deduplicator.get_methodology_name()} ---")

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

                # Store results
                results[method_name] = {
                    'methodology': deduplicator.get_methodology_name(),
                    'parameters': deduplicator.get_parameters(),
                    'clusters': clusters,
                    'dedup_graph': dedup_graph,
                    'num_clusters': len(clusters),
                    'num_nodes_before': base_graph.number_of_nodes(),
                    'num_nodes_after': dedup_graph.number_of_nodes(),
                    'reduction_pct': (
                        (base_graph.number_of_nodes() - dedup_graph.number_of_nodes())
                        / base_graph.number_of_nodes() * 100
                    ),
                    'execution_time_seconds': elapsed_time,
                }

                logger.info(
                    f"Results: {len(clusters)} clusters, "
                    f"{base_graph.number_of_nodes()} → {dedup_graph.number_of_nodes()} nodes "
                    f"({results[method_name]['reduction_pct']:.1f}% reduction) "
                    f"in {elapsed_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Failed to run {method_name}: {e}", exc_info=True)
                results[method_name] = {
                    'methodology': deduplicator.get_methodology_name(),
                    'error': str(e)
                }

        return results

    def _save_dedup_graphs(self, dedup_results: Dict):
        """
        Save all deduplicated graphs.

        Args:
            dedup_results: Dictionary of deduplication results
        """
        for method_name, result in dedup_results.items():
            if 'error' in result:
                continue

            dedup_graph = result['dedup_graph']

            # Save as pickle and GraphML
            save_path_pickle = Config.KNOWLEDGE_GRAPHS_DIR / f"dedup_{method_name}_kg.gpickle"
            save_path_graphml = Config.KNOWLEDGE_GRAPHS_DIR / f"dedup_{method_name}_kg.graphml"

            try:
                # Create temporary builder to use save methods
                temp_builder = KnowledgeGraphBuilder()
                temp_builder.graph = dedup_graph

                temp_builder.save_graph(save_path_pickle, format='gpickle')
                temp_builder.save_graph(save_path_graphml, format='graphml')

                logger.info(f"Saved {method_name} graph to {save_path_pickle.name}")

            except Exception as e:
                logger.error(f"Failed to save {method_name} graph: {e}")

    def _save_reasoning(self, method_name: str, reasoning_data: Dict[str, Any]):
        """
        Save LLM reasoning to JSON file.

        Args:
            method_name: Name of the methodology
            reasoning_data: Reasoning data from LLM deduplication
        """
        output_dir = Config.REASONING_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"{method_name}_reasoning_{timestamp}.json"

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved reasoning to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save reasoning: {e}")

    def _generate_report(self, dedup_results: Dict, start_time: float):
        """
        Generate benchmark report.

        Args:
            dedup_results: Dictionary of deduplication results
            start_time: Benchmark start time
        """
        total_time = time.time() - start_time

        report = {
            'timestamp': self.timestamp,
            'total_execution_time_seconds': total_time,
            'dataset': str(Config.DATASET_DIR),
            'configuration': {
                'model_heavy': Config.MODEL_HEAVY,
                'model_light': Config.MODEL_LIGHT,
                'max_tokens_per_request': Config.MAX_TOKENS_PER_REQUEST,
            },
            'non_dedup_graph_stats': self.kg_builder.get_statistics(),
            'deduplication_results': {}
        }

        # Add results for each methodology
        for method_name, result in dedup_results.items():
            if 'error' in result:
                report['deduplication_results'][method_name] = {
                    'error': result['error']
                }
            else:
                report['deduplication_results'][method_name] = {
                    'methodology': result['methodology'],
                    'parameters': result['parameters'],
                    'num_clusters_found': result['num_clusters'],
                    'nodes_before': result['num_nodes_before'],
                    'nodes_after': result['num_nodes_after'],
                    'reduction_percentage': result['reduction_pct'],
                    'execution_time_seconds': result['execution_time_seconds'],
                }

        # Save report
        report_path = Config.OUTPUTS_DIR / f"benchmark_report_{self.timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\nBenchmark report saved to: {report_path}")

        # Print summary
        self._print_summary(report)

    def _print_summary(self, report: Dict):
        """
        Print benchmark summary to console.

        Args:
            report: Benchmark report dictionary
        """
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\nTotal Execution Time: {report['total_execution_time_seconds']:.2f} seconds")

        print("\nNon-Deduplicated Graph:")
        print(f"  Nodes: {report['non_dedup_graph_stats']['num_nodes']:,}")
        print(f"  Edges: {report['non_dedup_graph_stats']['num_edges']:,}")

        print("\nDeduplication Results:")
        print(f"{'Methodology':<25} {'Clusters':<10} {'Nodes After':<12} {'Reduction':<12} {'Time (s)':<10}")
        print("-" * 80)

        for method_name, result in report['deduplication_results'].items():
            if 'error' in result:
                print(f"{result.get('methodology', method_name):<25} ERROR")
            else:
                print(
                    f"{result['methodology']:<25} "
                    f"{result['num_clusters_found']:<10} "
                    f"{result['nodes_after']:<12,} "
                    f"{result['reduction_percentage']:<11.1f}% "
                    f"{result['execution_time_seconds']:<10.2f}"
                )

        print("=" * 80 + "\n")


def create_benchmark_harness() -> BenchmarkHarness:
    """
    Create and return a BenchmarkHarness instance.

    Returns:
        BenchmarkHarness instance
    """
    return BenchmarkHarness()
