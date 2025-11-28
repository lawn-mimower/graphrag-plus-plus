"""
Probabilistic Deduplication Methodology using Splink.
Implements Fellegi-Sunter probabilistic record linkage.
"""

import logging
from typing import Set, List, Dict, Any
import networkx as nx
import pandas as pd
from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl

from src.config import Config

logger = logging.getLogger(__name__)


class ProbabilisticDeduplicator:
    """
    Probabilistic deduplication using Splink (Fellegi-Sunter model).
    Uses probabilistic matching scores to identify duplicates.
    """

    def __init__(self, match_threshold: float = None):
        """
        Initialize Probabilistic deduplicator.

        Args:
            match_threshold: Threshold for match probability
        """
        self.threshold = match_threshold or Config.SPLINK_MATCH_PROBABILITY_THRESHOLD

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform probabilistic deduplication on the knowledge graph.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        logger.info("Running Probabilistic (Splink) deduplication")

        # Convert graph to DataFrame
        df = self._graph_to_dataframe(graph)

        if len(df) < 2:
            logger.warning("Not enough entities for Splink matching")
            return []

        logger.info(f"Processing {len(df)} entities with Splink")

        # Build Splink settings
        settings = self._build_splink_settings(df)

        # Create linker
        linker = DuckDBLinker(df, settings)

        # Estimate parameters (EM algorithm)
        try:
            # Use deterministic rules for training
            linker.estimate_u_using_random_sampling(max_pairs=1e6)

            # Estimate m probabilities using EM
            # Training blocking rules to reduce comparison space
            blocking_rules = self._get_blocking_rules(df)

            for blocking_rule in blocking_rules:
                try:
                    linker.estimate_parameters_using_expectation_maximisation(
                        blocking_rule,
                        estimate_without_term_frequencies=True
                    )
                except Exception as e:
                    logger.warning(f"EM estimation failed for rule {blocking_rule}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Parameter estimation failed: {e}")

        # Perform deduplication
        try:
            df_predictions = linker.predict(threshold_match_probability=self.threshold)
            results = df_predictions.as_pandas_dataframe()
        except Exception as e:
            logger.error(f"Splink prediction failed: {e}")
            return []

        # Extract clusters from results
        clusters = self._extract_clusters(results, df)

        logger.info(f"Found {len(clusters)} duplicate clusters using Splink")

        return clusters

    def _graph_to_dataframe(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Convert knowledge graph to pandas DataFrame for Splink.

        Args:
            graph: Knowledge graph

        Returns:
            DataFrame with entity attributes
        """
        records = []

        for node, attrs in graph.nodes(data=True):
            record = {
                'node_id': node,
                'type': attrs.get('type', ''),
                'name': attrs.get('name', ''),
                'role': attrs.get('role', ''),
                'email': attrs.get('email', ''),
                'phone': attrs.get('phone', ''),
                'din': attrs.get('din', ''),
                'pan': attrs.get('pan', ''),
                'company': attrs.get('company', ''),
                'address': attrs.get('address', ''),
                'source_doc': attrs.get('source_doc', ''),
            }
            records.append(record)

        df = pd.DataFrame(records)

        # Fill NA with empty strings
        df = df.fillna('')

        return df

    def _build_splink_settings(self, df: pd.DataFrame) -> Dict:
        """
        Build Splink settings dictionary.

        Args:
            df: Input DataFrame

        Returns:
            Splink settings dictionary
        """
        # Detect which columns have data
        columns_to_compare = []

        for col in ['name', 'role', 'email', 'phone', 'din', 'pan', 'company', 'address']:
            if col in df.columns and df[col].str.len().sum() > 0:
                columns_to_compare.append(col)

        logger.debug(f"Splink will compare columns: {columns_to_compare}")

        # Build comparison configurations
        comparisons = []

        # Name comparison (fuzzy matching)
        if 'name' in columns_to_compare:
            comparisons.append(
                cl.jaro_winkler_at_thresholds("name", [0.9, 0.7])
            )

        # Email comparison (exact and fuzzy)
        if 'email' in columns_to_compare:
            comparisons.append(
                cl.exact_match("email")
            )

        # Phone comparison
        if 'phone' in columns_to_compare:
            comparisons.append(
                cl.exact_match("phone")
            )

        # ID comparisons (DIN, PAN)
        for id_field in ['din', 'pan']:
            if id_field in columns_to_compare:
                comparisons.append(
                    cl.exact_match(id_field)
                )

        # Role/Company comparisons
        for text_field in ['role', 'company']:
            if text_field in columns_to_compare:
                comparisons.append(
                    cl.jaro_winkler_at_thresholds(text_field, [0.9, 0.7])
                )

        # Address comparison
        if 'address' in columns_to_compare:
            comparisons.append(
                cl.jaro_winkler_at_thresholds("address", [0.85, 0.7])
            )

        settings = {
            "link_type": "dedupe_only",
            "unique_id_column_name": "node_id",
            "comparisons": comparisons,
            "retain_intermediate_calculation_columns": True,
        }

        # Add blocking rules if needed
        blocking_rules = self._get_blocking_rules_for_settings(df)
        if blocking_rules:
            settings["blocking_rules_to_generate_predictions"] = blocking_rules

        return settings

    def _get_blocking_rules(self, df: pd.DataFrame) -> List[str]:
        """
        Get blocking rules for training.

        Args:
            df: Input DataFrame

        Returns:
            List of blocking rule strings
        """
        rules = []

        # Block on entity type
        if 'type' in df.columns:
            rules.append("l.type = r.type")

        # Block on first character of name
        if 'name' in df.columns:
            rules.append("substr(l.name, 1, 1) = substr(r.name, 1, 1)")

        return rules

    def _get_blocking_rules_for_settings(self, df: pd.DataFrame) -> List[str]:
        """
        Get blocking rules for prediction settings.

        Args:
            df: Input DataFrame

        Returns:
            List of blocking rules
        """
        # For prediction, we can be more lenient
        # Only block on type to reduce comparison space
        rules = []

        if 'type' in df.columns and df['type'].nunique() > 1:
            rules.append("l.type = r.type")

        return rules

    def _extract_clusters(
        self,
        predictions: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> List[Set[str]]:
        """
        Extract duplicate clusters from Splink predictions.

        Args:
            predictions: Splink predictions DataFrame
            original_df: Original entity DataFrame

        Returns:
            List of duplicate clusters
        """
        # Build a graph of matches
        match_graph = nx.Graph()

        # Add all nodes
        for node_id in original_df['node_id']:
            match_graph.add_node(node_id)

        # Add edges for matches
        for _, row in predictions.iterrows():
            left_id = row['node_id_l']
            right_id = row['node_id_r']
            probability = row.get('match_probability', 0)

            if probability >= self.threshold:
                match_graph.add_edge(left_id, right_id, weight=probability)

        # Extract connected components as clusters
        clusters = []
        for component in nx.connected_components(match_graph):
            if len(component) > 1:
                clusters.append(set(component))

        return clusters

    def get_methodology_name(self) -> str:
        """Get the name of this methodology."""
        return "Probabilistic (Splink)"

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used by this methodology."""
        return {
            "methodology": "Probabilistic",
            "framework": "Splink",
            "model": "Fellegi-Sunter",
            "match_threshold": self.threshold
        }


def create_probabilistic_deduplicator() -> ProbabilisticDeduplicator:
    """
    Create and return a ProbabilisticDeduplicator instance.

    Returns:
        ProbabilisticDeduplicator instance
    """
    return ProbabilisticDeduplicator()
