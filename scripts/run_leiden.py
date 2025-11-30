#!/usr/bin/env python
"""
Leiden Community Detection - Batch Job Script

Runs the complete Leiden pipeline:
1. Apply schema (create tables if needed)
2. Calculate graph metrics
3. Auto-tune resolution parameters
4. Run Leiden at 3 hierarchical levels
5. Generate community summaries with Gemini
6. Update Milvus with community tags

Can be run:
- On-demand: python scripts/run_leiden.py
- Scheduled: via cron (nightly batch)
- Triggered: when graph changes significantly
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_metrics import GraphMetricsCalculator
from src.leiden_builder import LeidenCommunityBuilder
from src.community_summarizer import CommunitySummarizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_schema(db_path: str):
    """
    Apply Leiden schema to database if tables don't exist.

    Args:
        db_path: Path to database
    """
    import sqlite3

    schema_path = Path(__file__).parent.parent / "sql" / "leiden_schema.sql"

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    logger.info(f"Applying Leiden schema from: {schema_path}")

    try:
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.executescript(schema_sql)
            conn.commit()
            logger.info("✓ Schema applied successfully")
            return True
        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Failed to apply schema: {e}")
        return False


def run_leiden_pipeline(
    db_path: str = "knowledge_graph.db",
    force: bool = False,
    skip_summaries: bool = False
):
    """
    Run the complete Leiden community detection pipeline.

    Args:
        db_path: Path to SQLite database
        force: Force recomputation even if not needed
        skip_summaries: Skip Gemini summary generation (faster for testing)

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("LEIDEN COMMUNITY DETECTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Force rebuild: {force}")
    logger.info(f"Generate summaries: {not skip_summaries}")
    logger.info("=" * 80)

    try:
        # Step 0: Apply schema
        logger.info("\n**Step 0: Apply Schema**")
        if not apply_schema(db_path):
            logger.error("Schema application failed")
            return False

        # Step 1: Check if recomputation needed
        logger.info("\n**Step 1: Analyzing Graph State**")
        metrics_calc = GraphMetricsCalculator(db_path)

        if not force:
            should_recompute, reason = metrics_calc.should_recompute_leiden()
            logger.info(f"Recomputation check: {should_recompute}")
            logger.info(f"Reason: {reason}")

            if not should_recompute:
                logger.info("✓ Leiden communities are up-to-date, skipping")
                return True
        else:
            logger.info("Force mode enabled, rebuilding communities")

        # Step 2: Calculate metrics and optimal resolutions
        logger.info("\n**Step 2: Calculate Graph Metrics**")
        print(metrics_calc.get_summary_statistics())

        # Step 3: Run Leiden algorithm
        logger.info("\n**Step 3: Run Leiden Community Detection**")
        leiden_builder = LeidenCommunityBuilder(db_path)
        leiden_stats = leiden_builder.build_communities(force=True)

        logger.info("\nLeiden Statistics:")
        for key, value in leiden_stats.items():
            logger.info(f"  {key}: {value}")

        # Step 4: Generate summaries (optional)
        if not skip_summaries:
            logger.info("\n**Step 4: Generate Community Summaries**")
            summarizer = CommunitySummarizer(db_path)
            summary_counts = summarizer.generate_all_summaries()

            logger.info("\nSummary Counts:")
            for level, count in summary_counts.items():
                logger.info(f"  {level}: {count} summaries")
        else:
            logger.info("\n**Step 4: Skipped (--skip-summaries)**")

        # Step 5: Done
        logger.info("\n" + "=" * 80)
        logger.info("✓ LEIDEN PIPELINE COMPLETE")
        logger.info("=" * 80)

        return True

    except ImportError as e:
        logger.error(f"\nDependency Error: {e}")
        logger.error("\nPlease install required packages:")
        logger.error("  pip install leidenalg python-igraph")
        return False

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Leiden Community Detection Pipeline"
    )

    parser.add_argument(
        '--db',
        type=str,
        default='knowledge_graph.db',
        help='Path to SQLite database (default: knowledge_graph.db)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation even if communities are up-to-date'
    )

    parser.add_argument(
        '--skip-summaries',
        action='store_true',
        help='Skip Gemini summary generation (faster for testing)'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if recomputation is needed, don\'t run'
    )

    args = parser.parse_args()

    # Check-only mode
    if args.check_only:
        logger.info("Check-only mode: Analyzing graph state...")
        metrics_calc = GraphMetricsCalculator(args.db)
        should_recompute, reason = metrics_calc.should_recompute_leiden()

        print(f"\nShould recompute: {should_recompute}")
        print(f"Reason: {reason}")

        if should_recompute:
            print("\nRun without --check-only to recompute communities")

        sys.exit(0 if not should_recompute else 1)

    # Run pipeline
    success = run_leiden_pipeline(
        db_path=args.db,
        force=args.force,
        skip_summaries=args.skip_summaries
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
