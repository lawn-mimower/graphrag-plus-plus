#!/usr/bin/env python
"""
Initialize SQL Schema and Ingest into Milvus.

This script:
1. Ensures SQL tables are created from schema.sql
2. Verifies the database has entities
3. Runs Milvus ingestion
"""

import sqlite3
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def initialize_sql_database(db_path: str = "knowledge_graph.db", schema_path: str = "sql/schema.sql"):
    """
    Initialize SQL database with schema.

    Args:
        db_path: Path to SQLite database
        schema_path: Path to schema.sql file

    Returns:
        True if successful, False otherwise
    """
    print_banner("STEP 1: Initialize SQL Database")

    logger.info(f"Database path: {db_path}")
    logger.info(f"Schema path: {schema_path}")

    # Check if schema file exists
    schema_file = Path(schema_path)
    if not schema_file.exists():
        logger.error(f"Schema file not found at {schema_path}")
        return False

    # Read schema
    logger.info("Reading schema file...")
    with open(schema_file, 'r') as f:
        schema_sql = f.read()

    # Execute schema
    logger.info("Creating tables...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute schema (may contain multiple statements)
        cursor.executescript(schema_sql)
        conn.commit()

        logger.info("✓ SQL tables created successfully")

        # Verify tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"Found {len(tables)} tables in database:")
        for table in tables:
            logger.info(f"  - {table}")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create SQL tables: {e}")
        return False


def check_entity_count(db_path: str = "knowledge_graph.db"):
    """
    Check entity count in database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Number of entities, or None if error
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM entities")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    except Exception as e:
        logger.error(f"Failed to count entities: {e}")
        return None


def run_milvus_ingestion():
    """
    Run Milvus ingestion process.

    Returns:
        True if successful, False otherwise
    """
    print_banner("STEP 2: Ingest Entities into Milvus")

    try:
        from src.milvus_ingestion import run_ingestion

        stats = run_ingestion()

        logger.info("✓ Milvus ingestion completed successfully")
        logger.info(f"  Total entities: {stats['total_entities']}")
        logger.info(f"  Ingested: {stats['ingested']}")
        logger.info(f"  Collection: {stats['collection_name']}")

        return True

    except Exception as e:
        logger.error(f"Milvus ingestion failed: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    print_banner("BAKASUR PHASE 2 - Database Initialization & Milvus Ingestion")

    db_path = "knowledge_graph.db"
    schema_path = "sql/schema.sql"

    # Step 1: Initialize SQL Database
    if not initialize_sql_database(db_path, schema_path):
        logger.error("SQL database initialization failed")
        sys.exit(1)

    # Check entity count
    print()
    entity_count = check_entity_count(db_path)

    if entity_count is None:
        logger.error("Failed to check entity count")
        sys.exit(1)

    logger.info(f"Current entity count: {entity_count}")

    if entity_count == 0:
        logger.warning("⚠ WARNING: No entities found in database!")
        logger.warning("You need to populate the database with entities before running Milvus ingestion.")
        logger.warning("")
        logger.warning("Skipping Milvus ingestion step.")
        logger.warning("")
        logger.warning("To populate the database, run your entity extraction pipeline first.")
        sys.exit(0)

    # Step 2: Ingest into Milvus
    print()
    if not run_milvus_ingestion():
        logger.error("Milvus ingestion failed")
        sys.exit(1)

    # Success!
    print_banner("✓ INITIALIZATION AND INGESTION COMPLETE")

    logger.info(f"Database ready at: {db_path}")
    logger.info("Milvus database at: ./outputs/milvus_orchestrator.db")
    logger.info("")
    logger.info("You can now use the Query Orchestrator:")
    logger.info("  python -m src.orchestrator")


if __name__ == "__main__":
    main()
