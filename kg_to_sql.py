#!/usr/bin/env python3
"""
Knowledge Graph to SQL Converter

Converts NetworkX deduplicated knowledge graphs (.gpickle) to normalized SQLite database.
Creates 5 tables: documents, entities, relationships, entity_occurrences, relationship_occurrences.

Usage:
    python kg_to_sql.py --graph path/to/graph.gpickle --output kg.db
    python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle
"""

import sys
import sqlite3
import pickle
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from datetime import datetime
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_entity_hash(canonical_name: str, entity_type: str) -> str:
    """
    Generate deterministic hash for entity ID.

    Args:
        canonical_name: Entity's canonical name
        entity_type: Entity type (Person, Company, etc.)

    Returns:
        16-character hex string
    """
    content = f"{canonical_name.strip().upper()}|{entity_type.strip().upper()}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def generate_document_hash(document_name: str) -> str:
    """
    Generate deterministic hash for document ID.

    Args:
        document_name: Document name

    Returns:
        16-character hex string
    """
    return hashlib.md5(document_name.strip().encode('utf-8')).hexdigest()[:16]


def generate_relationship_hash(from_id: str, to_id: str, rel_type: str) -> str:
    """
    Generate deterministic hash for relationship ID.

    Args:
        from_id: Source entity ID
        to_id: Target entity ID
        rel_type: Relationship type

    Returns:
        16-character hex string
    """
    content = f"{from_id}|{to_id}|{rel_type.strip().upper()}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def extract_canonical_name(node_attrs: Dict[str, Any]) -> str:
    """
    Extract canonical name from node attributes.
    Priority: name > original_id > "UNKNOWN"

    Args:
        node_attrs: Node attributes dictionary

    Returns:
        Canonical name string
    """
    return (node_attrs.get('name') or
            node_attrs.get('original_id') or
            'UNKNOWN_ENTITY')


def extract_attributes_json(node_attrs: Dict[str, Any]) -> str:
    """
    Extract domain-specific attributes to JSON, excluding metadata fields.

    Args:
        node_attrs: Node attributes dictionary

    Returns:
        JSON string of attributes
    """
    # Metadata fields to exclude
    exclude_fields = {
        'original_id', 'type', 'source_doc', 'source_docs',
        'page_numbers', 'cluster_size', 'merge_reasoning', 'name'
    }

    # Extract remaining attributes
    attributes = {
        k: v for k, v in node_attrs.items()
        if k not in exclude_fields and v is not None
    }

    return json.dumps(attributes, ensure_ascii=False) if attributes else '{}'


def extract_documents_from_graph(graph: nx.DiGraph) -> Set[str]:
    """
    Extract unique document names from graph.

    Args:
        graph: NetworkX graph

    Returns:
        Set of document names
    """
    documents = set()

    for _, attrs in graph.nodes(data=True):
        # Check source_doc (single document)
        if 'source_doc' in attrs:
            documents.add(attrs['source_doc'])

        # Check source_docs (multiple documents)
        if 'source_docs' in attrs:
            docs = attrs['source_docs']
            if isinstance(docs, list):
                documents.update(docs)

    # Also check edges
    for _, _, attrs in graph.edges(data=True):
        if 'source_doc' in attrs:
            documents.add(attrs['source_doc'])
        if 'source_docs' in attrs:
            docs = attrs['source_docs']
            if isinstance(docs, list):
                documents.update(docs)

    return {d for d in documents if d}  # Remove empty strings


def create_schema(conn: sqlite3.Connection):
    """
    Create database schema from schema.sql file.

    Args:
        conn: SQLite connection
    """
    schema_path = Path(__file__).parent / 'sql' / 'schema.sql'

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    logger.info("Database schema created")


def insert_documents(conn: sqlite3.Connection, documents: Set[str]):
    """
    Insert documents into database with default UNCLASSIFIED access.

    Args:
        conn: SQLite connection
        documents: Set of document names
    """
    cursor = conn.cursor()

    for doc_name in documents:
        doc_id = generate_document_hash(doc_name)

        cursor.execute("""
            INSERT OR IGNORE INTO documents (document_id, document_name, access_tags)
            VALUES (?, ?, ?)
        """, (doc_id, doc_name, '["UNCLASSIFIED"]'))

    conn.commit()
    logger.info(f"Inserted {len(documents)} documents")


def insert_entities(conn: sqlite3.Connection, graph: nx.DiGraph) -> Dict[str, str]:
    """
    Insert entities into database and return node_id -> entity_id mapping.

    Args:
        conn: SQLite connection
        graph: NetworkX graph

    Returns:
        Dictionary mapping graph node IDs to database entity IDs
    """
    cursor = conn.cursor()
    node_to_entity = {}

    for node_id, attrs in graph.nodes(data=True):
        # Extract entity data
        entity_type = attrs.get('type', 'Unknown')
        canonical_name = extract_canonical_name(attrs)
        cluster_size = attrs.get('cluster_size', 1)

        # Generate unique ID
        entity_id = generate_entity_hash(canonical_name, entity_type)
        node_to_entity[node_id] = entity_id

        # Extract attributes
        attributes_json = extract_attributes_json(attrs)

        # Extract merge reasoning if available
        merge_reasoning = attrs.get('merge_reasoning')
        merge_reasoning_json = json.dumps(merge_reasoning) if merge_reasoning else None

        # Extract name variants (from merged_entities if available)
        name_variants = None
        if merge_reasoning and 'merged_entities' in merge_reasoning:
            variants = [node_id]  # Include current node ID
            name_variants = json.dumps(variants)

        # Insert entity
        cursor.execute("""
            INSERT OR REPLACE INTO entities (
                unique_entity_id, entity_type, canonical_name, name_variants,
                cluster_size, attributes, merge_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id, entity_type, canonical_name, name_variants,
            cluster_size, attributes_json, merge_reasoning_json
        ))

    conn.commit()
    logger.info(f"Inserted {len(node_to_entity)} entities")

    return node_to_entity


def insert_entity_occurrences(
    conn: sqlite3.Connection,
    graph: nx.DiGraph,
    node_to_entity: Dict[str, str]
):
    """
    Insert entity occurrences (which documents and pages).

    Args:
        conn: SQLite connection
        graph: NetworkX graph
        node_to_entity: Mapping of node IDs to entity IDs
    """
    cursor = conn.cursor()
    count = 0

    for node_id, attrs in graph.nodes(data=True):
        entity_id = node_to_entity[node_id]

        # Collect document-to-pages mapping
        doc_pages = {}

        # Handle source_doc + page_numbers (single document)
        if 'source_doc' in attrs:
            doc_name = attrs['source_doc']
            pages = attrs.get('page_numbers', [])
            if doc_name:
                doc_pages[doc_name] = pages

        # Handle source_docs (multiple documents - merged entity)
        if 'source_docs' in attrs:
            docs = attrs['source_docs']
            pages = attrs.get('page_numbers', [])
            if isinstance(docs, list):
                # Distribute pages across all docs (simple approach)
                for doc_name in docs:
                    if doc_name:
                        doc_pages[doc_name] = pages

        # Insert occurrences
        for doc_name, pages in doc_pages.items():
            doc_id = generate_document_hash(doc_name)
            pages_json = json.dumps(sorted(list(set(pages)))) if pages else '[]'

            cursor.execute("""
                INSERT OR REPLACE INTO entity_occurrences (
                    entity_id, document_id, page_numbers
                ) VALUES (?, ?, ?)
            """, (entity_id, doc_id, pages_json))
            count += 1

    conn.commit()
    logger.info(f"Inserted {count} entity occurrences")


def insert_relationships(
    conn: sqlite3.Connection,
    graph: nx.DiGraph,
    node_to_entity: Dict[str, str]
) -> Dict[Tuple[str, str, str], str]:
    """
    Insert relationships into database.

    Args:
        conn: SQLite connection
        graph: NetworkX graph
        node_to_entity: Mapping of node IDs to entity IDs

    Returns:
        Dictionary mapping (from, to, type) to relationship IDs
    """
    cursor = conn.cursor()
    relationship_map = {}

    for from_node, to_node, attrs in graph.edges(data=True):
        from_entity = node_to_entity[from_node]
        to_entity = node_to_entity[to_node]
        rel_type = attrs.get('relationship_type', 'RELATED_TO')

        # Generate relationship ID
        rel_id = generate_relationship_hash(from_entity, to_entity, rel_type)
        relationship_map[(from_entity, to_entity, rel_type)] = rel_id

        # Extract relationship attributes
        exclude_fields = {'relationship_type', 'source_doc', 'source_docs'}
        rel_attributes = {
            k: v for k, v in attrs.items()
            if k not in exclude_fields and v is not None
        }
        attributes_json = json.dumps(rel_attributes) if rel_attributes else '{}'

        # Insert relationship
        cursor.execute("""
            INSERT OR REPLACE INTO relationships (
                relationship_id, relationship_type, from_entity_id,
                to_entity_id, attributes, access_policy
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (rel_id, rel_type, from_entity, to_entity, attributes_json, 'OPEN'))

    conn.commit()
    logger.info(f"Inserted {len(relationship_map)} relationships")

    return relationship_map


def insert_relationship_occurrences(
    conn: sqlite3.Connection,
    graph: nx.DiGraph,
    node_to_entity: Dict[str, str],
    relationship_map: Dict[Tuple[str, str, str], str]
):
    """
    Insert relationship occurrences (which documents and pages).

    Args:
        conn: SQLite connection
        graph: NetworkX graph
        node_to_entity: Mapping of node IDs to entity IDs
        relationship_map: Mapping of relationships to IDs
    """
    cursor = conn.cursor()
    count = 0

    for from_node, to_node, attrs in graph.edges(data=True):
        from_entity = node_to_entity[from_node]
        to_entity = node_to_entity[to_node]
        rel_type = attrs.get('relationship_type', 'RELATED_TO')

        rel_id = relationship_map[(from_entity, to_entity, rel_type)]

        # Collect document-to-pages mapping
        doc_pages = {}

        # Handle source_doc (single document)
        if 'source_doc' in attrs:
            doc_name = attrs['source_doc']
            pages = attrs.get('pages', [])
            if doc_name:
                doc_pages[doc_name] = pages

        # Handle source_docs (multiple documents)
        if 'source_docs' in attrs:
            docs = attrs['source_docs']
            pages = attrs.get('pages', [])
            if isinstance(docs, list):
                for doc_name in docs:
                    if doc_name:
                        doc_pages[doc_name] = pages

        # Insert occurrences
        for doc_name, pages in doc_pages.items():
            doc_id = generate_document_hash(doc_name)
            pages_json = json.dumps(sorted(list(set(pages)))) if pages else None

            cursor.execute("""
                INSERT OR REPLACE INTO relationship_occurrences (
                    relationship_id, document_id, page_numbers
                ) VALUES (?, ?, ?)
            """, (rel_id, doc_id, pages_json))
            count += 1

    conn.commit()
    logger.info(f"Inserted {count} relationship occurrences")


def convert_graph_to_sql(graph_path: Path, db_path: Path):
    """
    Main conversion function: Load graph and convert to SQL database.

    Args:
        graph_path: Path to .gpickle graph file
        db_path: Path to output SQLite database
    """
    logger.info("=" * 80)
    logger.info("KNOWLEDGE GRAPH TO SQL CONVERTER")
    logger.info("=" * 80)

    # Load graph
    logger.info(f"Loading graph from: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Create database
    logger.info(f"Creating database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        # Create schema
        create_schema(conn)

        # Extract and insert documents
        logger.info("\n[Step 1/5] Extracting documents...")
        documents = extract_documents_from_graph(graph)
        insert_documents(conn, documents)

        # Insert entities
        logger.info("\n[Step 2/5] Inserting entities...")
        node_to_entity = insert_entities(conn, graph)

        # Insert entity occurrences
        logger.info("\n[Step 3/5] Inserting entity occurrences...")
        insert_entity_occurrences(conn, graph, node_to_entity)

        # Insert relationships
        logger.info("\n[Step 4/5] Inserting relationships...")
        relationship_map = insert_relationships(conn, graph, node_to_entity)

        # Insert relationship occurrences
        logger.info("\n[Step 5/5] Inserting relationship occurrences...")
        insert_relationship_occurrences(conn, graph, node_to_entity, relationship_map)

        # Print statistics
        print_statistics(conn)

        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Conversion completed successfully!")
        logger.info(f"Database saved to: {db_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise

    finally:
        conn.close()


def print_statistics(conn: sqlite3.Connection):
    """Print database statistics."""
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    tables = [
        'documents',
        'entities',
        'relationships',
        'entity_occurrences',
        'relationship_occurrences'
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table.replace('_', ' ').title():<30} {count:>10,}")

    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Knowledge Graph to SQL Database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert latest LLM Full Context graph
  python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle

  # Convert specific graph to custom database
  python kg_to_sql.py --graph my_graph.gpickle --output my_kg.db

  # Use default output location
  python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_rswoosh_*.gpickle
        """
    )

    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='Path to .gpickle graph file (supports wildcards)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='knowledge_graph.db',
        help='Output SQLite database path (default: knowledge_graph.db)'
    )

    args = parser.parse_args()

    # Handle wildcards in graph path
    graph_path = Path(args.graph)
    if '*' in str(graph_path):
        # Glob pattern
        matches = sorted(graph_path.parent.glob(graph_path.name))
        if not matches:
            logger.error(f"No files found matching: {graph_path}")
            return 1
        graph_path = matches[-1]  # Use latest
        logger.info(f"Using latest graph: {graph_path.name}")

    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        return 1

    db_path = Path(args.output)

    # Warn if database already exists
    if db_path.exists():
        logger.warning(f"Database already exists: {db_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return 0
        db_path.unlink()

    try:
        convert_graph_to_sql(graph_path, db_path)
        return 0

    except KeyboardInterrupt:
        logger.warning("\nConversion interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
