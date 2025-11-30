"""
Milvus Ingestion Script for Query Orchestrator.

Synchronizes SQL entities with Milvus vector database for semantic search.
Stores: sql_id, entity_type, canonical_name, and embedding vector.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

from src.config import Config

logger = logging.getLogger(__name__)


class MilvusIngestionEngine:
    """
    Ingests entities from SQL database into Milvus for semantic search.

    Schema Synchronization:
    - sql_id: Matches unique_entity_id from SQL (MD5 hash)
    - entity_type: Matches entity_type from SQL
    - canonical_name: Matches canonical_name from SQL
    - vector: Embedding of canonical_name
    """

    def __init__(
        self,
        db_path: str = "knowledge_graph.db",
        milvus_path: str = "./outputs/milvus_orchestrator.db",
        embedding_model: str = None,
        collection_name: str = "orchestrator_entities",
        auto_connect: bool = True
    ):
        """
        Initialize the Milvus ingestion engine.

        Args:
            db_path: Path to SQLite knowledge graph database
            milvus_path: Path to Milvus-lite database file
            embedding_model: Sentence-transformers model name
            collection_name: Milvus collection name
            auto_connect: If True, automatically connect to existing Milvus database
        """
        self.db_path = Path(db_path)
        self.milvus_path = milvus_path
        self.collection_name = collection_name

        # Initialize embedding model
        self.model_name = embedding_model or Config.SEMANTIC_EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Milvus client
        self.milvus_client = None

        # Auto-connect to existing Milvus database if it exists
        if auto_connect and Path(self.milvus_path).exists():
            self._connect_to_milvus()

        logger.info(f"Ingestion engine initialized (dim={self.embedding_dim})")

    def ingest_from_sql(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Ingest all entities from SQL database into Milvus.

        Args:
            batch_size: Number of entities to process per batch

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Milvus ingestion from SQL database")
        logger.info("=" * 80)

        # Step 1: Read entities from SQL
        entities = self._read_entities_from_sql()

        if not entities:
            logger.warning("No entities found in SQL database")
            return {"total_entities": 0, "ingested": 0}

        logger.info(f"Found {len(entities)} entities in SQL database")

        # Step 2: Initialize Milvus collection
        self._initialize_milvus_collection()

        # Step 3: Generate embeddings and ingest in batches
        total_ingested = 0

        for i in tqdm(range(0, len(entities), batch_size), desc="Ingesting batches"):
            batch = entities[i:i + batch_size]

            # Generate embeddings for batch
            texts = [entity['canonical_name'] for entity in batch]
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Prepare data for Milvus
            data = []
            for entity, embedding in zip(batch, embeddings):
                data.append({
                    "sql_id": entity['sql_id'],
                    "entity_type": entity['entity_type'],
                    "canonical_name": entity['canonical_name'],
                    "vector": embedding.tolist()
                })

            # Insert into Milvus
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=data
            )

            total_ingested += len(batch)

        logger.info(f"Successfully ingested {total_ingested} entities into Milvus")

        # Step 4: Create index for better search performance
        self._create_index()

        return {
            "total_entities": len(entities),
            "ingested": total_ingested,
            "collection_name": self.collection_name
        }

    def _read_entities_from_sql(self) -> List[Dict[str, str]]:
        """
        Read all entities from SQL database.

        Returns:
            List of entity dictionaries with sql_id, entity_type, canonical_name
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            query = """
            SELECT
                unique_entity_id,
                entity_type,
                canonical_name
            FROM entities
            ORDER BY unique_entity_id
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            entities = []
            for row in rows:
                entities.append({
                    'sql_id': row[0],
                    'entity_type': row[1],
                    'canonical_name': row[2]
                })

            return entities

        finally:
            conn.close()

    def _connect_to_milvus(self):
        """
        Connect to existing Milvus database without re-creating collection.
        """
        logger.info(f"Connecting to existing Milvus database: {self.milvus_path}")

        # Initialize client
        self.milvus_client = MilvusClient(self.milvus_path)

        # Check if collection exists
        if self.milvus_client.has_collection(self.collection_name):
            logger.info(f"Connected to existing collection: {self.collection_name}")
        else:
            logger.warning(f"Collection '{self.collection_name}' not found. Run ingest_from_sql() first.")
            self.milvus_client.close()
            self.milvus_client = None

    def _initialize_milvus_collection(self):
        """
        Initialize Milvus client and create collection with proper schema.
        """
        logger.info(f"Initializing Milvus collection: {self.collection_name}")

        # Initialize client
        self.milvus_client = MilvusClient(self.milvus_path)

        # Drop existing collection if it exists
        if self.milvus_client.has_collection(self.collection_name):
            logger.warning(f"Dropping existing collection: {self.collection_name}")
            self.milvus_client.drop_collection(self.collection_name)

        # Create collection with schema
        # Note: MilvusClient (lite) uses simplified schema creation
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",  # Cosine similarity
            primary_field_name="sql_id",  # Use sql_id as primary key
            id_type="string",  # Primary key is string type
            max_length=32,  # MD5 hash is 32 characters
        )

        logger.info(f"Collection created successfully")

    def _create_index(self):
        """
        Create index on vector field for efficient similarity search.
        """
        logger.info("Creating index for vector field...")

        # For Milvus-lite, index is automatically created
        # No explicit index creation needed with MilvusClient

        logger.info("Index creation completed")

    def search_similar_entities(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entities using semantic similarity.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching entities with sql_id, entity_type, canonical_name, score
        """
        if not self.milvus_client:
            raise RuntimeError("Milvus client not initialized. Call ingest_from_sql() first.")

        # Generate embedding for query
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]

        # Search in Milvus
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=["sql_id", "entity_type", "canonical_name"]
        )

        # Filter by similarity threshold and format results
        matches = []
        for hit in results[0]:
            # For COSINE metric, distance = 1 - cosine_similarity
            # So similarity = 1 - distance
            similarity = 1 - hit['distance']

            if similarity >= similarity_threshold:
                matches.append({
                    'sql_id': hit['entity']['sql_id'],
                    'entity_type': hit['entity']['entity_type'],
                    'canonical_name': hit['entity']['canonical_name'],
                    'similarity_score': round(similarity, 4)
                })

        return matches

    def close(self):
        """Close Milvus client connection."""
        if self.milvus_client:
            self.milvus_client.close()
            logger.info("Milvus client closed")


def run_ingestion(db_path: str = "knowledge_graph.db") -> Dict[str, int]:
    """
    Convenience function to run the ingestion process.

    Args:
        db_path: Path to SQLite database

    Returns:
        Ingestion statistics
    """
    engine = MilvusIngestionEngine(db_path=db_path)
    stats = engine.ingest_from_sql()
    engine.close()

    return stats


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run ingestion
    print("\nStarting Milvus ingestion...")
    stats = run_ingestion()

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"Total entities processed: {stats['total_entities']}")
    print(f"Successfully ingested: {stats['ingested']}")
    print(f"Collection name: {stats['collection_name']}")
    print("=" * 80)
