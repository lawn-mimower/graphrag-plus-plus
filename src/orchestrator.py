"""
Query Orchestrator - Phase 2 of Bakasur System.

Maps Natural Language to SQL Schema using Gemini and Milvus.
Provides secure, schema-aware query processing with ACL enforcement.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from google import genai

from src.config import Config
from src.inference_engine import SecureGraphTraverser
from src.milvus_ingestion import MilvusIngestionEngine

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """
    Sophisticated Query Orchestrator that maps Natural Language to SQL Schema.

    Pipeline:
    1. Schema-Aware Intent Extraction (Gemini)
    2. Candidate Resolution (Milvus + SQL)
    3. Security Filtering (ACL Bouncer)
    4. Execution (Walker/Graph Traverser)
    """

    def __init__(
        self,
        db_path: str = "knowledge_graph.db",
        milvus_path: str = "./outputs/milvus_orchestrator.db",
        gemini_model: str = None
    ):
        """
        Initialize the Query Orchestrator.

        Args:
            db_path: Path to SQLite knowledge graph database
            milvus_path: Path to Milvus-lite database
            gemini_model: Gemini model name for intent extraction
        """
        self.db_path = Path(db_path)
        self.milvus_path = milvus_path

        # Validate database
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Initialize Gemini client
        self.gemini_model_name = gemini_model or Config.MODEL_HEAVY
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

        # Initialize Milvus engine
        self.milvus_engine = MilvusIngestionEngine(
            db_path=str(self.db_path),
            milvus_path=self.milvus_path
        )

        # Initialize Graph Traverser (Walker)
        self.traverser = SecureGraphTraverser(db_path=str(self.db_path))

        logger.info(f"QueryOrchestrator initialized with model: {self.gemini_model_name}")

    def process_query(
        self,
        user_query: str,
        user_tags: List[str],
        max_depth: int = 2,
        similarity_threshold: float = 0.7,
        max_candidates_per_entity: int = 5,
        max_type_results: int = 20
    ) -> Dict[str, Any]:
        """
        Process a natural language query through the orchestration pipeline.

        Args:
            user_query: Natural language query from user
            user_tags: User's access tags (e.g., ["HR", "FINANCE"])
            max_depth: Maximum graph traversal depth
            similarity_threshold: Minimum similarity score for Milvus matches
            max_candidates_per_entity: Top-K results from Milvus per entity
            max_type_results: Maximum entities to fetch per entity_type

        Returns:
            Dictionary with keys:
            - status: "success" | "ambiguous" | "no_access" | "error"
            - intent: Extracted intent (if successful)
            - candidates: Candidate entities found (if any)
            - nodes: Node details with attributes (if successful)
            - mini_graph: Graph edges (if successful)
            - citation_context: Document citations (if successful)
            - message: Human-readable message
        """
        logger.info("=" * 80)
        logger.info(f"Processing query: {user_query}")
        logger.info(f"User tags: {user_tags}")
        logger.info("=" * 80)

        try:
            # STEP 1: Schema-Aware Intent Extraction
            logger.info("STEP 1: Schema-Aware Intent Extraction")
            intent = self._extract_intent(user_query)

            if not intent or self._is_intent_empty(intent):
                logger.warning("Intent extraction failed or returned empty results")
                return {
                    "status": "ambiguous",
                    "intent": intent,
                    "message": "Could not understand the query. Please be more specific about entities, types, or relationships."
                }

            logger.info(f"Extracted intent: {json.dumps(intent, indent=2)}")

            # STEP 2: Candidate Resolution
            logger.info("STEP 2: Candidate Resolution (Milvus + SQL)")
            candidate_ids = self._resolve_candidates(
                intent,
                similarity_threshold,
                max_candidates_per_entity,
                max_type_results
            )

            if not candidate_ids:
                logger.warning("No candidates found matching the query")
                return {
                    "status": "ambiguous",
                    "intent": intent,
                    "candidates": [],
                    "message": "No entities found matching your query. Try different search terms."
                }

            logger.info(f"Found {len(candidate_ids)} candidate entities")

            # STEP 3: Security Filtering
            logger.info("STEP 3: Security Filtering (ACL Bouncer)")
            valid_ids = self._apply_security_filter(candidate_ids, user_tags)

            if not valid_ids:
                logger.warning("All candidates filtered out by ACL")

                # Get candidate names for better error message
                candidate_names = self._get_entity_names(candidate_ids[:5])

                return {
                    "status": "no_access",
                    "intent": intent,
                    "candidates": candidate_names,
                    "message": f"Found {len(candidate_ids)} entities but you don't have access to them. "
                               f"Candidates: {', '.join(candidate_names[:3])}..."
                }

            logger.info(f"Security filter passed: {len(valid_ids)} valid entities")

            # STEP 4: Execution (Graph Traversal)
            logger.info("STEP 4: Execution (Walker/Graph Traverser)")
            context = self.traverser.get_context(
                start_entity_ids=valid_ids,
                user_tags=user_tags,
                max_depth=max_depth
            )

            # Get candidate details for response
            candidate_details = self._get_candidate_details(valid_ids)

            logger.info(f"Query processing complete: {len(context.get('nodes', {}))} nodes, "
                       f"{len(context['mini_graph'])} edges, "
                       f"{len(context['citation_context'])} documents")

            return {
                "status": "success",
                "intent": intent,
                "candidates": candidate_details,
                "nodes": context.get('nodes', {}),
                "mini_graph": context['mini_graph'],
                "citation_context": context['citation_context'],
                "message": f"Found {len(valid_ids)} entities, {len(context.get('nodes', {}))} nodes, and {len(context['mini_graph'])} relationships."
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }

    def _get_schema_summary(self) -> str:
        """
        Get a summary of available entity types and relationship types from SQL.

        Returns:
            Formatted string with schema information
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get distinct entity types
            cursor.execute("""
                SELECT DISTINCT entity_type
                FROM entities
                ORDER BY entity_type
            """)
            entity_types = [row[0] for row in cursor.fetchall()]

            # Get distinct relationship types
            cursor.execute("""
                SELECT DISTINCT relationship_type
                FROM relationships
                ORDER BY relationship_type
            """)
            relationship_types = [row[0] for row in cursor.fetchall()]

            # Format as string
            schema_summary = f"""Available Entity Types: {entity_types}
Available Relationship Types: {relationship_types}"""

            return schema_summary

        finally:
            conn.close()

    def _extract_intent(self, user_query: str) -> Optional[Dict[str, List[str]]]:
        """
        Extract intent from user query using Gemini with schema awareness.

        Args:
            user_query: Natural language query

        Returns:
            Dictionary with keys: entities, entity_types, relationship_types
            Returns None if extraction fails
        """
        schema_summary = self._get_schema_summary()

        prompt = f"""You are a query intent analyzer for a knowledge graph system.

SCHEMA INFORMATION:
{schema_summary}

USER QUERY:
{user_query}

TASK:
Analyze the user query and extract items that match our schema. Output a JSON object with these fields:
- "entities": List of specific named entities mentioned (e.g., ["John Smith", "Acme Corp"])
- "entity_types": List of entity types to search for (must match available types)
- "relationship_types": List of relationship types mentioned (must match available types)

RULES:
1. Only include entity_types and relationship_types that EXACTLY match the available types
2. Extract all specific entity names mentioned in the query
3. If you cannot find clear matches, return empty lists
4. Output ONLY valid JSON, no additional text

EXAMPLE OUTPUT:
{{
  "entities": ["John Smith", "TechCorp"],
  "entity_types": ["Person", "Project"],
  "relationship_types": ["WORKS_ON", "MANAGES"]
}}

Now analyze the query and output JSON:"""

        try:
            response = self.client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for structured extraction
                    max_output_tokens=8192,
                )
            )

            # Check for valid response
            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response from Gemini")
                return None

            response_text = response.text

            if not response_text:
                logger.error("Gemini returned empty response")
                return None

            response_text = response_text.strip()

            # Try to extract JSON from response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            intent = json.loads(response_text)

            # Validate structure
            if not isinstance(intent, dict):
                logger.error("Intent is not a dictionary")
                return None

            # Ensure required keys exist
            intent.setdefault('entities', [])
            intent.setdefault('entity_types', [])
            intent.setdefault('relationship_types', [])

            return intent

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error extracting intent: {e}", exc_info=True)
            return None

    def _is_intent_empty(self, intent: Dict[str, List[str]]) -> bool:
        """
        Check if intent has no useful information.

        Args:
            intent: Intent dictionary

        Returns:
            True if intent is empty/useless
        """
        return (
            not intent.get('entities') and
            not intent.get('entity_types') and
            not intent.get('relationship_types')
        )

    def _resolve_candidates(
        self,
        intent: Dict[str, List[str]],
        similarity_threshold: float,
        max_candidates_per_entity: int,
        max_type_results: int
    ) -> List[str]:
        """
        Resolve candidates using Milvus (for entities) and SQL (for types).

        Args:
            intent: Extracted intent
            similarity_threshold: Minimum similarity for Milvus
            max_candidates_per_entity: Top-K from Milvus
            max_type_results: Max results per entity type

        Returns:
            List of candidate sql_ids
        """
        candidate_ids = set()

        # Part A: Vector search for named entities
        for entity_name in intent.get('entities', []):
            try:
                matches = self.milvus_engine.search_similar_entities(
                    query_text=entity_name,
                    top_k=max_candidates_per_entity,
                    similarity_threshold=similarity_threshold
                )

                for match in matches:
                    candidate_ids.add(match['sql_id'])
                    logger.debug(
                        f"Milvus match: {entity_name} -> {match['canonical_name']} "
                        f"(score={match['similarity_score']})"
                    )

            except Exception as e:
                logger.error(f"Error searching for '{entity_name}': {e}")
                continue

        # Part B: SQL search for entity types
        for entity_type in intent.get('entity_types', []):
            try:
                type_ids = self._get_entities_by_type(entity_type, max_type_results)
                candidate_ids.update(type_ids)
                logger.debug(f"Type match: {entity_type} -> {len(type_ids)} entities")

            except Exception as e:
                logger.error(f"Error searching for type '{entity_type}': {e}")
                continue

        return list(candidate_ids)

    def _get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 20
    ) -> List[str]:
        """
        Get entity IDs by entity type from SQL.

        Args:
            entity_type: Entity type to search for
            limit: Maximum number of results

        Returns:
            List of entity IDs
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT unique_entity_id FROM entities WHERE entity_type = ? LIMIT ?",
                (entity_type, limit)
            )

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def _apply_security_filter(
        self,
        candidate_ids: List[str],
        user_tags: List[str]
    ) -> List[str]:
        """
        Filter candidates by ACL - only keep entities in accessible documents.

        Args:
            candidate_ids: List of candidate entity IDs
            user_tags: User's access tags

        Returns:
            List of valid entity IDs that pass ACL
        """
        if not candidate_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Generate placeholders
            id_placeholders = ', '.join('?' * len(candidate_ids))
            tag_placeholders = ', '.join('?' * len(user_tags))

            # Build query to check which entities are in accessible documents
            query = f"""
            SELECT DISTINCT eo.entity_id
            FROM entity_occurrences eo
            INNER JOIN documents d ON eo.document_id = d.document_id
            WHERE
                eo.entity_id IN ({id_placeholders})
                AND (
                    -- Always allow UNCLASSIFIED
                    d.access_tags LIKE '%"UNCLASSIFIED"%'
                    OR
                    -- Allow if any user tag matches
                    EXISTS (
                        SELECT 1
                        FROM json_each(d.access_tags) AS tag
                        WHERE tag.value IN ({tag_placeholders})
                    )
                )
            """

            params = tuple(candidate_ids) + tuple(user_tags)
            cursor.execute(query, params)

            valid_ids = [row[0] for row in cursor.fetchall()]

            logger.debug(
                f"ACL filter: {len(candidate_ids)} candidates -> {len(valid_ids)} valid"
            )

            return valid_ids

        finally:
            conn.close()

    def _get_entity_names(self, entity_ids: List[str]) -> List[str]:
        """
        Get canonical names for entity IDs.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of canonical names
        """
        if not entity_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            placeholders = ', '.join('?' * len(entity_ids))
            cursor.execute(
                f"SELECT canonical_name FROM entities WHERE unique_entity_id IN ({placeholders})",
                tuple(entity_ids)
            )

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def _get_candidate_details(self, entity_ids: List[str]) -> List[Dict[str, str]]:
        """
        Get detailed information about candidate entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of dictionaries with entity details
        """
        if not entity_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            placeholders = ', '.join('?' * len(entity_ids))
            cursor.execute(
                f"""
                SELECT unique_entity_id, entity_type, canonical_name
                FROM entities
                WHERE unique_entity_id IN ({placeholders})
                """,
                tuple(entity_ids)
            )

            candidates = []
            for row in cursor.fetchall():
                candidates.append({
                    'entity_id': row[0],
                    'entity_type': row[1],
                    'canonical_name': row[2]
                })

            return candidates

        finally:
            conn.close()

    def close(self):
        """Clean up resources."""
        if self.milvus_engine:
            self.milvus_engine.close()
        logger.info("QueryOrchestrator closed")


def create_orchestrator(
    db_path: str = "knowledge_graph.db",
    milvus_path: str = "./outputs/milvus_orchestrator.db"
) -> QueryOrchestrator:
    """
    Create and return a QueryOrchestrator instance.

    Args:
        db_path: Path to SQLite database
        milvus_path: Path to Milvus database

    Returns:
        Initialized QueryOrchestrator
    """
    return QueryOrchestrator(db_path=db_path, milvus_path=milvus_path)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("\n" + "=" * 80)
    print("QUERY ORCHESTRATOR - DEMO")
    print("=" * 80)

    # Initialize orchestrator
    orchestrator = QueryOrchestrator()

    # Example query
    test_query = "who is director a to director b?"
    test_tags = ["UNCLASSIFIED"]

    print(f"\nQuery: {test_query}")
    print(f"User Tags: {test_tags}\n")

    # Process query
    result = orchestrator.process_query(
        user_query=test_query,
        user_tags=test_tags,
        max_depth=2
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    # Cleanup
    orchestrator.close()
