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
    Sophisticated Query Orchestrator that maps Natural Language to a graph.
    It uses a "retrieve-then-reason" approach, first searching for candidate
    entities and then using an LLM to plan the graph traversal.

    New Pipeline:
    1. Broad-Phase Candidate Retrieval (Milvus)
    2. LLM-Powered Query Planning (Gemini) -> 'laser' or 'flashlight' mode
    3. Security Filtering (ACL Bouncer)
    4. Execution (SecureGraphTraverser -> find_shortest_path or get_context)
    """

    def __init__(
        self,
        db_path: str = "knowledge_graph.db",
        milvus_path: str = "./outputs/milvus_orchestrator.db",
        gemini_model: str = None
    ):
        """
        Initialize the Query Orchestrator.
        """
        self.db_path = Path(db_path)
        self.milvus_path = milvus_path

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.gemini_model_name = gemini_model or Config.MODEL_HEAVY
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.milvus_engine = MilvusIngestionEngine(
            db_path=str(self.db_path),
            milvus_path=self.milvus_path
        )
        self.traverser = SecureGraphTraverser(db_path=str(self.db_path))

        logger.info(f"QueryOrchestrator initialized with model: {self.gemini_model_name}")

    def process_query(
        self,
        user_query: str,
        user_tags: List[str],
        max_depth: int = 2,
        similarity_threshold: float = 0.1,
        max_candidates_per_entity: int = 5,
        max_type_results: int = 20
    ) -> Dict[str, Any]:
        """
        Process a natural language query using the retrieve-then-reason pipeline.

        similarity_threshold is a cosine similarity. Entities are embedded from
        their full SQL record, so short name queries typically score 0.1-0.3
        against the right entity; results are ranked, the threshold only drops
        clearly unrelated hits.
        """
        logger.info("=" * 80)
        logger.info(f"Processing query: {user_query}")
        logger.info(f"User tags: {user_tags}")
        logger.info("=" * 80)

        try:
            # STEP 1: Broad-Phase Candidate Retrieval (Milvus)
            logger.info("STEP 1: Broad-Phase Candidate Retrieval (Milvus)")
            pre_fetched_candidates = self.milvus_engine.search_similar_entities(
                query_text=user_query,
                top_k=10, # Fetch a broad set of initial candidates
                similarity_threshold=max(similarity_threshold - 0.1, 0.0) # Use a slightly lower threshold for broad phase
            )
            logger.info(f"Found {len(pre_fetched_candidates)} initial candidates via semantic search.")

            # STEP 2: LLM-Powered Query Planning
            logger.info("STEP 2: LLM-Powered Query Planning")
            intent = self._extract_intent(user_query, pre_fetched_candidates)

            if not intent or self._is_intent_empty(intent):
                logger.warning("Intent extraction failed or returned empty results")
                return {
                    "status": "ambiguous",
                    "intent": intent,
                    "message": "Could not understand the query. Please be more specific."
                }

            logger.info(f"LLM Query Plan: {json.dumps(intent, indent=2)}")

            # STEP 3 & 4: Execute based on query mode
            if intent.get("query_mode") == "laser":
                return self._execute_laser_mode(intent, user_tags)
            else: # Default to flashlight mode
                return self._execute_flashlight_mode(
                    intent, user_tags, max_depth, similarity_threshold,
                    max_candidates_per_entity, max_type_results
                )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _execute_laser_mode(self, intent: Dict, user_tags: List[str]) -> Dict[str, Any]:
        """Executor for 'laser' mode (shortest path)."""
        logger.info("Executing in LASER mode (shortest path).")
        source_id = intent.get("source_entity", {}).get("id")
        target_id = intent.get("target_entity", {}).get("id")

        if not source_id or not target_id:
            return {
                "status": "ambiguous", "intent": intent,
                "message": "LLM failed to identify a clear source or target entity for pathfinding."
            }

        # STEP 3: Security Filtering
        logger.info("STEP 3: Security Filtering (ACL Bouncer)")
        valid_ids = self._apply_security_filter([source_id, target_id], user_tags)
        
        if not {source_id, target_id}.issubset(set(valid_ids)):
            logger.warning(f"Source or target entity for path search failed security check.")
            return {
                "status": "no_access", "intent": intent,
                "message": "You do not have access to the source or target entity for the path query."
            }
        
        logger.info("Security filter passed for source and target.")

        # STEP 4: Execution
        logger.info(f"STEP 4: Traverser executing find_shortest_path({source_id}, {target_id})")
        context = self.traverser.find_shortest_path(
            source_id=source_id,
            target_id=target_id,
            user_tags=user_tags
        )
        
        candidate_details = self._get_candidate_details([source_id, target_id])

        if not context.get("mini_graph"):
             message = f"Successfully found entities, but no secure path exists between them."
        else:
             message = f"Found a path with {len(context['mini_graph'])} edges between the entities."

        return {
            "status": "success", "intent": intent, "candidates": candidate_details,
            "nodes": context.get('nodes', {}), "mini_graph": context['mini_graph'],
            "citation_context": context['citation_context'], "message": message
        }

    def _execute_flashlight_mode(self, intent: Dict, user_tags: List[str], max_depth: int,
                                 similarity_threshold: float, max_candidates_per_entity: int,
                                 max_type_results: int) -> Dict[str, Any]:
        """Executor for 'flashlight' mode (general traversal)."""
        logger.info("Executing in FLASHLIGHT mode (general traversal).")
        
        logger.info("STEP 2b: Candidate Resolution (Milvus + SQL)")
        candidate_ids = self._resolve_candidates(
            intent, similarity_threshold, max_candidates_per_entity, max_type_results
        )

        if not candidate_ids:
            return {
                "status": "ambiguous", "intent": intent, "candidates": [],
                "message": "No entities found matching your query. Try different search terms."
            }
        logger.info(f"Found {len(candidate_ids)} candidate entities.")

        # STEP 3: Security Filtering
        logger.info("STEP 3: Security Filtering (ACL Bouncer)")
        valid_ids = self._apply_security_filter(candidate_ids, user_tags)

        if not valid_ids:
            candidate_names = self._get_entity_names(candidate_ids[:5])
            return {
                "status": "no_access", "intent": intent, "candidates": candidate_names,
                "message": f"Found {len(candidate_ids)} potential entities but you don't have access. "
                           f"Candidates: {', '.join(candidate_names[:3])}..."
            }
        logger.info(f"Security filter passed: {len(valid_ids)} valid entities.")

        # STEP 4: Execution
        logger.info(f"STEP 4: Traverser executing get_context(depth={max_depth})")
        context = self.traverser.get_context(
            start_entity_ids=valid_ids, user_tags=user_tags, max_depth=max_depth
        )
        candidate_details = self._get_candidate_details(valid_ids)

        return {
            "status": "success", "intent": intent, "candidates": candidate_details,
            "nodes": context.get('nodes', {}), "mini_graph": context['mini_graph'],
            "citation_context": context['citation_context'],
            "message": f"Found {len(valid_ids)} entities, {len(context.get('nodes', {}))} nodes, and {len(context['mini_graph'])} relationships."
        }

    def _get_schema_summary(self) -> str:
        """Get a summary of available entity and relationship types from SQL."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT entity_type FROM entities ORDER BY entity_type")
            entity_types = [row[0] for row in cursor.fetchall()]
            cursor.execute("SELECT DISTINCT relationship_type FROM relationships ORDER BY relationship_type")
            relationship_types = [row[0] for row in cursor.fetchall()]
            return f"Available Entity Types: {entity_types}\nAvailable Relationship Types: {relationship_types}"
        finally:
            conn.close()
    
    def _format_candidates_for_prompt(self, candidates: List[Dict]) -> str:
        """Formats a list of candidate entities for inclusion in an LLM prompt."""
        if not candidates:
            return "No pre-fetched candidates found."
        
        lines = ["Pre-fetched candidates from semantic search:"]
        for cand in candidates:
            lines.append(f"- ID: {cand['sql_id']}, Name: \"{cand['canonical_name']}\", Type: {cand['entity_type']} (Score: {cand['similarity_score']:.2f})")
        return "\n".join(lines)

    def _extract_intent(self, user_query: str, pre_fetched_candidates: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Acts as a query planner, determining the mode ('laser' or 'flashlight')
        and extracting parameters using a list of pre-fetched candidates.
        """
        schema_summary = self._get_schema_summary()
        candidate_summary = self._format_candidates_for_prompt(pre_fetched_candidates)

        prompt = f"""You are a Query Planner for a knowledge graph. Your job is to analyze a user's query and decide the best way to traverse the graph.

There are two modes:
1.  **laser**: Use for "how is X related to Y?" or "find path between X and Y" questions. This mode requires exactly one `source_entity` and one `target_entity`.
2.  **flashlight**: Use for all other questions, like "who is X?", "what is X?", or "show me all companies". This mode explores around one or more entities.

You are given a list of pre-fetched candidates from a semantic search. You must use the IDs and Names from this list.

CONTEXT:
---
{schema_summary}
---
{candidate_summary}
---

USER QUERY:
"{user_query}"

TASK:
First, determine the query type. Look for keywords like "related to", "path between", "connection between", "how is... connected to". If the query structure involves two distinct entities and one of these connecting phrases, you MUST set `query_mode` to "laser". Otherwise, the mode is "flashlight".

Then, based on the mode, output a JSON object with your query plan.

1.  If `query_mode` is "laser":
    -   From the pre-fetched candidates, identify the best match for the TWO entities in the query.
    -   Populate `source_entity` and `target_entity` with their exact ID and Name from the candidate list.
    -   Do NOT use the `entities` or `entity_types` fields.

2.  If `query_mode` is "flashlight":
    -   Populate `entities` with any specific entity names mentioned.
    -   Populate `entity_types` and `relationship_types` if they are mentioned and match the schema.
    -   Do NOT use the `source_entity` or `target_entity` fields.

JSON OUTPUT RULES:
- Output ONLY a valid JSON object.
- For "laser" mode, the JSON must contain: `{{ "query_mode": "laser", "source_entity": {{ "id": "ID_of_source", "name": "Name of Source" }}, "target_entity": {{ "id": "ID_of_target", "name": "Name of Target" }} }}`
- For "flashlight" mode, the JSON must contain: `{{ "query_mode": "flashlight", "entities": ["Entity Name"], "entity_types": ["Type"], "relationship_types": ["Type"] }}`

Now, analyze the query and the provided context and generate the JSON plan.
"""
        logger.info(f"Full LLM planning prompt:\n{prompt}")
        try:
            response = self.client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192)
            )

            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            intent = json.loads(response_text)

            if not isinstance(intent, dict): return None

            if intent.get("query_mode") == "flashlight":
                intent.setdefault('entities', [])
                intent.setdefault('entity_types', [])
                intent.setdefault('relationship_types', [])

            return intent

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse or execute intent extraction: {e}", exc_info=True)
            return None

    def _is_intent_empty(self, intent: Dict[str, Any]) -> bool:
        """Check if the extracted intent is empty or unusable."""
        if not intent:
            return True
        mode = intent.get("query_mode")
        if mode == "laser":
            source = intent.get("source_entity", {})
            target = intent.get("target_entity", {})
            return not all([source.get("id"), target.get("id")])
        elif mode == "flashlight":
            return not any([
                intent.get('entities'),
                intent.get('entity_types'),
                intent.get('relationship_types')
            ])
        return True # Unrecognized mode is considered empty

    def _resolve_candidates(
        self,
        intent: Dict[str, List[str]],
        similarity_threshold: float,
        max_candidates_per_entity: int,
        max_type_results: int
    ) -> List[str]:
        """
        Resolve candidates using Milvus (for entities) and SQL (for types).
        Used only in 'flashlight' mode.
        """
        candidate_ids = set()
        for entity_name in intent.get('entities', []):
            try:
                matches = self.milvus_engine.search_similar_entities(
                    query_text=entity_name,
                    top_k=max_candidates_per_entity,
                    similarity_threshold=similarity_threshold
                )
                for match in matches:
                    candidate_ids.add(match['sql_id'])
            except Exception as e:
                logger.error(f"Error searching for '{entity_name}': {e}")
        
        for entity_type in intent.get('entity_types', []):
            try:
                type_ids = self._get_entities_by_type(entity_type, max_type_results)
                candidate_ids.update(type_ids)
            except Exception as e:
                logger.error(f"Error searching for type '{entity_type}': {e}")

        return list(candidate_ids)

    def _get_entities_by_type(self, entity_type: str, limit: int = 20) -> List[str]:
        """Get entity IDs by entity type from SQL."""
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

    def _apply_security_filter(self, candidate_ids: List[str], user_tags: List[str]) -> List[str]:
        """Filter candidates by ACL."""
        if not candidate_ids: return []
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Filter out potential None values from candidate_ids
            filtered_candidate_ids = [cid for cid in candidate_ids if cid]
            if not filtered_candidate_ids:
                return []
            
            id_placeholders = ', '.join('?' * len(filtered_candidate_ids))
            tag_placeholders = ', '.join('?' * len(user_tags))
            query = f"""
            SELECT DISTINCT eo.entity_id
            FROM entity_occurrences eo
            INNER JOIN documents d ON eo.document_id = d.document_id
            WHERE eo.entity_id IN ({id_placeholders}) AND (
                d.access_tags LIKE '%"UNCLASSIFIED"%' OR EXISTS (
                    SELECT 1 FROM json_each(d.access_tags) AS tag
                    WHERE tag.value IN ({tag_placeholders})
                )
            )
            """
            params = tuple(filtered_candidate_ids) + tuple(user_tags)
            cursor.execute(query, params)
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def _get_entity_names(self, entity_ids: List[str]) -> List[str]:
        """Get canonical names for entity IDs."""
        if not entity_ids: return []
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
        """Get detailed information about candidate entities."""
        if not entity_ids: return []
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            placeholders = ', '.join('?' * len(entity_ids))
            cursor.execute(
                f"SELECT unique_entity_id, entity_type, canonical_name FROM entities WHERE unique_entity_id IN ({placeholders})",
                tuple(entity_ids)
            )
            return [{'entity_id': row[0], 'entity_type': row[1], 'canonical_name': row[2]} for row in cursor.fetchall()]
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
    test_query = "who are the key investors in the picture?"
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