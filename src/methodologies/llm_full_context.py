"""
Full-Context LLM Deduplication using Gemini 2.5 Pro.
Sends entire entity extraction JSON to identify duplicates in one shot.
"""

import logging
from typing import Set, List, Dict, Any
import networkx as nx
from google import genai
import json

from src.config import Config

logger = logging.getLogger(__name__)


class LLMFullContextDeduplicator:
    """
    Full-context LLM deduplication using Gemini 2.5 Pro.
    Sends all entities and relationships at once for holistic analysis.
    """

    def __init__(self, client: genai.Client = None):
        """
        Initialize LLM Full-Context deduplicator.

        Args:
            client: Google GenAI client
        """
        self.client = client or genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model_name = Config.MODEL_DEDUP  # Pro model by default for complex reasoning

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform full-context LLM deduplication on the knowledge graph.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        logger.info("Running Full-Context LLM (Gemini 2.5 Pro) deduplication")

        # Convert graph to JSON format
        entities_json = self._graph_to_json(graph)

        # Build the deduplication prompt
        prompt = self._build_deduplication_prompt(entities_json)

        # Call Gemini
        logger.info("Sending full context to Gemini 2.5 Pro...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=16384,  # Large output for detailed analysis
                )
            )

            response_text = response.text

            if not response_text:
                logger.error("Empty response from Gemini")
                return []

            logger.debug(f"Received response ({len(response_text)} chars)")

            # Parse the response JSON
            result = self._parse_deduplication_response(response_text)

            # Convert to cluster format
            clusters = self._result_to_clusters(result)

            logger.info(f"Found {len(clusters)} duplicate clusters using full-context LLM")

            return clusters

        except Exception as e:
            logger.error(f"Error in LLM deduplication: {e}", exc_info=True)
            return []

    def _graph_to_json(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Convert NetworkX graph to JSON format for LLM.

        Args:
            graph: Knowledge graph

        Returns:
            Dictionary with entities and relationships
        """
        entities = []
        relationships = []

        # Extract entities
        for node_id, attrs in graph.nodes(data=True):
            entity = {
                "id": node_id,
                "type": attrs.get("type", "Unknown"),
                "attributes": {k: v for k, v in attrs.items() if k not in ["type"]}
            }
            entities.append(entity)

        # Extract relationships
        for source, target, attrs in graph.edges(data=True):
            relationship = {
                "from_id": source,
                "to_id": target,
                "type": attrs.get("type", "RELATED_TO"),
                "attributes": {k: v for k, v in attrs.items() if k not in ["type"]}
            }
            relationships.append(relationship)

        return {
            "entities": entities,
            "relationships": relationships
        }

    def _build_deduplication_prompt(self, entities_json: Dict[str, Any]) -> str:
        """
        Build the prompt for full-context deduplication.

        Args:
            entities_json: JSON containing entities and relationships

        Returns:
            Prompt string
        """
        json_str = json.dumps(entities_json, indent=2, ensure_ascii=False)

        prompt = f"""You are an expert in entity resolution and deduplication. Your task is to identify duplicate entities in the following knowledge graph extracted from financial documents.

KNOWLEDGE GRAPH DATA:
```json
{json_str}
```

INSTRUCTIONS:
1. Analyze ALL entities and their relationships carefully
2. Identify entities with the same TYPE that represent the same real-world entity
3. Consider ALL attributes (names, IDs, numbers, addresses, etc.)
4. Consider relationships to other entities as evidence
5. Handle name variations, abbreviations, typos, and formatting differences
6. For each group of duplicates, provide detailed analysis and justification

IMPORTANT MATCHING CRITERIA:
- Unique identifiers (PAN, CIN, DIN, etc.) are strongest evidence
- Name similarity with context (role, company, address)
- Relationship patterns (same connections to other entities)
- Temporal consistency (dates, periods mentioned)

OUTPUT REQUIREMENTS:
Return a JSON object with this EXACT structure:

{{
  "duplicates": [
    {{
      "cluster_id": 1,
      "entities": ["entity_id_1", "entity_id_2", "entity_id_3"],
      "analysis": "Detailed explanation of why these are duplicates, citing specific attributes and relationships",
      "deduplication_decision": "MERGE - entity_id_1, entity_id_2, and entity_id_3 represent [entity description]"
    }},
    {{
      "cluster_id": 2,
      "entities": ["entity_id_4", "entity_id_5"],
      "analysis": "...",
      "deduplication_decision": "MERGE - ..."
    }}
  ],
  "summary": {{
    "total_entities": <number>,
    "total_duplicates_found": <number>,
    "total_clusters": <number>
  }}
}}

CRITICAL RULES:
- Only include entities that are DEFINITELY duplicates (high confidence)
- Each entity can appear in only ONE cluster
- Provide mandatory "analysis" field with evidence
- Provide mandatory "deduplication_decision" field starting with "MERGE - "
- Return ONLY valid JSON, no markdown code blocks

Now analyze the knowledge graph and identify all duplicates:
"""
        return prompt

    def _parse_deduplication_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract deduplication results.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed deduplication result
        """
        try:
            # Remove markdown code blocks if present
            json_str = response_text.strip()

            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            # Parse JSON
            result = json.loads(json_str.strip())

            # Validate structure
            if "duplicates" not in result:
                logger.warning("Response missing 'duplicates' field")
                result["duplicates"] = []

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:1000]}...")

            # Return empty result
            return {
                "duplicates": [],
                "summary": {
                    "total_entities": 0,
                    "total_duplicates_found": 0,
                    "total_clusters": 0
                }
            }

    def _result_to_clusters(self, result: Dict[str, Any]) -> List[Set[str]]:
        """
        Convert LLM result to cluster format.

        Args:
            result: Parsed deduplication result

        Returns:
            List of duplicate clusters
        """
        clusters = []

        for duplicate_group in result.get("duplicates", []):
            entity_ids = duplicate_group.get("entities", [])

            if len(entity_ids) > 1:
                clusters.append(set(entity_ids))
            elif len(entity_ids) == 1:
                logger.warning(f"Cluster {duplicate_group.get('cluster_id')} has only 1 entity")

        return clusters

    def deduplicate_with_details(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Perform deduplication and return detailed results including analysis.

        Args:
            graph: Input knowledge graph

        Returns:
            Full deduplication result with analysis
        """
        logger.info("Running Full-Context LLM deduplication with detailed output")

        entities_json = self._graph_to_json(graph)
        prompt = self._build_deduplication_prompt(entities_json)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                )
            )

            response_text = response.text

            if not response_text:
                return {"duplicates": [], "summary": {}}

            result = self._parse_deduplication_response(response_text)

            return result

        except Exception as e:
            logger.error(f"Error in detailed deduplication: {e}", exc_info=True)
            return {"duplicates": [], "summary": {}}

    def get_methodology_name(self) -> str:
        """Get the name of this methodology."""
        return "LLM Full-Context"

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used by this methodology."""
        return {
            "methodology": "LLM Full-Context",
            "model": self.model_name,
            "temperature": 0.1,
            "approach": "single_shot_full_context",
            "max_output_tokens": 16384
        }


def create_llm_full_context_deduplicator() -> LLMFullContextDeduplicator:
    """
    Create and return a LLMFullContextDeduplicator instance.

    Returns:
        LLMFullContextDeduplicator instance
    """
    return LLMFullContextDeduplicator()
