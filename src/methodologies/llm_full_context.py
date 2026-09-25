"""
Full-Context LLM Deduplication using Gemini 2.5 Pro.
Sends entire entity extraction JSON to identify duplicates in one shot.
"""

import logging
from typing import Optional, Set, List, Dict, Any
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

    def __init__(self, client: genai.Client = None, batch_size: int = 30):
        """
        Initialize LLM Full-Context deduplicator.

        Args:
            client: Google GenAI client
        """
        self.client = client or genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model_name = Config.MODEL_DEDUP  # Pro model by default for complex reasoning
        self.batch_size = batch_size  # entities per LLM call; batches are formed within an entity type
        self.failed_batches = 0

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Perform LLM deduplication on the knowledge graph.

        Entities are sent to the model in batches of at most ``batch_size``, formed
        within each entity type (duplicates never span types). When a type needs more
        than one batch, a reconciliation pass is run over one representative per
        cluster so duplicates split across batches are still merged. Graphs with at
        most ``batch_size`` entities are sent in a single call, as before.

        Args:
            graph: Input knowledge graph

        Returns:
            List of duplicate clusters (sets of node IDs)
        """
        return self._result_to_clusters(self.deduplicate_with_details(graph))

    def _call_model(self, entities_json: Dict[str, Any], label: str) -> Dict[str, Any]:
        """One deduplication call over ``entities_json``; empty result on failure."""
        prompt = self._build_deduplication_prompt(entities_json)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=16384,
                )
            )
            response_text = getattr(response, "text", None) or ""
            if not response_text.strip():
                logger.error(f"Empty deduplication reply for {label}")
                self.failed_batches += 1
                return {"duplicates": []}
            result = self._parse_deduplication_response(response_text)
            if result is None:
                logger.error(f"Unparseable deduplication reply for {label}")
                self.failed_batches += 1
                return {"duplicates": []}
            return result
        except Exception as e:
            logger.error(f"Error in LLM deduplication for {label}: {e}", exc_info=True)
            self.failed_batches += 1
            return {"duplicates": []}

    @staticmethod
    def _subset(entities_json: Dict[str, Any], ids: Set[str]) -> Dict[str, Any]:
        return {
            "entities": [e for e in entities_json["entities"] if e["id"] in ids],
            "relationships": [r for r in entities_json["relationships"]
                              if r["from_id"] in ids and r["to_id"] in ids],
        }

    @staticmethod
    def _batches(items: List[Any], size: int) -> List[List[Any]]:
        return [items[k:k + size] for k in range(0, len(items), size)] or [[]]

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

    def _parse_deduplication_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM reply. Markdown fences are stripped; a reply that is not strict
        JSON (trailing commas, a truncated ending) is repaired with ``json_repair``.
        Returns None when no object can be recovered.
        """
        json_str = response_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        json_str = json_str.strip()
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Deduplication reply is not strict JSON ({e}); attempting repair")
            try:
                from json_repair import loads as repair_loads
                result = repair_loads(json_str)
            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                logger.debug(f"Response text: {response_text[:1000]}...")
                return None
        if not isinstance(result, dict):
            return None
        if not isinstance(result.get("duplicates"), list):
            logger.warning("Response missing 'duplicates' field")
            result["duplicates"] = []
        return result

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
            Full deduplication result: ``duplicates`` (cluster groups with the model's
            reasoning), ``summary`` and ``batches`` (number of LLM calls made)
        """
        entities_json = self._graph_to_json(graph)
        entities = entities_json["entities"]
        self.failed_batches = 0
        groups: List[Dict[str, Any]] = []
        calls = 0

        if len(entities) <= self.batch_size:
            logger.info(f"Running full-context LLM deduplication over {len(entities)} entities in one call")
            groups.extend(self._call_model(entities_json, "all entities").get("duplicates", []))
            calls += 1
        else:
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for e in entities:
                by_type.setdefault(e.get("type", "Unknown"), []).append(e)
            logger.info(f"Running batched LLM deduplication: {len(entities)} entities, "
                        f"{len(by_type)} types, batches of {self.batch_size}")
            for etype, members in by_type.items():
                batches = self._batches(members, self.batch_size)
                type_groups: List[Dict[str, Any]] = []
                for n, batch in enumerate(batches, 1):
                    ids = {e["id"] for e in batch}
                    result = self._call_model(self._subset(entities_json, ids), f"{etype} batch {n}/{len(batches)}")
                    calls += 1
                    type_groups.extend(g for g in result.get("duplicates", []) if len(g.get("entities", [])) > 1)
                if len(batches) > 1:
                    # Reconciliation: one representative per cluster plus the singletons, so
                    # duplicates that landed in different batches can still be merged. Repeated
                    # until the representatives fit in one call or a pass finds nothing new.
                    for round_no in range(1, 4):
                        clustered = {eid for g in type_groups for eid in g["entities"]}
                        reps = {g["entities"][0] for g in type_groups} | {e["id"] for e in members if e["id"] not in clustered}
                        rep_batches = self._batches(sorted(reps), self.batch_size)
                        found_before = len(type_groups)
                        for n, batch in enumerate(rep_batches, 1):
                            result = self._call_model(self._subset(entities_json, set(batch)),
                                                      f"{etype} reconciliation {round_no}.{n}")
                            calls += 1
                            type_groups.extend(g for g in result.get("duplicates", []) if len(g.get("entities", [])) > 1)
                        if len(rep_batches) == 1 or len(type_groups) == found_before:
                            break
                        type_groups = self._merge_groups(type_groups)
                groups.extend(type_groups)

        merged = self._merge_groups(groups)
        return {
            "duplicates": merged,
            "summary": {
                "total_entities": len(entities),
                "total_duplicates_found": sum(len(g["entities"]) for g in merged),
                "total_clusters": len(merged),
            },
            "batches": calls,
            "failed_batches": self.failed_batches,
        }

    @staticmethod
    def _merge_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Union-find over the model's groups: a group from one call may overlap another's."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for g in groups:
            ids = [i for i in g.get("entities", []) if isinstance(i, str)]
            for other in ids[1:]:
                parent[find(ids[0])] = find(other)
        clusters: Dict[str, List[str]] = {}
        for eid in parent:
            clusters.setdefault(find(eid), []).append(eid)
        reasoning: Dict[str, Any] = {}
        for g in groups:
            for eid in g.get("entities", []):
                reasoning.setdefault(find(eid), {k: v for k, v in g.items() if k != "entities"})
        out = []
        for n, (root, members) in enumerate(sorted(clusters.items(), key=lambda kv: sorted(kv[1])), 1):
            if len(members) > 1:
                out.append({**reasoning.get(root, {}), "cluster_id": n, "entities": sorted(members)})
        return out

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
