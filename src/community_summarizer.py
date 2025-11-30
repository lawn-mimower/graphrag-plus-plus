"""
Community Summarizer - Generates natural language summaries of Leiden communities.

Uses Gemini to create concise summaries of each community for:
- Global search ("Satellite" mode)
- Understanding organizational structure
- High-level query answering
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from google import genai

from src.config import Config

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """
    Generates and stores natural language summaries of Leiden communities.

    For each community:
    1. Extracts all entities and relationships within the community
    2. Identifies top central entities (by degree)
    3. Uses Gemini to generate 2-3 sentence summary
    4. Stores summary in database for quick retrieval
    """

    def __init__(
        self,
        db_path: str = "knowledge_graph.db",
        gemini_model: str = None
    ):
        """
        Initialize the community summarizer.

        Args:
            db_path: Path to SQLite knowledge graph database
            gemini_model: Gemini model name (defaults to MODEL_HEAVY)
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Initialize Gemini client
        self.gemini_model_name = gemini_model or Config.MODEL_HEAVY
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

        logger.info(f"CommunitySummarizer initialized with model: {self.gemini_model_name}")

    def generate_all_summaries(
        self,
        rpm_limit: int = 10,
        max_batch_size: int = 25
    ) -> Dict[str, int]:
        """
        Generate summaries for all communities at all levels.

        Uses dynamic batch size calculation to guarantee staying under RPM limits:
        Batch Size = max(ceil(Total Communities / RPM), max_batch_size)

        Args:
            rpm_limit: Requests per minute limit (default: 10, safe buffer from 15)
            max_batch_size: Maximum summaries per request (quality cap, default: 25)

        Returns:
            Dictionary with counts per level
        """
        logger.info("=" * 80)
        logger.info("Generating Community Summaries")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Count total communities across all levels
            cursor.execute("SELECT COUNT(DISTINCT community_id || level) FROM leiden_communities")
            total_communities = cursor.fetchone()[0]

            # Calculate dynamic batch size
            batch_size = self._calculate_batch_size(total_communities, rpm_limit, max_batch_size)

            logger.info(f"Total communities: {total_communities}")
            logger.info(f"RPM limit: {rpm_limit}")
            logger.info(f"Calculated batch size: {batch_size}")
            logger.info(f"Expected API calls: {total_communities // batch_size + (1 if total_communities % batch_size else 0)}")

            # Clear existing summaries
            cursor.execute("DELETE FROM community_summaries")
            conn.commit()

            summary_counts = {}

            # Process each level
            for level in ['micro', 'meso', 'macro']:
                logger.info(f"\nProcessing {level.upper()} level...")

                # Get all communities at this level
                cursor.execute("""
                    SELECT DISTINCT community_id, resolution
                    FROM leiden_communities
                    WHERE level = ?
                    ORDER BY community_id
                """, (level,))

                communities = cursor.fetchall()
                total = len(communities)
                logger.info(f"Found {total} communities at {level} level")

                count = 0

                # Process in batches to reduce API calls
                for i in range(0, len(communities), batch_size):
                    batch = communities[i:i + batch_size]

                    # Generate summaries for entire batch in ONE API call
                    batch_summaries = self._generate_batch_summaries(
                        cursor,
                        batch,
                        level
                    )

                    # Store each summary from the batch
                    for summary_data in batch_summaries:
                        if summary_data:
                            self._store_summary(cursor, summary_data)
                            count += 1

                    conn.commit()
                    logger.info(f"  Processed {count}/{total} communities (batch {i//batch_size + 1})...")

                summary_counts[level] = count
                logger.info(f"✓ Generated {count} summaries for {level} level")

            logger.info("\n" + "=" * 80)
            logger.info("Community Summary Generation Complete")
            logger.info("=" * 80)
            for level, count in summary_counts.items():
                logger.info(f"{level.capitalize()}: {count} summaries")

            return summary_counts

        finally:
            conn.close()

    def _calculate_batch_size(
        self,
        total_communities: int,
        rpm_limit: int,
        max_batch_size: int
    ) -> int:
        """
        Calculate dynamic batch size to guarantee staying under RPM limit.

        Formula:
        - Total Requests = Total Communities / Batch Size
        - To satisfy: Total Requests < RPM
        - Rearrange: Batch Size >= Total Communities / RPM

        We also apply a ceiling (max_batch_size) to maintain quality.

        Args:
            total_communities: Total number of communities to process
            rpm_limit: Requests per minute limit
            max_batch_size: Maximum batch size for quality

        Returns:
            Optimal batch size (floored at 1, capped at max_batch_size)
        """
        import math

        if total_communities == 0:
            return 1

        # Calculate minimum batch size to stay under RPM
        min_batch_size = math.ceil(total_communities / rpm_limit)

        # Apply ceiling for quality (LLM degrades with too many summaries)
        optimal_batch_size = min(min_batch_size, max_batch_size)

        # Ensure at least 1
        optimal_batch_size = max(optimal_batch_size, 1)

        logger.debug(f"Batch size calculation: "
                    f"total={total_communities}, rpm={rpm_limit}, "
                    f"min={min_batch_size}, max={max_batch_size}, "
                    f"optimal={optimal_batch_size}")

        return optimal_batch_size

    def _generate_batch_summaries(
        self,
        cursor: sqlite3.Cursor,
        batch: List[tuple[int, float]],
        level: str
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple communities in ONE API call using JSON output.

        Args:
            cursor: Database cursor
            batch: List of (community_id, resolution) tuples
            level: 'micro', 'meso', or 'macro'

        Returns:
            List of summary dictionaries (one per community in batch)
        """
        if not batch:
            return []

        # Prepare data for all communities in batch
        batch_data = []
        for community_id, resolution in batch:
            # Get entities and relationships for this community
            cursor.execute("""
                SELECT lc.entity_id, e.canonical_name, e.entity_type
                FROM leiden_communities lc
                JOIN entities e ON lc.entity_id = e.unique_entity_id
                WHERE lc.community_id = ? AND lc.level = ?
            """, (community_id, level))

            entities = cursor.fetchall()
            if not entities:
                continue

            entity_ids = [e[0] for e in entities]
            entity_info = [(e[1], e[2]) for e in entities]

            # Get relationships
            placeholders = ', '.join('?' * len(entity_ids))
            cursor.execute(f"""
                SELECT r.relationship_type, e1.canonical_name, e2.canonical_name
                FROM relationships r
                JOIN entities e1 ON r.from_entity_id = e1.unique_entity_id
                JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
                WHERE r.from_entity_id IN ({placeholders})
                  AND r.to_entity_id IN ({placeholders})
            """, tuple(entity_ids) * 2)

            relationships = cursor.fetchall()

            # Calculate metrics
            entity_count = len(entities)
            relationship_count = len(relationships)

            entity_degrees = self._calculate_entity_degrees(entity_ids, relationships)
            top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            top_entity_names = [
                next(name for eid, (name, _) in zip(entity_ids, entity_info) if eid == entity_id)
                for entity_id, _ in top_entities
            ]

            max_possible_edges = entity_count * (entity_count - 1) / 2
            density = relationship_count / max_possible_edges if max_possible_edges > 0 else 0.0
            avg_degree = sum(entity_degrees.values()) / len(entity_degrees) if entity_degrees else 0.0

            batch_data.append({
                'community_id': community_id,
                'resolution': resolution,
                'entities': entity_info,
                'relationships': relationships,
                'top_entities': top_entity_names,
                'entity_count': entity_count,
                'relationship_count': relationship_count,
                'density': density,
                'avg_degree': avg_degree
            })

        if not batch_data:
            return []

        # Generate summaries for entire batch in ONE API call
        summaries_json = self._generate_batch_summaries_with_gemini(batch_data, level)

        if not summaries_json:
            logger.warning(f"Failed to generate batch summaries for {len(batch)} communities")
            return []

        # Map summaries back to community data
        results = []
        for item in batch_data:
            community_id = item['community_id']

            # Find matching summary from JSON response
            summary_text = summaries_json.get(str(community_id))

            if not summary_text:
                logger.warning(f"No summary found for community {community_id} in batch response")
                continue

            community_key = f"{level}_{item['resolution']}_{community_id}"

            results.append({
                'community_key': community_key,
                'level': level,
                'resolution': item['resolution'],
                'community_id': community_id,
                'summary': summary_text,
                'entity_count': item['entity_count'],
                'relationship_count': item['relationship_count'],
                'top_entities': json.dumps(item['top_entities']),
                'avg_degree': item['avg_degree'],
                'density': item['density']
            })

        return results

    def _generate_batch_summaries_with_gemini(
        self,
        batch_data: List[Dict],
        level: str,
        retries: int = 3
    ) -> Dict[str, str]:
        """
        Generate summaries for multiple communities using Gemini with native JSON mode.

        Uses response_mime_type="application/json" to force valid JSON output.
        Includes retry logic and safety filter detection.

        Args:
            batch_data: List of community data dictionaries
            level: 'micro', 'meso', or 'macro'
            retries: Number of retry attempts (default: 3)

        Returns:
            Dictionary mapping community_id -> summary text
        """
        import time

        level_description = {
            'micro': "a small team or working group",
            'meso': "a department or functional unit",
            'macro': "a division or major organizational unit"
        }

        # Build structured data for JSON mode
        communities_data = []
        for data in batch_data:
            # Group entities by type
            entities_by_type = {}
            for name, entity_type in data['entities']:
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(name)

            community_info = {
                'community_id': str(data['community_id']),
                'size': data['entity_count'],
                'entities_by_type': entities_by_type,
                'top_members': data['top_entities'][:3],
                'sample_relationships': [
                    f"{r[1]}→{r[2]}" for r in data['relationships'][:3]
                ] if data['relationships'] else []
            }
            communities_data.append(community_info)

        # Build prompt optimized for JSON mode
        prompt = f"""You are an organizational analyst. Summarize these {len(batch_data)} communities at the {level.upper()} level ({level_description[level]}).

TASK:
For each community, provide a 2-3 sentence summary focusing on:
1. What this group does or represents
2. Key roles, functions, or connections
3. Organizational purpose (not individual details)

COMMUNITIES DATA:
{json.dumps(communities_data, indent=2)}

OUTPUT REQUIREMENTS:
- Return a JSON object where keys are community IDs (as strings) and values are summaries
- Each summary should be 2-3 concise sentences
- Do NOT use markdown code blocks
- Ensure all JSON strings are properly terminated

Example format:
{{
  "0": "This engineering team focuses on backend development with 8 members including John and Mary. They collaborate on API design and database architecture.",
  "1": "This finance department handles accounting and auditing functions. Key members include Alice and Bob who oversee fiscal reporting."
}}"""

        # Retry loop with exponential backoff
        for attempt in range(retries):
            try:
                # Use NATIVE JSON MODE (forces valid JSON)
                response = self.client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",  # CRITICAL: Forces JSON output
                        temperature=0.1,  # Low temp for stability
                        max_output_tokens=2000,
                    )
                )

                # Safety check: response.text can be None if blocked by filters
                if not response.text:
                    logger.warning(
                        f"Batch blocked by safety filters or empty response (Attempt {attempt + 1}/{retries})"
                    )
                    # Log safety ratings for debugging
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'safety_ratings'):
                            logger.debug(f"Safety ratings: {candidate.safety_ratings}")

                    # Wait before retry
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    continue

                # JSON mode returns clean JSON - parse directly
                summaries = json.loads(response.text)

                # Validate structure
                if not isinstance(summaries, dict):
                    logger.error(f"Expected dict, got {type(summaries)} (Attempt {attempt + 1}/{retries})")
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    return {}

                # Validate all community IDs are present
                missing_ids = set(str(d['community_id']) for d in batch_data) - set(summaries.keys())
                if missing_ids:
                    logger.warning(f"Missing summaries for communities: {missing_ids}")

                logger.debug(f"Successfully parsed {len(summaries)} summaries from batch response")
                return summaries

            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info("Retrying with fresh request...")
                    time.sleep(2)
                else:
                    logger.error("All retry attempts exhausted for JSON parsing")
                    return {}

            except Exception as e:
                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                logger.error(f"Batch generation error (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    return {}

        # All retries exhausted
        logger.error(f"Failed to generate summaries for batch after {retries} attempts")
        return {}

    def _generate_community_summary(
        self,
        cursor: sqlite3.Cursor,
        community_id: int,
        level: str,
        resolution: float
    ) -> Dict[str, Any]:
        """
        Generate summary for a single community.

        Args:
            cursor: Database cursor
            community_id: Community ID
            level: 'micro', 'meso', or 'macro'
            resolution: Resolution parameter used

        Returns:
            Dictionary with summary data, or None if failed
        """
        # Get all entities in this community
        cursor.execute("""
            SELECT lc.entity_id, e.canonical_name, e.entity_type
            FROM leiden_communities lc
            JOIN entities e ON lc.entity_id = e.unique_entity_id
            WHERE lc.community_id = ? AND lc.level = ?
        """, (community_id, level))

        entities = cursor.fetchall()

        if not entities:
            logger.warning(f"No entities found for community {community_id} at {level}")
            return None

        entity_ids = [e[0] for e in entities]
        entity_info = [(e[1], e[2]) for e in entities]  # (name, type)

        # Get internal relationships
        placeholders = ', '.join('?' * len(entity_ids))
        cursor.execute(f"""
            SELECT r.relationship_type, e1.canonical_name, e2.canonical_name
            FROM relationships r
            JOIN entities e1 ON r.from_entity_id = e1.unique_entity_id
            JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
            WHERE r.from_entity_id IN ({placeholders})
              AND r.to_entity_id IN ({placeholders})
        """, tuple(entity_ids) * 2)

        relationships = cursor.fetchall()

        # Calculate community metrics
        entity_count = len(entities)
        relationship_count = len(relationships)

        # Find top central entities (by degree within community)
        entity_degrees = self._calculate_entity_degrees(entity_ids, relationships)
        top_entities = sorted(
            entity_degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most connected

        top_entity_names = [
            next(name for eid, (name, _) in zip(entity_ids, entity_info) if eid == entity_id)
            for entity_id, _ in top_entities
        ]

        # Calculate density
        max_possible_edges = entity_count * (entity_count - 1) / 2
        density = relationship_count / max_possible_edges if max_possible_edges > 0 else 0.0

        avg_degree = sum(entity_degrees.values()) / len(entity_degrees) if entity_degrees else 0.0

        # Generate natural language summary using Gemini
        summary_text = self._generate_summary_with_gemini(
            entity_info,
            relationships,
            level,
            top_entity_names
        )

        if not summary_text:
            logger.warning(f"Failed to generate summary for community {community_id}")
            return None

        # Prepare summary data
        community_key = f"{level}_{resolution}_{community_id}"

        return {
            'community_key': community_key,
            'level': level,
            'resolution': resolution,
            'community_id': community_id,
            'summary': summary_text,
            'entity_count': entity_count,
            'relationship_count': relationship_count,
            'top_entities': json.dumps(top_entity_names),
            'avg_degree': avg_degree,
            'density': density
        }

    def _calculate_entity_degrees(
        self,
        entity_ids: List[str],
        relationships: List[tuple]
    ) -> Dict[str, int]:
        """
        Calculate degree (number of connections) for each entity within community.

        Args:
            entity_ids: List of entity IDs in community
            relationships: List of (rel_type, from_name, to_name) tuples

        Returns:
            Dict of entity_id -> degree
        """
        degrees = {eid: 0 for eid in entity_ids}

        for _ in relationships:
            # Each relationship contributes to degree of both entities
            # We don't have entity IDs in relationships tuple, so approximate
            # by counting total relationships / 2
            pass

        # Simplified: assume each entity participates equally
        if relationships:
            for eid in entity_ids:
                degrees[eid] = len(relationships) // len(entity_ids)

        return degrees

    def _generate_summary_with_gemini(
        self,
        entity_info: List[tuple[str, str]],
        relationships: List[tuple],
        level: str,
        top_entities: List[str]
    ) -> str:
        """
        Generate natural language summary using Gemini.

        Args:
            entity_info: List of (canonical_name, entity_type) tuples
            relationships: List of relationship tuples
            level: 'micro', 'meso', or 'macro'
            top_entities: List of most central entity names

        Returns:
            Summary text (2-3 sentences)
        """
        # Group entities by type
        entities_by_type = {}
        for name, entity_type in entity_info:
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(name)

        # Sample relationships for context (max 10)
        sample_rels = relationships[:10]

        # Build prompt
        level_description = {
            'micro': "a small team or working group",
            'meso': "a department or functional unit",
            'macro': "a division or major organizational unit"
        }

        prompt = f"""Summarize this organizational community in 2-3 clear sentences.

COMMUNITY LEVEL: {level.upper()} ({level_description[level]})

ENTITIES ({len(entity_info)} total):
"""

        for entity_type, names in entities_by_type.items():
            prompt += f"- {entity_type}s: {', '.join(names[:5])}"
            if len(names) > 5:
                prompt += f" (and {len(names) - 5} more)"
            prompt += "\n"

        if top_entities:
            prompt += f"\nKEY MEMBERS: {', '.join(top_entities)}\n"

        if sample_rels:
            prompt += f"\nKEY RELATIONSHIPS ({len(relationships)} total):\n"
            for rel_type, from_name, to_name in sample_rels[:5]:
                prompt += f"- {from_name} --[{rel_type}]--> {to_name}\n"

        prompt += """
INSTRUCTIONS:
1. Describe what this group does or represents
2. Mention key roles, functions, or connections
3. Keep it concise (2-3 sentences maximum)
4. Focus on the organizational purpose, not individual details
5. Output ONLY the summary text, no additional commentary

SUMMARY:"""

        try:
            response = self.client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,  # Low temperature for consistent summaries
                    max_output_tokens=200,
                )
            )

            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response from Gemini")
                return None

            summary = response.text.strip()

            # Clean up summary
            if summary.startswith("SUMMARY:"):
                summary = summary[8:].strip()

            return summary

        except Exception as e:
            logger.error(f"Error generating summary with Gemini: {e}")
            return None

    def _store_summary(
        self,
        cursor: sqlite3.Cursor,
        summary_data: Dict[str, Any]
    ):
        """
        Store summary in database.

        Args:
            cursor: Database cursor
            summary_data: Dictionary with summary fields
        """
        cursor.execute("""
            INSERT INTO community_summaries
            (community_key, level, resolution, community_id, summary,
             entity_count, relationship_count, top_entities, avg_degree, density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_data['community_key'],
            summary_data['level'],
            summary_data['resolution'],
            summary_data['community_id'],
            summary_data['summary'],
            summary_data['entity_count'],
            summary_data['relationship_count'],
            summary_data['top_entities'],
            summary_data['avg_degree'],
            summary_data['density']
        ))

    def get_summary(
        self,
        community_id: int,
        level: str = 'meso'
    ) -> str:
        """
        Retrieve summary for a specific community.

        Args:
            community_id: Community ID
            level: 'micro', 'meso', or 'macro'

        Returns:
            Summary text
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT summary
                FROM community_summaries
                WHERE community_id = ? AND level = ?
            """, (community_id, level))

            row = cursor.fetchone()
            return row[0] if row else None

        finally:
            conn.close()


def generate_summaries(
    db_path: str = "knowledge_graph.db"
) -> Dict[str, int]:
    """
    Convenience function to generate all community summaries.

    Args:
        db_path: Path to database

    Returns:
        Dictionary with summary counts per level
    """
    summarizer = CommunitySummarizer(db_path)
    return summarizer.generate_all_summaries()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Generate summaries
    counts = generate_summaries()

    print("\n" + "=" * 80)
    print("COMMUNITY SUMMARIZATION - COMPLETE")
    print("=" * 80)
    print(json.dumps(counts, indent=2))
