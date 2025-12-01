"""
Secure Graph Traversal Engine for Inference System.

Provides ACL-enforced graph traversal with citation tracking.
Uses recursive SQL CTEs for optimal performance without NetworkX.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)


class SecureGraphTraverser:
    """
    Secure graph traversal engine with multi-layer ACL enforcement.

    Security Layers:
    1. Document ACL: Filter by access_tags
    2. Ghost Node Elimination: Only entities in accessible documents
    3. Edge Policy: Check relationship access_policy
    4. Node Security: Only traverse to grounded entities

    Features:
    - Bidirectional graph traversal (treats as undirected)
    - Cycle prevention via path tracking
    - Citation context with page numbers
    - Parameterized queries for SQL injection protection
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        """
        Initialize the secure graph traverser.

        Args:
            db_path: Path to SQLite knowledge graph database

        Raises:
            FileNotFoundError: If database doesn't exist
            ValueError: If database schema is invalid
        """
        self.db_path = Path(db_path)
        self._validate_database()
        logger.info(f"SecureGraphTraverser initialized with database: {self.db_path}")

    def _validate_database(self):
        """
        Validate that database exists and has required tables.

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If required tables are missing
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        required_tables = {
            'documents', 'entities', 'relationships',
            'entity_occurrences', 'relationship_occurrences'
        }

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}

            missing = required_tables - existing_tables
            if missing:
                raise ValueError(
                    f"Database missing required tables: {missing}"
                )

            logger.debug(f"Database validation passed: {len(existing_tables)} tables found")

        finally:
            conn.close()

    def get_context(
        self,
        start_entity_ids: List[str],
        user_tags: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Perform secure graph traversal and return mini-graph with citations.

        Args:
            start_entity_ids: Entity IDs to start traversal from
            user_tags: User's access tags (e.g., ["HR", "FINANCE"])
            max_depth: Maximum traversal depth (default: 2)

        Returns:
            Dictionary with three keys:
            - "nodes": Dictionary mapping entity_id -> node details with attributes
            - "mini_graph": List of edge dictionaries
            - "citation_context": Document name -> page numbers mapping

        Example:
            >>> traverser = SecureGraphTraverser()
            >>> result = traverser.get_context(
            ...     start_entity_ids=["entity_001"],
            ...     user_tags=["HR"],
            ...     max_depth=2
            ... )
            >>> result["nodes"]
            {
                "entity_001": {
                    "name": "John Smith",
                    "type": "Person",
                    "attributes": {"email": "john@example.com", "role": "Manager"}
                }
            }
            >>> result["mini_graph"]
            [
                {
                    "source": "entity_001",
                    "target": "entity_002",
                    "relation": "WORKS_FOR",
                    "properties": {"role": "admin"}
                }
            ]
        """
        if not start_entity_ids:
            logger.warning("No start_entity_ids provided, returning empty context")
            return {"nodes": {}, "mini_graph": [], "citation_context": {}}

        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")

        logger.info(
            f"Starting traversal: {len(start_entity_ids)} entities, "
            f"tags={user_tags}, max_depth={max_depth}"
        )

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Step 1: Execute recursive traversal
            edges = self._execute_traversal(
                cursor, start_entity_ids, user_tags, max_depth
            )

            logger.debug(f"Traversal found {len(edges)} edges")

            # Step 2: Format mini_graph
            mini_graph = self._format_mini_graph(edges)

            # Step 3: Collect all traversed entity IDs
            traversed_entities = set()
            for edge in edges:
                traversed_entities.add(edge[0])  # source_id
                traversed_entities.add(edge[2])  # target_id
            
            # Also include the start entities, in case they are isolated
            traversed_entities.update(start_entity_ids)

            # Step 4: Fetch node details with attributes
            node_details = self._get_node_details(
                cursor,
                list(traversed_entities)
            )

            # Step 5: Collect citations
            citation_context = self._collect_citations(
                cursor,
                list(traversed_entities),
                user_tags
            )

            logger.info(
                f"Context generated: {len(node_details)} nodes, {len(mini_graph)} edges, "
                f"{len(citation_context)} documents"
            )

            return {
                "nodes": node_details,
                "mini_graph": mini_graph,
                "citation_context": citation_context
            }

        finally:
            conn.close()


    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        user_tags: List[str]
    ) -> Dict[str, Any]:
        """
        Finds the shortest path between two entities using a secure,
        SQL-based breadth-first search.

        Args:
            source_id: The starting entity ID
            target_id: The target entity ID
            user_tags: User's access tags for ACL filtering

        Returns:
            A context dictionary in the same format as get_context,
            or an empty context if no path is found.
        """
        if not all([source_id, target_id]):
            logger.warning("Source or target ID missing, cannot find path.")
            return {"nodes": {}, "mini_graph": [], "citation_context": {}}

        if source_id == target_id:
            logger.warning("Source and target IDs are the same. Returning context for single node.")
            return self.get_context([source_id], user_tags, max_depth=0)

        logger.info(
            f"Finding shortest path: {source_id} -> {target_id}, "
            f"tags={user_tags}"
        )

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            query, params = self._build_shortest_path_query(source_id, target_id, user_tags)

            logger.debug(f"Executing shortest path query with {len(params)} parameters")
            cursor.execute(query, params)
            result = cursor.fetchone()

            if not result or not result[0]:
                logger.warning(f"No secure path found between {source_id} and {target_id}")
                return {"nodes": {}, "mini_graph": [], "citation_context": {}}

            path_str = result[0]
            path_nodes = [node for node in path_str.split('|') if node]

            logger.debug(f"Shortest path found: {' -> '.join(path_nodes)}")

            edges = self._get_path_edges(cursor, path_nodes)

            mini_graph = self._format_mini_graph(edges)
            node_details = self._get_node_details(cursor, path_nodes)
            citation_context = self._collect_citations(cursor, path_nodes, user_tags)

            logger.info(
                f"Path context generated: {len(node_details)} nodes, {len(mini_graph)} edges, "
                f"{len(citation_context)} documents"
            )

            return {
                "nodes": node_details,
                "mini_graph": mini_graph,
                "citation_context": citation_context
            }

        finally:
            conn.close()


    def _build_shortest_path_query(
        self,
        source_id: str,
        target_id: str,
        user_tags: List[str]
    ) -> Tuple[str, Tuple]:
        """
        Builds a recursive CTE query to find the shortest path via BFS.
        Returns the path as a delimited string.
        """
        tag_placeholders = ', '.join('?' * len(user_tags))
        params = tuple(user_tags) + (source_id, f'|{source_id}|', source_id, target_id)

        query = f"""
        WITH RECURSIVE allowed_docs AS (
            SELECT document_id FROM documents
            WHERE access_tags LIKE '%"UNCLASSIFIED"%' OR EXISTS (
                SELECT 1 FROM json_each(access_tags) AS tag WHERE tag.value IN ({tag_placeholders})
            )
        ),
        grounded_entities AS (
            SELECT DISTINCT eo.entity_id FROM entity_occurrences eo
            JOIN allowed_docs ad ON eo.document_id = ad.document_id
        ),
        bfs(current_node, path, depth) AS (
            -- Base case: start with the source node, ensuring it's grounded
            SELECT ?, ?, 0
            WHERE ? IN (SELECT entity_id FROM grounded_entities)

            UNION ALL

            -- Recursive step: explore neighbors
            SELECT
                CASE WHEN b.current_node = r.from_entity_id THEN r.to_entity_id ELSE r.from_entity_id END,
                b.path || CASE WHEN b.current_node = r.from_entity_id THEN r.to_entity_id ELSE r.from_entity_id END || '|',
                b.depth + 1
            FROM bfs b
            JOIN relationships r ON (b.current_node = r.from_entity_id OR b.current_node = r.to_entity_id)
            WHERE
                r.access_policy = 'OPEN'
                AND (CASE WHEN b.current_node = r.from_entity_id THEN r.to_entity_id ELSE r.from_entity_id END) IN (SELECT entity_id FROM grounded_entities)
                AND b.path NOT LIKE '%|' || CASE WHEN b.current_node = r.from_entity_id THEN r.to_entity_id ELSE r.from_entity_id END || '|%'
        )
        SELECT path FROM bfs WHERE current_node = ? ORDER BY depth ASC LIMIT 1;
        """
        return query, params


    def _get_path_edges(self, cursor: sqlite3.Cursor, path_nodes: List[str]) -> List[Tuple]:
        """
        Fetches all edges that constitute a given path.
        """
        if len(path_nodes) < 2:
            return []

        # Create placeholders for all nodes in the path to fetch their relationships at once
        node_pairs = []
        for i in range(len(path_nodes) - 1):
            node_pairs.append((path_nodes[i], path_nodes[i+1]))
            node_pairs.append((path_nodes[i+1], path_nodes[i]))

        placeholders = ', '.join(['(?, ?)'] * len(node_pairs))
        params = [item for pair in node_pairs for item in pair]

        query = f"""
            SELECT
                r.from_entity_id AS source_id,
                e1.canonical_name AS source_name,
                r.to_entity_id AS target_id,
                e2.canonical_name AS target_name,
                r.relationship_type AS relation,
                r.attributes AS properties_json
            FROM relationships r
            JOIN entities e1 ON r.from_entity_id = e1.unique_entity_id
            JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
            WHERE (r.from_entity_id, r.to_entity_id) IN ({placeholders})
        """
        cursor.execute(query, tuple(params))
        
        # We need to order the edges correctly according to the path
        edge_map = {(row[0], row[2]): row for row in cursor.fetchall()}
        
        ordered_edges = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge = edge_map.get((u, v)) or edge_map.get((v, u))
            if edge:
                ordered_edges.append(edge)
        
        return ordered_edges
    

    def _execute_traversal(
        self,
        cursor: sqlite3.Cursor,
        start_ids: List[str],
        user_tags: List[str],
        max_depth: int
    ) -> List[Tuple]:
        """
        Execute the recursive CTE traversal query.

        Args:
            cursor: Database cursor
            start_ids: Starting entity IDs
            user_tags: User access tags
            max_depth: Maximum depth

        Returns:
            List of tuples: (source_id, source_name, target_id, target_name,
                            relation, properties_json)
        """
        query, params = self._build_recursive_query(start_ids, user_tags, max_depth)

        logger.debug(f"Executing traversal query with {len(params)} parameters")
        cursor.execute(query, params)

        return cursor.fetchall()

    def _build_recursive_query(
        self,
        start_ids: List[str],
        user_tags: List[str],
        max_depth: int
    ) -> Tuple[str, Tuple]:
        """
        Build the recursive CTE query with parameterized placeholders.

        Args:
            start_ids: Starting entity IDs
            user_tags: User access tags
            max_depth: Maximum depth

        Returns:
            Tuple of (query_string, parameters_tuple)
        """
        # Generate placeholders for dynamic lists
        start_placeholders = ', '.join('?' * len(start_ids))
        tag_placeholders = ', '.join('?' * len(user_tags))

        # Build parameter tuple (order matters!)
        # Order: user_tags (allowed_docs), start_ids (base case), max_depth (recursive)
        params = tuple(user_tags) + tuple(start_ids) + (max_depth,)

        query = f"""
        WITH allowed_docs AS (
            -- CTE 1: Filter documents by ACL
            SELECT document_id, document_name
            FROM documents
            WHERE
                -- Always allow UNCLASSIFIED
                access_tags LIKE '%"UNCLASSIFIED"%'
                OR
                -- Allow if any user tag matches
                EXISTS (
                    SELECT 1
                    FROM json_each(access_tags) AS tag
                    WHERE tag.value IN ({tag_placeholders})
                )
        ),
        grounded_entities AS (
            -- CTE 2: Eliminate ghost nodes
            SELECT DISTINCT eo.entity_id
            FROM entity_occurrences eo
            INNER JOIN allowed_docs ad ON eo.document_id = ad.document_id
        ),
        graph_traversal AS (
            -- CTE 3: Recursive bidirectional traversal
            -- BASE CASE: Start nodes (if grounded)
            SELECT
                e.unique_entity_id AS current_entity,
                0 AS depth,
                '|' || e.unique_entity_id || '|' AS path
            FROM entities e
            WHERE
                e.unique_entity_id IN ({start_placeholders})
                AND e.unique_entity_id IN (SELECT entity_id FROM grounded_entities)

            UNION ALL

            -- RECURSIVE CASE: Bidirectional step
            SELECT
                CASE
                    WHEN gt.current_entity = r.from_entity_id THEN r.to_entity_id
                    ELSE r.from_entity_id
                END AS current_entity,
                gt.depth + 1 AS depth,
                gt.path ||
                    CASE
                        WHEN gt.current_entity = r.from_entity_id THEN r.to_entity_id
                        ELSE r.from_entity_id
                    END || '|' AS path
            FROM graph_traversal gt
            INNER JOIN relationships r ON (
                gt.current_entity = r.from_entity_id OR
                gt.current_entity = r.to_entity_id
            )
            WHERE
                gt.depth < ?  -- max_depth parameter
                AND r.access_policy = 'OPEN'  -- Edge security
                AND (
                    -- Node security: target must be grounded
                    CASE
                        WHEN gt.current_entity = r.from_entity_id THEN r.to_entity_id
                        ELSE r.from_entity_id
                    END
                ) IN (SELECT entity_id FROM grounded_entities)
                AND gt.path NOT LIKE '%|' ||
                    CASE
                        WHEN gt.current_entity = r.from_entity_id THEN r.to_entity_id
                        ELSE r.from_entity_id
                    END || '|%'  -- Cycle prevention
        )
        -- CTE 4: Extract mini-graph edges
        SELECT DISTINCT
            r.from_entity_id AS source_id,
            e1.canonical_name AS source_name,
            r.to_entity_id AS target_id,
            e2.canonical_name AS target_name,
            r.relationship_type AS relation,
            r.attributes AS properties_json
        FROM relationships r
        INNER JOIN graph_traversal gt1 ON r.from_entity_id = gt1.current_entity
        INNER JOIN graph_traversal gt2 ON r.to_entity_id = gt2.current_entity
        INNER JOIN entities e1 ON r.from_entity_id = e1.unique_entity_id
        INNER JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
        WHERE r.access_policy = 'OPEN'
        """

        return query, params

    def _format_mini_graph(self, edges: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Convert SQL result tuples to mini_graph format.

        Args:
            edges: List of tuples from query result

        Returns:
            List of edge dictionaries with parsed properties
        """
        mini_graph = []

        for edge in edges:
            source_id, source_name, target_id, target_name, relation, props_json = edge

            # Parse JSON properties
            try:
                properties = json.loads(props_json) if props_json else {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse properties for {source_id}->{target_id}: {e}"
                )
                properties = {}

            mini_graph.append({
                "source": source_id,
                "source_name": source_name,
                "target": target_id,
                "target_name": target_name,
                "relation": relation,
                "properties": properties
            })

        return mini_graph

    def _get_node_details(
        self,
        cursor: sqlite3.Cursor,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch full node details with attributes for given entity IDs.

        Args:
            cursor: Database cursor
            entity_ids: List of entity IDs to fetch details for

        Returns:
            Dictionary mapping entity_id -> node details
            Format: {
                "entity_id": {
                    "name": "canonical_name",
                    "type": "entity_type",
                    "attributes": {...}  # Parsed JSON attributes
                }
            }
        """
        if not entity_ids:
            return {}

        # Generate placeholders
        placeholders = ', '.join('?' * len(entity_ids))

        query = f"""
        SELECT
            unique_entity_id,
            entity_type,
            canonical_name,
            attributes,
            name_variants,
            cluster_size
        FROM entities
        WHERE unique_entity_id IN ({placeholders})
        """

        cursor.execute(query, tuple(entity_ids))
        rows = cursor.fetchall()

        node_details = {}
        for row in rows:
            entity_id, entity_type, canonical_name, attributes_json, name_variants_json, cluster_size = row

            # Parse JSON attributes
            try:
                attributes = json.loads(attributes_json) if attributes_json else {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse attributes for {entity_id}: {e}"
                )
                attributes = {}

            # Parse name variants
            try:
                name_variants = json.loads(name_variants_json) if name_variants_json else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse name_variants for {entity_id}: {e}"
                )
                name_variants = []

            node_details[entity_id] = {
                "name": canonical_name,
                "type": entity_type,
                "attributes": attributes,
                "name_variants": name_variants,
                "cluster_size": cluster_size
            }

        logger.debug(f"Fetched details for {len(node_details)} nodes")

        return node_details

    def _collect_citations(
        self,
        cursor: sqlite3.Cursor,
        entity_ids: List[str],
        user_tags: List[str]
    ) -> Dict[str, List[int]]:
        """
        Collect page citations from allowed documents.

        Combines pages from both entity_occurrences and relationship_occurrences.

        Args:
            cursor: Database cursor
            entity_ids: Entities in the traversed subgraph
            user_tags: User access tags

        Returns:
            Dictionary mapping document_name -> sorted list of page numbers
        """
        if not entity_ids:
            return {}

        # Generate placeholders
        entity_placeholders = ', '.join('?' * len(entity_ids))
        tag_placeholders = ', '.join('?' * len(user_tags))

        # Build parameters: user_tags (allowed_docs), entity_ids (entity occurrences),
        #                   entity_ids again (relationship from), entity_ids again (relationship to)
        params = tuple(user_tags) + tuple(entity_ids) + tuple(entity_ids) + tuple(entity_ids)

        query = f"""
        WITH allowed_docs AS (
            SELECT document_id, document_name
            FROM documents
            WHERE
                access_tags LIKE '%"UNCLASSIFIED"%'
                OR EXISTS (
                    SELECT 1 FROM json_each(access_tags) AS tag
                    WHERE tag.value IN ({tag_placeholders})
                )
        ),
        all_pages AS (
            -- Pages from entity occurrences
            SELECT
                ad.document_name,
                eo.page_numbers
            FROM entity_occurrences eo
            INNER JOIN allowed_docs ad ON eo.document_id = ad.document_id
            WHERE eo.entity_id IN ({entity_placeholders})

            UNION ALL

            -- Pages from relationship occurrences
            SELECT
                ad.document_name,
                ro.page_numbers
            FROM relationship_occurrences ro
            INNER JOIN allowed_docs ad ON ro.document_id = ad.document_id
            WHERE ro.relationship_id IN (
                SELECT relationship_id
                FROM relationships
                WHERE from_entity_id IN ({entity_placeholders})
                   OR to_entity_id IN ({entity_placeholders})
            )
        )
        SELECT document_name, page_numbers
        FROM all_pages
        WHERE page_numbers IS NOT NULL
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Aggregate pages per document
        citations = {}
        for doc_name, pages_json in rows:
            if not pages_json:
                continue

            try:
                pages = json.loads(pages_json)
                if not isinstance(pages, list):
                    logger.warning(
                        f"Invalid page_numbers format in {doc_name}: {pages_json}"
                    )
                    continue

                if doc_name not in citations:
                    citations[doc_name] = set()

                citations[doc_name].update(pages)

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse page_numbers for {doc_name}: {e}"
                )
                continue

        # Convert sets to sorted lists
        return {
            doc: sorted(list(pages))
            for doc, pages in citations.items()
        }


# Utility function for convenience
def create_traverser(db_path: str = "knowledge_graph.db") -> SecureGraphTraverser:
    """
    Create and return a SecureGraphTraverser instance.

    Args:
        db_path: Path to knowledge graph database

    Returns:
        Initialized SecureGraphTraverser
    """
    return SecureGraphTraverser(db_path)
