-- Knowledge Graph SQL Schema
-- 5-table normalized design for fast entity and relationship lookups
-- Database: SQLite 3.37+

-- ============================================================================
-- 1. DOCUMENTS TABLE
-- Stores information about source documents
-- ============================================================================
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,          -- MD5 hash of document name
    document_name TEXT NOT NULL UNIQUE,    -- Original document name
    file_path TEXT,                        -- Path to source PDF
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_doc_name ON documents(document_name);

-- ============================================================================
-- 2. ENTITIES TABLE
-- Stores deduplicated entities with all attributes
-- ============================================================================
CREATE TABLE IF NOT EXISTS entities (
    unique_entity_id TEXT PRIMARY KEY,     -- MD5 hash(canonical_name|entity_type)
    entity_type TEXT NOT NULL,             -- Person, Company, Organization, etc.
    canonical_name TEXT NOT NULL,          -- Golden truth name
    name_variants TEXT,                    -- JSON array of name variations
    cluster_size INTEGER DEFAULT 1,        -- Number of entities merged
    attributes TEXT,                       -- JSON blob of all domain attributes
    merge_reasoning TEXT,                  -- JSON blob of LLM reasoning (if available)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_cluster_size ON entities(cluster_size);

-- ============================================================================
-- 3. RELATIONSHIPS TABLE
-- Stores relationships between entities
-- ============================================================================
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,      -- MD5 hash(from_id|to_id|type)
    relationship_type TEXT NOT NULL,       -- DIRECTOR_OF, AUDITED_BY, etc.
    from_entity_id TEXT NOT NULL,          -- Source entity
    to_entity_id TEXT NOT NULL,            -- Target entity
    attributes TEXT,                       -- JSON blob of relationship attributes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_entity_id) REFERENCES entities(unique_entity_id) ON DELETE CASCADE,
    FOREIGN KEY (to_entity_id) REFERENCES entities(unique_entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_from_to ON relationships(from_entity_id, to_entity_id);

-- ============================================================================
-- 4. ENTITY_OCCURRENCES TABLE (Junction Table)
-- Tracks which entities appear in which documents and on which pages
-- ============================================================================
CREATE TABLE IF NOT EXISTS entity_occurrences (
    entity_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page_numbers TEXT NOT NULL,            -- JSON array [1, 5, 10]
    PRIMARY KEY (entity_id, document_id),
    FOREIGN KEY (entity_id) REFERENCES entities(unique_entity_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eo_entity ON entity_occurrences(entity_id);
CREATE INDEX IF NOT EXISTS idx_eo_document ON entity_occurrences(document_id);

-- ============================================================================
-- 5. RELATIONSHIP_OCCURRENCES TABLE (Junction Table)
-- Tracks which relationships appear in which documents and on which pages
-- ============================================================================
CREATE TABLE IF NOT EXISTS relationship_occurrences (
    relationship_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page_numbers TEXT,                     -- JSON array (may be NULL)
    PRIMARY KEY (relationship_id, document_id),
    FOREIGN KEY (relationship_id) REFERENCES relationships(relationship_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ro_rel ON relationship_occurrences(relationship_id);
CREATE INDEX IF NOT EXISTS idx_ro_document ON relationship_occurrences(document_id);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Entity with all occurrence details
CREATE VIEW IF NOT EXISTS v_entity_details AS
SELECT
    e.unique_entity_id,
    e.entity_type,
    e.canonical_name,
    e.cluster_size,
    e.attributes,
    e.merge_reasoning,
    GROUP_CONCAT(DISTINCT d.document_name) AS documents,
    COUNT(DISTINCT eo.document_id) AS document_count
FROM entities e
LEFT JOIN entity_occurrences eo ON e.unique_entity_id = eo.entity_id
LEFT JOIN documents d ON eo.document_id = d.document_id
GROUP BY e.unique_entity_id;

-- View: Relationship with entity names
CREATE VIEW IF NOT EXISTS v_relationship_details AS
SELECT
    r.relationship_id,
    r.relationship_type,
    e1.canonical_name AS from_entity_name,
    e1.entity_type AS from_entity_type,
    e2.canonical_name AS to_entity_name,
    e2.entity_type AS to_entity_type,
    r.attributes,
    GROUP_CONCAT(DISTINCT d.document_name) AS documents
FROM relationships r
JOIN entities e1 ON r.from_entity_id = e1.unique_entity_id
JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
LEFT JOIN relationship_occurrences ro ON r.relationship_id = ro.relationship_id
LEFT JOIN documents d ON ro.document_id = d.document_id
GROUP BY r.relationship_id;

-- ============================================================================
-- STATISTICS QUERIES
-- ============================================================================

-- Uncomment to see statistics after data load:
-- SELECT 'Documents' AS table_name, COUNT(*) AS count FROM documents
-- UNION ALL
-- SELECT 'Entities', COUNT(*) FROM entities
-- UNION ALL
-- SELECT 'Relationships', COUNT(*) FROM relationships
-- UNION ALL
-- SELECT 'Entity Occurrences', COUNT(*) FROM entity_occurrences
-- UNION ALL
-- SELECT 'Relationship Occurrences', COUNT(*) FROM relationship_occurrences;
