#!/bin/bash

# ============================================================================
# Initialize SQL Schema and Ingest into Milvus
# ============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "BAKASUR PHASE 2 - Database Initialization & Milvus Ingestion"
echo "================================================================================"
echo ""

# Configuration
DB_PATH="knowledge_graph.db"
SCHEMA_PATH="sql/schema.sql"

# Step 1: Initialize SQL Database
echo "Step 1: Initializing SQL Database..."
echo "Database path: $DB_PATH"
echo "Schema path: $SCHEMA_PATH"
echo ""

if [ ! -f "$SCHEMA_PATH" ]; then
    echo "ERROR: Schema file not found at $SCHEMA_PATH"
    exit 1
fi

# Create/recreate database with schema
echo "Creating tables from schema..."
sqlite3 "$DB_PATH" < "$SCHEMA_PATH"

if [ $? -eq 0 ]; then
    echo "✓ SQL tables created successfully"
else
    echo "✗ Failed to create SQL tables"
    exit 1
fi

# Verify tables were created
echo ""
echo "Verifying tables..."
TABLE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
echo "Found $TABLE_COUNT tables in database"

# List all tables
echo ""
echo "Tables in database:"
sqlite3 "$DB_PATH" "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"

# Check entity count
echo ""
ENTITY_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM entities;")
echo "Current entity count: $ENTITY_COUNT"

if [ "$ENTITY_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠ WARNING: No entities found in database!"
    echo "You need to populate the database with entities before running Milvus ingestion."
    echo ""
    echo "Skipping Milvus ingestion step."
    echo ""
    echo "To populate the database, run your entity extraction pipeline first."
    exit 0
fi

# Step 2: Ingest into Milvus
echo ""
echo "================================================================================"
echo "Step 2: Ingesting entities into Milvus..."
echo "================================================================================"
echo ""

python -m src.milvus_ingestion

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ INITIALIZATION AND INGESTION COMPLETE"
    echo "================================================================================"
    echo ""
    echo "Database ready at: $DB_PATH"
    echo "Milvus database at: ./outputs/milvus_orchestrator.db"
    echo ""
    echo "You can now use the Query Orchestrator:"
    echo "  python -m src.orchestrator"
else
    echo ""
    echo "✗ Milvus ingestion failed"
    exit 1
fi
