"""
Configuration module for Entity Resolution Benchmarking Suite.
Handles environment variables, model configuration, paths, and thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class for the benchmarking suite."""

    # ==================== API Configuration ====================
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Please create a .env file with your API key."
        )

    # ==================== Model Configuration ====================
    # Heavy logic (extraction, reasoning, agentic adjudication)
    MODEL_HEAVY = "gemini-2.5-flash"

    # Light/comparison tasks
    MODEL_LIGHT = "gemini-2.5-flash-lite"

    # Token limits
    MAX_TOKENS_PER_REQUEST = 250_000  # Auto-chunk if exceeded

    # Agentic LLM configuration
    AGENTIC_CANDIDATE_COUNT = 8
    AGENTIC_TEMPERATURE = 0.7

    # ==================== Path Configuration ====================
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # Dataset directory
    DATASET_DIR = PROJECT_ROOT / "dataset"

    # Output directories
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MINERU_PARSED_DIR = OUTPUTS_DIR / "mineru_parsed"
    ENTITIES_EXTRACTED_DIR = OUTPUTS_DIR / "entities_extracted"
    KNOWLEDGE_GRAPHS_DIR = OUTPUTS_DIR / "knowledge_graphs"
    REASONING_DIR = OUTPUTS_DIR / "reasoning"

    # Ensure output directories exist
    for dir_path in [OUTPUTS_DIR, MINERU_PARSED_DIR, ENTITIES_EXTRACTED_DIR, KNOWLEDGE_GRAPHS_DIR, REASONING_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ==================== Mineru Configuration ====================
    MINERU_USE_GPU = True  # User confirmed CUDA available
    MINERU_DPI = 300  # High-quality image extraction

    # ==================== Deduplication Thresholds ====================
    # R-Swoosh
    RSWOOSH_SIMILARITY_THRESHOLD = 0.85

    # Probabilistic (Splink)
    SPLINK_MATCH_PROBABILITY_THRESHOLD = 0.8

    # Topological (NetworkX Jaccard)
    TOPOLOGICAL_JACCARD_THRESHOLD = 0.7

    # Semantic (Embeddings)
    SEMANTIC_SIMILARITY_THRESHOLD = 0.9
    SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # High accuracy

    # Agentic LLM
    AGENTIC_MAJORITY_THRESHOLD = 5  # Out of 8 candidates

    # ==================== Milvus Configuration ====================
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    MILVUS_COLLECTION_NAME = "entity_embeddings"
    MILVUS_DIMENSION = 768  # all-mpnet-base-v2 embedding dimension

    # ==================== Entity Extraction Configuration ====================
    ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
You are analyzing a financial report. Extract ALL entities and their relationships from the provided pages.

INSTRUCTIONS:
1. Extract ALL entity types you find (People, Companies, Organizations, etc.)
2. For each entity, extract as many attributes as possible (names, IDs, roles, addresses, etc.)
3. Extract relationships between entities (e.g., "Person A is Director of Company B")
4. Return the result as a valid JSON object with this structure:
{
  "entities": [
    {
      "id": "unique_id",
      "type": "Person|Company|Organization|etc",
      "attributes": {
        "name": "...",
        "role": "...",
        "other_field": "..."
      }
    }
  ],
  "relationships": [
    {
      "from_id": "entity_id",
      "to_id": "entity_id",
      "type": "WORKS_FOR|DIRECTOR_OF|AUDITOR_FOR|etc",
      "attributes": {}
    }
  ]
}

IMPORTANT:
- Assign unique IDs to each entity (e.g., "PERSON_001", "COMPANY_001")
- Extract ALL available attributes (don't limit to a fixed schema)
- Include page numbers where entities are mentioned
- Be thorough - extract every person, company, and organization mentioned

Now process the following pages:
"""

    # ==================== Logging Configuration ====================
    LOG_LEVEL = "INFO"
    LOG_FILE = OUTPUTS_DIR / "benchmark.log"

    # ==================== Benchmark Configuration ====================
    SAVE_INTERMEDIATE_OUTPUTS = True
    GENERATE_VISUALIZATIONS = False  # Can be enabled later

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in vars(cls).items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 80)
        print("ENTITY RESOLUTION BENCHMARKING SUITE - CONFIGURATION")
        print("=" * 80)
        print(f"Model (Heavy): {cls.MODEL_HEAVY}")
        print(f"Model (Light): {cls.MODEL_LIGHT}")
        print(f"Max Tokens: {cls.MAX_TOKENS_PER_REQUEST:,}")
        print(f"Dataset: {cls.DATASET_DIR}")
        print(f"Outputs: {cls.OUTPUTS_DIR}")
        print(f"Mineru GPU: {cls.MINERU_USE_GPU}")
        print(f"Embedding Model: {cls.SEMANTIC_EMBEDDING_MODEL}")
        print("=" * 80)


# Initialize config on import
if __name__ == "__main__":
    Config.print_config()
