"""
Multimodal Entity Extractor using Gemini 2.5 Flash.
Extracts entities and relationships from (page_image, page_text) pairs.
"""

import logging
import json
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
from google import genai

from src.config import Config
from src.token_manager import TokenManager

logger = logging.getLogger(__name__)


class MultimodalEntityExtractor:
    """
    Extracts entities and relationships from multimodal content using Gemini.
    Handles token management and chunking for large documents.
    """

    def __init__(self, client: genai.Client, token_manager: TokenManager):
        """
        Initialize the extractor.

        Args:
            client: Initialized Google GenAI client
            token_manager: TokenManager for handling token limits
        """
        self.client = client
        self.token_manager = token_manager
        self.model_name = Config.MODEL_HEAVY
        self.failed_documents: List[str] = []  # chunks whose extraction failed after a retry

    def extract_entities(
        self,
        page_pairs: List[Tuple[Image.Image, str, Dict]],
        doc_name: str,
        save_path: Path = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from page pairs.

        Args:
            page_pairs: List of (page_image, page_text, metadata) tuples
            doc_name: Name of the document being processed
            save_path: Optional path to save extraction results

        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        logger.info(f"Extracting entities from {doc_name} ({len(page_pairs)} pages)")

        # Build the instruction prompt (DocLens interleaved approach)
        instruction_prompt = self._build_instruction_prompt()

        # Check if chunking is needed
        chunks = self.token_manager.chunk_multimodal_content(
            page_pairs,
            instruction_prompt
        )

        if len(chunks) == 1:
            # Single API call
            logger.info("Processing document in single API call")
            result = self._extract_from_chunk(chunks[0], doc_name)
        else:
            # Multiple API calls needed
            logger.info(f"Processing document in {len(chunks)} chunks")
            result = self._extract_from_multiple_chunks(chunks, doc_name)

        # Save results if path provided
        if save_path:
            self._save_results(result, save_path)

        logger.info(
            f"Extracted {len(result['entities'])} entities and "
            f"{len(result['relationships'])} relationships from {doc_name}"
        )

        return result

    def _build_instruction_prompt(self) -> str:
        """
        Build the instruction prompt for entity extraction.

        DocLens Phase 1 Philosophy:
        - Text is PRIMARY (ground truth for data/values)
        - Images are SECONDARY (provide structure/layout context)
        - Content is INTERLEAVED (text-image pairs, not separated)
        - NO summarization (preserve all detail for recall)

        Returns:
            Instruction prompt string with DocLens guidance
        """
        base_prompt = Config.ENTITY_EXTRACTION_PROMPT_TEMPLATE

        # Add DocLens-specific guidance
        doclens_guidance = """

**EXTRACTION STRATEGY (DocLens Phase 1):**

For each page, you will see:
1. **OCR Text (PRIMARY)** - Use this as your ground truth for all entity values, names, dates, numbers
2. **Page Image (SECONDARY)** - Use this for understanding structure, layout, and spatial relationships

**Critical Instructions:**
- Extract entity VALUES (names, dates, IDs, amounts) from the OCR TEXT, not by reading the image
- Use the IMAGE to understand context (e.g., is this a table? a form? what fields relate to each other?)
- When OCR text quality is low, you may reference the image, but prefer text-based extraction
- DO NOT summarize or skip entities - extract ALL relevant information with maximum recall

This approach prevents hallucination and ensures accurate entity extraction.
"""

        full_prompt = base_prompt + doclens_guidance

        return full_prompt

    def _extract_from_chunk(
        self,
        chunk_content: List[Any],
        doc_name: str
    ) -> Dict[str, Any]:
        """
        Extract entities from a single content chunk.

        The model is called at most twice. A second call is made when the first
        reply is empty, is not valid JSON even after repair, or contains no entities
        although the chunk holds a substantial amount of text. The retry appends an
        instruction to return only the JSON object. If both attempts fail the chunk is
        recorded in ``failed_documents`` and an empty result is returned.

        Args:
            chunk_content: List of content parts (instruction + images + text)
            doc_name: Document name for logging

        Returns:
            Extraction result dictionary
        """
        text_chars = sum(len(getattr(part, "text", "") or "") for part in chunk_content[1:])
        contents = list(chunk_content)
        last_error = "empty reply"
        for attempt in (1, 2):
            try:
                logger.debug(f"Calling the model for {doc_name} (attempt {attempt})")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for factual extraction
                        max_output_tokens=65536,
                    )
                )
                response_text = getattr(response, "text", None) or ""
                if response_text.strip():
                    result = self._parse_extraction_response(response_text)
                    if result is None:
                        last_error = "reply was not valid JSON"
                    elif not result["entities"] and text_chars >= 200:
                        last_error = "no entities from a chunk with substantial text"
                    else:
                        return result
                else:
                    last_error = "empty reply"
                    self._log_empty_response(response, doc_name)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.error(f"Error extracting entities from {doc_name}: {e}", exc_info=True)
            if attempt == 1:
                logger.warning(f"Extraction for {doc_name} failed ({last_error}); retrying once")
                contents = list(chunk_content) + [genai.types.Part(
                    text="Return only the JSON object described above, with no text before or after it.")]
        logger.error(f"Extraction failed for {doc_name} after 2 attempts: {last_error}")
        self.failed_documents.append(doc_name)
        return {"entities": [], "relationships": []}

    @staticmethod
    def _log_empty_response(response: Any, doc_name: str) -> None:
        logger.error(f"Model returned an empty response for {doc_name}")
        if getattr(response, "prompt_feedback", None):
            logger.error(f"Prompt feedback: {response.prompt_feedback}")
        for candidate in getattr(response, "candidates", None) or []:
            if getattr(candidate, "finish_reason", None):
                logger.error(f"Finish reason: {candidate.finish_reason}")

    def _extract_from_multiple_chunks(
        self,
        chunks: List[List[Any]],
        doc_name: str
    ) -> Dict[str, Any]:
        """
        Extract entities from multiple chunks and merge results.

        Args:
            chunks: List of content chunks
            doc_name: Document name for logging

        Returns:
            Merged extraction result dictionary
        """
        all_entities = []
        all_relationships = []
        entity_id_mapping = {}  # Map chunk-local IDs to global IDs

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")

            chunk_result = self._extract_from_chunk(chunk, f"{doc_name}_chunk{chunk_idx}")

            # Remap entity IDs to avoid conflicts
            for entity in chunk_result.get('entities', []):
                old_id = entity['id']
                new_id = f"{old_id}_chunk{chunk_idx}"

                entity['id'] = new_id
                entity_id_mapping[old_id] = new_id

                all_entities.append(entity)

            # Remap relationship IDs
            for rel in chunk_result.get('relationships', []):
                rel['from_id'] = entity_id_mapping.get(rel['from_id'], rel['from_id'])
                rel['to_id'] = entity_id_mapping.get(rel['to_id'], rel['to_id'])
                all_relationships.append(rel)

        # Optional: Post-process to merge duplicate entities across chunks
        # For now, we keep them separate and let deduplication handle it

        return {
            "entities": all_entities,
            "relationships": all_relationships
        }

    def _parse_extraction_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the JSON reply. Markdown fences are stripped first; a reply that is not
        strict JSON is repaired with ``json_repair`` (trailing commas, unquoted keys,
        truncated endings). Returns None when no object can be recovered.
        """
        json_str = response_text
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        json_str = json_str.strip()
        result = None
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Reply is not strict JSON ({e}); attempting repair")
            try:
                from json_repair import loads as repair_loads
                result = repair_loads(json_str)
            except Exception as repair_error:  # json_repair missing or hopeless input
                logger.error(f"JSON repair failed: {repair_error}")
                logger.debug(f"Response text: {response_text[:500]}...")
                return None
        if not isinstance(result, dict):
            return None
        if not isinstance(result.get("entities"), list):
            result["entities"] = []
        if not isinstance(result.get("relationships"), list):
            result["relationships"] = []
        return result

    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback parser when JSON parsing fails.
        Uses simple heuristics to extract entity mentions.

        Args:
            response_text: Raw response text

        Returns:
            Dictionary with entities (relationships will be empty)
        """
        logger.warning("Using fallback parser for entity extraction")

        # Simple heuristic: look for capitalized words/phrases
        # This is very basic and should only be used as last resort
        entities = []

        # For now, return empty result
        # In production, you might want to implement more sophisticated fallback
        return {"entities": [], "relationships": []}

    def _save_results(self, result: Dict[str, Any], save_path: Path):
        """
        Save extraction results to JSON file.

        Args:
            result: Extraction result dictionary
            save_path: Path to save file
        """
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved extraction results to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save extraction results: {e}")

    def batch_extract(
        self,
        documents: Dict[str, List[Tuple]],
        save_dir: Path = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from multiple documents.

        Args:
            documents: Dictionary mapping doc_name -> page_pairs
            save_dir: Optional directory to save individual results

        Returns:
            Dictionary mapping doc_name -> extraction_result
        """
        results = {}

        for doc_name, page_pairs in documents.items():
            logger.info(f"Processing document: {doc_name}")

            save_path = None
            if save_dir:
                save_path = save_dir / f"{doc_name}_entities.json"

            try:
                result = self.extract_entities(page_pairs, doc_name, save_path)
                results[doc_name] = result
            except Exception as e:
                logger.error(f"Failed to extract entities from {doc_name}: {e}")
                results[doc_name] = {"entities": [], "relationships": []}

        return results


# Utility function
def create_entity_extractor() -> MultimodalEntityExtractor:
    """
    Create and return a MultimodalEntityExtractor instance.

    Returns:
        MultimodalEntityExtractor instance
    """
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    token_manager = TokenManager(client)
    return MultimodalEntityExtractor(client, token_manager)
