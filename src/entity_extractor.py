"""
Multimodal Entity Extractor using Gemini 2.5 Flash.
Extracts entities and relationships from (page_image, page_text) pairs.
"""

import logging
import json
from typing import List, Tuple, Dict, Any
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

        # Build the instruction prompt
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

        Returns:
            Instruction prompt string
        """
        return Config.ENTITY_EXTRACTION_PROMPT_TEMPLATE

    def _extract_from_chunk(
        self,
        chunk_content: List[Any],
        doc_name: str
    ) -> Dict[str, Any]:
        """
        Extract entities from a single content chunk.

        Args:
            chunk_content: List of content parts (instruction + images + text)
            doc_name: Document name for logging

        Returns:
            Extraction result dictionary
        """
        try:
            logger.debug(f"Calling Gemini API for {doc_name}")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=chunk_content,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for factual extraction
                    max_output_tokens=65536,
                )
            )

            # Check for blocked or empty responses
            if not response or not hasattr(response, 'text'):
                logger.error(f"Invalid response from Gemini for {doc_name}")
                logger.debug(f"Response object: {response}")
                return {"entities": [], "relationships": []}

            # Parse the response
            response_text = response.text

            if response_text is None or response_text.strip() == "":
                logger.error(f"Gemini returned empty/None response for {doc_name}")
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback'):
                    logger.error(f"Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'finish_reason'):
                            logger.error(f"Finish reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            logger.error(f"Safety ratings: {candidate.safety_ratings}")
                return {"entities": [], "relationships": []}

            logger.debug(f"Received response ({len(response_text)} chars)")

            # Extract JSON from response
            result = self._parse_extraction_response(response_text)

            return result

        except Exception as e:
            logger.error(f"Error extracting entities from {doc_name}: {e}", exc_info=True)
            return {"entities": [], "relationships": []}

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

    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the JSON response from Gemini.

        Args:
            response_text: Raw response text from Gemini

        Returns:
            Parsed dictionary with entities and relationships
        """
        try:
            # Try to find JSON in the response
            # Sometimes the model wraps JSON in markdown code blocks
            json_str = response_text

            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            # Parse JSON
            result = json.loads(json_str.strip())

            # Validate structure
            if "entities" not in result:
                result["entities"] = []
            if "relationships" not in result:
                result["relationships"] = []

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")

            # Fallback: try to extract entities with simpler parsing
            return self._fallback_parse(response_text)

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
