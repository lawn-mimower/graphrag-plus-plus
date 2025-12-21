"""
Token Manager for handling token counting and chunking with multimodal content.
Ensures requests stay within the 250k token limit.
"""

import logging
from typing import List, Tuple, Any, Dict
from google import genai
from PIL import Image
import io

from src.config import Config

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages token counting and chunking for multimodal content.
    Ensures all API requests stay within the token limit.
    """

    def __init__(self, client: genai.Client):
        """
        Initialize TokenManager.

        Args:
            client: Initialized Google GenAI client
        """
        self.client = client
        self.max_tokens = Config.MAX_TOKENS_PER_REQUEST
        self.model_name = Config.MODEL_HEAVY

    def count_tokens(self, contents: List[Any], use_api: bool = False) -> int:
        """
        Count tokens in the provided content using local estimation.
        Optionally use API for precise counting (slower, uses API quota).

        Args:
            contents: List of content parts (text, images, etc.)
            use_api: If True, use Gemini API for precise count (default: False)

        Returns:
            Total token count
        """
        if use_api:
            try:
                response = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=contents
                )
                total_tokens = response.total_tokens
                logger.debug(f"API token count: {total_tokens:,}")
                return total_tokens
            except Exception as e:
                logger.warning(f"API token counting failed, using estimation: {e}")
                return self._estimate_tokens(contents)
        else:
            # Use local estimation (fast, no API calls)
            return self._estimate_tokens(contents)

    def _estimate_tokens(self, contents: List[Any]) -> int:
        """
        Estimate token count locally without API calls.
        Based on Gemini's actual token counting behavior:
        - Text: ~4 chars per token
        - Images: Based on resolution (258 tokens per 512x512 tile)

        Args:
            contents: List of content parts

        Returns:
            Estimated token count
        """
        total = 0
        for item in contents:
            if isinstance(item, str):
                # Text estimation: ~4 chars per token
                total += len(item) // 4
            elif hasattr(item, 'text') and item.text is not None:
                # Part with text attribute
                total += len(item.text) // 4
            elif hasattr(item, 'inline_data') and hasattr(item.inline_data, 'data'):
                # Image estimation based on Gemini's tile-based counting
                # Gemini divides images into 512x512 tiles, ~258 tokens per tile
                # We'll use a conservative estimate based on typical PDF page size
                # A typical PDF page rendered at 200 DPI is ~1600x2200 pixels
                # That's roughly (1600/512) * (2200/512) = 3.1 * 4.3 ≈ 13 tiles
                # 13 tiles * 258 tokens ≈ 3354 tokens per page image
                total += 3500  # Conservative estimate for typical PDF page
        return total

    def chunk_multimodal_content(
        self,
        page_pairs: List[Tuple[Image.Image, str, Dict]],
        instruction_prompt: str
    ) -> List[List[Any]]:
        """
        Chunk multimodal content (image, text pairs) to stay within token limit.
        Strategy: Build full content first, count once, then chunk if needed.
        Preserves page pair integrity (never splits a single page).

        Args:
            page_pairs: List of (page_image, page_text, metadata) tuples
            instruction_prompt: The instruction/system prompt to prepend

        Returns:
            List of content chunks, where each chunk is a list of content parts
        """
        # Step 1: Build all page parts and calculate tokens once per page
        instruction_part = genai.types.Part(text=instruction_prompt)
        instruction_tokens = self.count_tokens([instruction_part])

        page_data = []  # List of (page_parts, token_count) tuples

        logger.info(f"Building content for {len(page_pairs)} pages...")
        for page_img, page_text, metadata in page_pairs:
            page_parts = self._create_page_parts(page_img, page_text, metadata)
            page_tokens = self.count_tokens(page_parts)
            page_data.append((page_parts, page_tokens))

        # Step 2: Calculate total tokens (instruction + all pages)
        total_tokens = instruction_tokens + sum(tokens for _, tokens in page_data)
        logger.info(f"Total estimated tokens: {total_tokens:,} (limit: {self.max_tokens:,})")

        # Step 3: If within limit, return as single chunk
        if total_tokens <= self.max_tokens:
            logger.info("All content fits in single API call")
            full_content = [instruction_part]
            for page_parts, _ in page_data:
                full_content.extend(page_parts)
            return [full_content]

        # Step 4: Need to chunk - split pages across multiple chunks
        logger.info(f"Content exceeds limit, chunking required...")
        chunks = []
        current_chunk_parts = []
        current_chunk_tokens = instruction_tokens
        current_chunk_page_count = 0

        for page_parts, page_tokens in page_data:
            # Use pre-calculated token count (no recalculation needed!)

            # Check if adding this page would exceed limit
            if current_chunk_tokens + page_tokens > self.max_tokens and current_chunk_parts:
                # Save current chunk and start new one
                chunk = [instruction_part] + current_chunk_parts
                chunks.append(chunk)
                logger.info(
                    f"Chunk {len(chunks)} complete: {current_chunk_page_count} pages, "
                    f"~{current_chunk_tokens:,} tokens"
                )

                # Start new chunk
                current_chunk_parts = page_parts.copy()
                current_chunk_tokens = instruction_tokens + page_tokens
                current_chunk_page_count = 1
            else:
                # Add to current chunk
                current_chunk_parts.extend(page_parts)
                current_chunk_tokens += page_tokens
                current_chunk_page_count += 1

        # Add final chunk
        if current_chunk_parts:
            chunk = [instruction_part] + current_chunk_parts
            chunks.append(chunk)
            logger.info(
                f"Chunk {len(chunks)} complete: {current_chunk_page_count} pages, "
                f"~{current_chunk_tokens:,} tokens"
            )

        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def _create_page_parts(
        self,
        page_img: Image.Image,
        page_text: str,
        metadata: Dict
    ) -> List[Any]:
        """
        Create Gemini API parts for a single page (IMAGE ONLY - Option D).

        OCR text is included as supplementary context in the instruction prompt,
        not inline with each page. This allows Gemini's vision to be the primary
        processor, with OCR as backup reference.

        Args:
            page_img: PIL Image object
            page_text: Extracted text from page (unused - in instruction instead)
            metadata: Page metadata (page number, etc.)

        Returns:
            List of parts [image_part]
        """
        parts = []

        # Add image part (PRIMARY CONTENT)
        if page_img:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            page_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            image_part = genai.types.Part(
                inline_data=genai.types.Blob(
                    mime_type="image/png",
                    data=img_bytes
                )
            )
            parts.append(image_part)

        # Note: Text is NOT included inline - it's in the instruction prompt
        # This is Option D: Images primary, OCR as supplementary context

        return parts

    def should_chunk(self, contents: List[Any]) -> bool:
        """
        Check if content needs to be chunked.

        Args:
            contents: Content to check

        Returns:
            True if chunking is needed, False otherwise
        """
        token_count = self.count_tokens(contents)
        return token_count > self.max_tokens

    def validate_chunk_size(self, chunk: List[Any]) -> bool:
        """
        Validate that a chunk is within token limits.

        Args:
            chunk: Content chunk to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        token_count = self.count_tokens(chunk)
        if token_count > self.max_tokens:
            logger.warning(
                f"Chunk exceeds token limit: {token_count:,} > {self.max_tokens:,}"
            )
            return False
        return True

    def estimate_api_calls(self, page_pairs: List[Tuple]) -> int:
        """
        Estimate number of API calls needed for given page pairs.

        Args:
            page_pairs: List of (image, text, metadata) tuples

        Returns:
            Estimated number of API calls
        """
        # Simple estimation: count pages and assume average page is ~5k tokens
        total_pages = len(page_pairs)
        avg_tokens_per_page = 5000  # Conservative estimate
        total_estimated_tokens = total_pages * avg_tokens_per_page

        estimated_calls = (total_estimated_tokens // self.max_tokens) + 1
        logger.info(
            f"Estimated API calls for {total_pages} pages: {estimated_calls}"
        )
        return estimated_calls


# Utility function for initializing TokenManager
def create_token_manager() -> TokenManager:
    """
    Create and return a TokenManager instance with initialized client.

    Returns:
        TokenManager instance
    """
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    return TokenManager(client)
