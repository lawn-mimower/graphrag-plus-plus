"""
Mineru Parser for PDF documents.
Uses magic-pdf Python API directly (bypassing CLI) to extract page images, text, and layout.
"""

import logging
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import io

# Import Mineru (Magic-PDF) components
try:
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

from src.config import Config

logger = logging.getLogger(__name__)

class MinervParser:
    """
    Wrapper for magic-pdf (Mineru) to parse PDFs using the Python API.
    Extracts page images and text for multimodal processing.
    """

    def __init__(self, use_gpu: bool = None, dpi: int = None):
        """
        Initialize MinervParser.
        """
        if not MINERU_AVAILABLE:
            logger.warning("magic-pdf not installed. Fallback parser will be used.")

        self.use_gpu = use_gpu if use_gpu is not None else Config.MINERU_USE_GPU
        self.dpi = dpi if dpi is not None else Config.MINERU_DPI
        self.output_base_dir = Path(Config.MINERU_PARSED_DIR)

    def parse_pdf(self, pdf_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """
        Parse a PDF file and extract page images and text.
        """
        logger.info(f"Parsing PDF: {pdf_path.name}")

        # Create output directory for this PDF
        pdf_name = pdf_path.stem
        output_dir = self.output_base_dir / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Try Mineru Python API
        if MINERU_AVAILABLE:
            try:
                self._run_magic_pdf_api(pdf_path, output_dir, pdf_name)
                # If successful, extract the results
                return self._extract_page_pairs(output_dir, pdf_name)
            except Exception as e:
                logger.error(f"Mineru API failed for {pdf_path.name}: {e}")
                logger.warning("Attempting fallback parser...")

        # 2. Fallback to basic PyMuPDF if Mineru is missing or fails
        return self._fallback_parse(pdf_path, output_dir)

    def _run_magic_pdf_api(self, pdf_path: Path, output_dir: Path, pdf_name: str):
        """
        Run Mineru using the direct Python API (UNIPipe).
        This avoids 'command not found' errors from subprocess.
        """
        try:
            # Prepare the writer
            # DiskReaderWriter needs the base directory
            image_writer = DiskReaderWriter(str(output_dir))

            # Read PDF bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # Initialize Pipeline
            # "model_list": [] tells it to use the default config (magic-pdf.json)
            # "_pdf_type": "" lets it auto-detect (OCR vs Text)
            jso_useful_key = {"_pdf_type": "", "model_list": []}
            
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)

            # Execute the pipeline steps
            pipe.pipe_classify() # Classify PDF type
            pipe.pipe_parse()    # Analyze layout
            
            # Generate Markdown and save images
            # image_dir is relative to the writer's base dir
            pipe.pipe_mk_markdown(
                image_dir="images", 
                drop_mode="none" # Keep all images/tables
            )
            
            logger.debug(f"Mineru API execution successful for {pdf_name}")

        except Exception as e:
            raise RuntimeError(f"Mineru Pipeline Error: {str(e)}")

    def _extract_page_pairs(
        self,
        output_dir: Path,
        pdf_name: str
    ) -> List[Tuple[Image.Image, str, Dict]]:
        """
        Extract (image, text, metadata) pairs from the generated output.
        """
        page_pairs = []

        # Mineru API creates:
        # output_dir/
        #   ├── images/ (if we set image_dir="images")
        #   ├── {pdf_name}.md
        #   ├── {pdf_name}_content_list.json

        images_dir = output_dir / "images"
        content_file_md = output_dir / f"{pdf_name}.md"
        content_file_json = output_dir / f"{pdf_name}_content_list.json"

        # Special case: The API might name the json file strictly as "content_list.json" 
        # or include the name depending on version. We check standard variations.
        if not content_file_json.exists():
             # Try generic name
             content_file_json = output_dir / "content_list.json"

        # Load page texts
        page_texts = self._load_page_texts(content_file_json, content_file_md)

        # Load page images
        if images_dir.exists():
            image_files = sorted(images_dir.glob("*.png"))
            if not image_files:
                # Sometimes it uses jpg
                image_files = sorted(images_dir.glob("*.jpg"))
                
            for idx, img_path in enumerate(image_files):
                # Mineru images are usually named "0.png", "1.png" etc, or "span_...".
                # Mapping exactly to page numbers is tricky without the JSON map.
                # We will rely on the index for now, assuming sequential output.
                page_num = idx + 1

                try:
                    # Load image and immediately copy to memory, then close file
                    with Image.open(img_path) as img:
                        page_img = img.copy()  # Copy to memory so file can be closed
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    page_img = None

                # Get text for this page
                page_text = page_texts.get(page_num, "")

                metadata = {
                    'page_number': page_num,
                    'source_pdf': pdf_name,
                    'image_path': str(img_path),
                }

                page_pairs.append((page_img, page_text, metadata))
        else:
            logger.warning(f"Images directory not found at {images_dir}")
            # Return text only
            for page_num, text in page_texts.items():
                metadata = {'page_number': page_num, 'source_pdf': pdf_name}
                page_pairs.append((None, text, metadata))

        return page_pairs

    def _load_page_texts(self, json_path: Path, md_path: Path) -> Dict[int, str]:
        """
        Load page texts, preferring JSON structure if available.
        """
        page_texts = {}

        # 1. Try JSON
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)

                for item in content_list:
                    # Mineru JSON usually has 'page_idx' (0-based)
                    if 'page_idx' in item:
                        page_num = item['page_idx'] + 1
                        text = item.get('text', '')
                        # Append text if we already have some for this page
                        page_texts[page_num] = page_texts.get(page_num, "") + "\n" + text
                
                return page_texts
            except Exception as e:
                logger.warning(f"Failed to parse JSON content: {e}")

        # 2. Fallback to Markdown
        if md_path.exists():
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Rough assignment to page 1 if we can't split it
                page_texts[1] = text 
            except Exception as e:
                logger.warning(f"Failed to read Markdown: {e}")

        return page_texts

    def _fallback_parse(self, pdf_path: Path, output_dir: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """
        Fallback using PyMuPDF (fitz) if Mineru fails.
        Optimized to save images to disk and load them properly to avoid memory issues.
        """
        logger.warning(f"Using PyMuPDF fallback for {pdf_path.name}")
        page_pairs = []

        try:
            import fitz
            doc = fitz.open(pdf_path)

            for i, page in enumerate(doc):
                page_num = i + 1
                text = page.get_text()

                # Render image
                pix = page.get_pixmap(dpi=self.dpi or 200)
                img_data = pix.tobytes("png")

                # Save to disk first
                img_path = output_dir / f"page_{page_num}.png"
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # Load from disk (this ensures proper memory management)
                with Image.open(img_path) as img:
                    image = img.copy()

                # Clean up pixmap immediately
                pix = None
                img_data = None

                metadata = {
                    'page_number': page_num,
                    'source_pdf': pdf_path.stem,
                    'image_path': str(img_path)
                }

                page_pairs.append((image, text, metadata))

            doc.close()
            return page_pairs

        except ImportError:
            logger.error("PyMuPDF (fitz) not installed. Cannot perform fallback.")
            return []
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return []

    def batch_parse(self, pdf_paths: List[Path]) -> Dict[str, List[Tuple]]:
        results = {}
        for pdf_path in pdf_paths:
            results[pdf_path.stem] = self.parse_pdf(pdf_path)
        return results

def create_parser() -> MinervParser:
    return MinervParser()