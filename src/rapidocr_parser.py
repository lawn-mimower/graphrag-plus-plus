"""
Universal Document Parser using RapidOCR.
Supports multiple document formats with OCR capabilities.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import io

from src.config import Config

logger = logging.getLogger(__name__)

# Try to import RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    logger.warning("RapidOCR not available. Install with: pip install rapidocr-onnxruntime")


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """
        Parse document and return (page_image, page_text, metadata) tuples.

        Args:
            file_path: Path to document

        Returns:
            List of (Image, text, metadata) for each page/section
        """
        pass

    @abstractmethod
    def supports_format(self, file_path: Path) -> bool:
        """Check if this parser supports the given file format."""
        pass


class OCREngine:
    """Wrapper for RapidOCR with quality assessment."""

    def __init__(self):
        """Initialize OCR engine."""
        if not RAPIDOCR_AVAILABLE:
            raise ImportError(
                "RapidOCR not installed. Install with: pip install rapidocr-onnxruntime"
            )

        # Initialize RapidOCR
        # use_gpu is automatically handled by onnxruntime if available
        self.ocr = RapidOCR()
        logger.info("RapidOCR engine initialized")

    def extract_text(self, image: Image.Image) -> Tuple[str, float, List[Dict]]:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image

        Returns:
            Tuple of (extracted_text, avg_confidence, ocr_results)
        """
        try:
            # Convert PIL Image to format RapidOCR expects
            img_array = self._pil_to_array(image)

            # Run OCR
            result, elapse = self.ocr(img_array)

            if result is None or len(result) == 0:
                logger.warning("No text detected by OCR")
                return "", 0.0, []

            # Parse results
            # RapidOCR returns: List of [bbox, text, confidence]
            texts = []
            confidences = []
            ocr_details = []

            for item in result:
                bbox, text, confidence = item
                texts.append(text)
                confidences.append(confidence)
                ocr_details.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })

            # Combine all text
            full_text = "\n".join(texts)

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.debug(f"OCR extracted {len(texts)} text blocks, avg confidence: {avg_confidence:.2f}")

            return full_text, avg_confidence, ocr_details

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return "", 0.0, []

    def _pil_to_array(self, image: Image.Image):
        """Convert PIL Image to numpy array for RapidOCR."""
        import numpy as np

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array
        return np.array(image)

    def assess_text_quality(self, text: str, confidence: float = None) -> float:
        """
        Assess the quality of extracted text.

        Args:
            text: Extracted text
            confidence: OCR confidence score (0.0-1.0)

        Returns:
            Quality score 0.0-1.0
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        # Factor 1: OCR confidence (if available)
        confidence_score = confidence if confidence is not None else 0.5

        # Factor 2: Alphanumeric ratio
        alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)

        # Factor 3: Special character ratio (low is better)
        special_char_ratio = sum(
            not c.isalnum() and not c.isspace() for c in text
        ) / len(text)

        # Factor 4: Word length distribution
        words = text.split()
        if not words:
            return 0.0

        avg_word_length = sum(len(w) for w in words) / len(words)
        word_length_score = 1.0 if 3 <= avg_word_length <= 10 else 0.5

        # Combine scores
        quality = (
            confidence_score * 0.4 +
            alphanumeric_ratio * 0.3 +
            (1 - special_char_ratio) * 0.2 +
            word_length_score * 0.1
        )

        return min(1.0, max(0.0, quality))


class PDFParser(DocumentParser):
    """PDF parser with OCR support using RapidOCR."""

    def __init__(self, ocr_engine: OCREngine = None, dpi: int = None):
        """
        Initialize PDF parser.

        Args:
            ocr_engine: OCR engine instance
            dpi: DPI for rendering pages
        """
        self.ocr_engine = ocr_engine or OCREngine()
        self.dpi = dpi or Config.RENDER_DPI

    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """Parse PDF file."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
            return []

        logger.info(f"Parsing PDF: {file_path.name}")
        page_pairs = []

        try:
            doc = fitz.open(file_path)

            # Detect if PDF has text layer
            has_text_layer = self._has_text_layer(doc)
            logger.info(f"PDF type: {'Text-based' if has_text_layer else 'Scanned/Image-based'}")

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1

                # Extract or OCR text
                if has_text_layer:
                    # Text-based PDF - extract directly
                    text = page.get_text()
                    ocr_confidence = 1.0  # Native text extraction
                    ocr_results = []
                else:
                    # Scanned PDF - use OCR
                    text, ocr_confidence, ocr_results = self._ocr_page(page)

                # Render page as image
                pix = page.get_pixmap(dpi=self.dpi)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # Assess text quality
                text_quality = self.ocr_engine.assess_text_quality(text, ocr_confidence)

                # Create metadata
                metadata = {
                    'page_number': page_num,
                    'source_file': file_path.stem,
                    'original_format': 'pdf',
                    'has_text_layer': has_text_layer,
                    'ocr_confidence': ocr_confidence,
                    'text_quality': text_quality,
                    'ocr_results': ocr_results if not has_text_layer else None
                }

                page_pairs.append((image, text, metadata))

            doc.close()
            logger.info(f"Successfully parsed {len(page_pairs)} pages from PDF")

            return page_pairs

        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}", exc_info=True)
            return []

    def _has_text_layer(self, doc) -> bool:
        """Check if PDF has extractable text."""
        # Sample first 3 pages
        sample_size = min(3, len(doc))
        text_found = False

        for i in range(sample_size):
            text = doc[i].get_text().strip()
            if len(text) > 100:  # Threshold: at least 100 chars
                text_found = True
                break

        return text_found

    def _ocr_page(self, page) -> Tuple[str, float, List[Dict]]:
        """OCR a single PDF page."""
        # Render page as image
        pix = page.get_pixmap(dpi=self.dpi)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        # Run OCR
        return self.ocr_engine.extract_text(image)

    def supports_format(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == '.pdf'


class ImageParser(DocumentParser):
    """Parser for image files (JPG, PNG, TIFF)."""

    def __init__(self, ocr_engine: OCREngine = None):
        """Initialize image parser."""
        self.ocr_engine = ocr_engine or OCREngine()

    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """Parse image file."""
        logger.info(f"Parsing image: {file_path.name}")

        try:
            # Load image
            with Image.open(file_path) as img:
                image = img.copy()

            # Run OCR
            text, ocr_confidence, ocr_results = self.ocr_engine.extract_text(image)

            # Assess quality
            text_quality = self.ocr_engine.assess_text_quality(text, ocr_confidence)

            metadata = {
                'page_number': 1,
                'source_file': file_path.stem,
                'original_format': file_path.suffix.lower().replace('.', ''),
                'is_image': True,
                'ocr_confidence': ocr_confidence,
                'text_quality': text_quality,
                'ocr_results': ocr_results
            }

            return [(image, text, metadata)]

        except Exception as e:
            logger.error(f"Failed to parse image {file_path}: {e}", exc_info=True)
            return []

    def supports_format(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']


class DOCXParser(DocumentParser):
    """Parser for Microsoft Word documents (.docx)."""

    def __init__(self, ocr_engine: OCREngine = None, dpi: int = None):
        """Initialize DOCX parser."""
        self.ocr_engine = ocr_engine or OCREngine()
        self.dpi = dpi or Config.RENDER_DPI

    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """Parse DOCX file."""
        logger.info(f"Parsing Word document: {file_path.name}")

        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return []

        try:
            # Extract text directly
            doc = Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

            # Convert to PDF then to images for visual representation
            # This requires LibreOffice or similar
            images = self._docx_to_images(file_path)

            if not images:
                # Fallback: create a simple text representation
                logger.warning("Could not convert DOCX to images, using text only")
                return self._text_only_representation(full_text, file_path)

            # Create page pairs
            page_pairs = []
            # Split text by pages (rough approximation)
            text_per_page = self._split_text_by_pages(full_text, len(images))

            for idx, (image, page_text) in enumerate(zip(images, text_per_page), 1):
                metadata = {
                    'page_number': idx,
                    'source_file': file_path.stem,
                    'original_format': 'docx',
                    'has_text_layer': True,
                    'ocr_confidence': 1.0,  # Native text extraction
                    'text_quality': 1.0
                }
                page_pairs.append((image, page_text, metadata))

            return page_pairs

        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}", exc_info=True)
            return []

    def _docx_to_images(self, file_path: Path) -> List[Image.Image]:
        """Convert DOCX to images via PDF."""
        import tempfile
        import fitz

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / f"{file_path.stem}.pdf"

                # Try converting with LibreOffice (if available)
                import subprocess
                result = subprocess.run(
                    ['libreoffice', '--headless', '--convert-to', 'pdf',
                     '--outdir', tmpdir, str(file_path)],
                    capture_output=True,
                    timeout=30
                )

                if result.returncode != 0 or not pdf_path.exists():
                    logger.warning("LibreOffice conversion failed")
                    return []

                # Convert PDF pages to images
                doc = fitz.open(pdf_path)
                images = []

                for page in doc:
                    pix = page.get_pixmap(dpi=self.dpi)
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    images.append(image)

                doc.close()
                return images

        except Exception as e:
            logger.warning(f"DOCX to image conversion failed: {e}")
            return []

    def _text_only_representation(self, text: str, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """Create a text-only representation when image conversion fails."""
        metadata = {
            'page_number': 1,
            'source_file': file_path.stem,
            'original_format': 'docx',
            'has_text_layer': True,
            'ocr_confidence': 1.0,
            'text_quality': 1.0,
            'warning': 'Image conversion failed, text-only mode'
        }

        # Create a blank image placeholder
        placeholder = Image.new('RGB', (800, 1000), color='white')

        return [(placeholder, text, metadata)]

    def _split_text_by_pages(self, text: str, num_pages: int) -> List[str]:
        """Split text roughly into pages."""
        if num_pages <= 1:
            return [text]

        # Simple character-based split
        chars_per_page = len(text) // num_pages
        pages = []

        for i in range(num_pages):
            start = i * chars_per_page
            end = start + chars_per_page if i < num_pages - 1 else len(text)
            pages.append(text[start:end])

        return pages

    def supports_format(self, file_path: Path) -> bool:
        """Check if file is a Word document."""
        return file_path.suffix.lower() in ['.docx']


class XLSXParser(DocumentParser):
    """Parser for Excel spreadsheets (.xlsx)."""

    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """Parse Excel file."""
        logger.info(f"Parsing Excel file: {file_path.name}")

        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.error("pandas/matplotlib not installed. Install with: pip install pandas matplotlib")
            return []

        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            page_pairs = []

            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, 1):
                df = excel_file.parse(sheet_name)

                # Convert to text
                sheet_text = f"SHEET: {sheet_name}\n\n{df.to_string()}"

                # Create visual representation
                image = self._render_dataframe(df, sheet_name)

                metadata = {
                    'page_number': sheet_idx,
                    'sheet_name': sheet_name,
                    'source_file': file_path.stem,
                    'original_format': 'xlsx',
                    'has_text_layer': True,
                    'ocr_confidence': 1.0,
                    'text_quality': 1.0,
                    'rows': len(df),
                    'columns': len(df.columns)
                }

                page_pairs.append((image, sheet_text, metadata))

            logger.info(f"Successfully parsed {len(page_pairs)} sheets from Excel")
            return page_pairs

        except Exception as e:
            logger.error(f"Failed to parse Excel {file_path}: {e}", exc_info=True)
            return []

    def _render_dataframe(self, df: 'pd.DataFrame', title: str) -> Image.Image:
        """
        Render DataFrame as image using PIL (much faster than matplotlib).

        Args:
            df: Pandas DataFrame
            title: Sheet title

        Returns:
            PIL Image of the rendered table
        """
        try:
            from PIL import ImageDraw, ImageFont

            # Limit display for large DataFrames
            display_df = df.head(50) if len(df) > 50 else df
            truncated = len(df) > 50

            # Configuration
            CELL_PADDING = 10
            HEADER_HEIGHT = 40
            TITLE_HEIGHT = 50
            ROW_HEIGHT = 30
            MIN_CELL_WIDTH = 80
            MAX_CELL_WIDTH = 300
            FONT_SIZE = 12
            HEADER_FONT_SIZE = 14
            TITLE_FONT_SIZE = 16

            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", HEADER_FONT_SIZE)
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TITLE_FONT_SIZE)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                title_font = ImageFont.load_default()

            # Calculate column widths based on content
            col_widths = []
            for col in display_df.columns:
                # Get max width of column name and values
                header_text = str(col)[:30]  # Truncate long headers
                max_content_width = len(header_text) * 8  # Rough estimate

                for val in display_df[col]:
                    val_text = str(val)[:30]  # Truncate long values
                    max_content_width = max(max_content_width, len(val_text) * 8)

                # Apply bounds
                width = max(MIN_CELL_WIDTH, min(max_content_width + CELL_PADDING * 2, MAX_CELL_WIDTH))
                col_widths.append(width)

            # Calculate image dimensions
            total_width = sum(col_widths) + CELL_PADDING * 2
            total_height = TITLE_HEIGHT + HEADER_HEIGHT + (len(display_df) * ROW_HEIGHT) + CELL_PADDING * 2

            if truncated:
                total_height += ROW_HEIGHT  # Extra row for truncation notice

            # Create image
            img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(img)

            # Draw title
            title_y = CELL_PADDING
            draw.text((CELL_PADDING, title_y), title, fill='black', font=title_font)

            # Draw header background
            header_y = TITLE_HEIGHT
            draw.rectangle(
                [(CELL_PADDING, header_y), (total_width - CELL_PADDING, header_y + HEADER_HEIGHT)],
                fill='#E0E0E0',
                outline='black'
            )

            # Draw column headers
            x_offset = CELL_PADDING
            for i, (col, width) in enumerate(zip(display_df.columns, col_widths)):
                col_text = str(col)[:30]  # Truncate if too long
                draw.text(
                    (x_offset + CELL_PADDING // 2, header_y + CELL_PADDING),
                    col_text,
                    fill='black',
                    font=header_font
                )

                # Draw vertical separator
                if i < len(display_df.columns) - 1:
                    draw.line(
                        [(x_offset + width, header_y), (x_offset + width, header_y + HEADER_HEIGHT)],
                        fill='black',
                        width=1
                    )

                x_offset += width

            # Draw data rows
            y_offset = header_y + HEADER_HEIGHT
            for row_idx, row in display_df.iterrows():
                x_offset = CELL_PADDING

                # Alternate row background
                if row_idx % 2 == 0:
                    draw.rectangle(
                        [(CELL_PADDING, y_offset), (total_width - CELL_PADDING, y_offset + ROW_HEIGHT)],
                        fill='#F5F5F5'
                    )

                # Draw horizontal line
                draw.line(
                    [(CELL_PADDING, y_offset), (total_width - CELL_PADDING, y_offset)],
                    fill='black',
                    width=1
                )

                # Draw cell values
                for i, (val, width) in enumerate(zip(row, col_widths)):
                    val_text = str(val)[:30]  # Truncate if too long
                    draw.text(
                        (x_offset + CELL_PADDING // 2, y_offset + CELL_PADDING // 2),
                        val_text,
                        fill='black',
                        font=font
                    )

                    # Draw vertical separator
                    if i < len(display_df.columns) - 1:
                        draw.line(
                            [(x_offset + width, y_offset), (x_offset + width, y_offset + ROW_HEIGHT)],
                            fill='#CCCCCC',
                            width=1
                        )

                    x_offset += width

                y_offset += ROW_HEIGHT

            # Draw bottom border
            draw.line(
                [(CELL_PADDING, y_offset), (total_width - CELL_PADDING, y_offset)],
                fill='black',
                width=2
            )

            # Add truncation notice if applicable
            if truncated:
                draw.text(
                    (CELL_PADDING, y_offset + CELL_PADDING),
                    f"... ({len(df) - 50} more rows not shown)",
                    fill='#666666',
                    font=font
                )

            # Draw outer border
            draw.rectangle(
                [(CELL_PADDING, TITLE_HEIGHT), (total_width - CELL_PADDING, y_offset)],
                outline='black',
                width=2
            )

            return img

        except Exception as e:
            logger.warning(f"Failed to render DataFrame with PIL: {e}")
            # Return blank placeholder with error message
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), f"Error rendering table: {title}", fill='red')
            return img

    def supports_format(self, file_path: Path) -> bool:
        """Check if file is an Excel spreadsheet."""
        return file_path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']


class RapidOCRParser:
    """
    Universal document parser that routes to appropriate handler.
    Main entry point for document parsing.
    """

    def __init__(self):
        """Initialize universal parser."""
        logger.info("Initializing RapidOCR Universal Document Parser")

        # Initialize OCR engine (shared across parsers)
        self.ocr_engine = OCREngine()

        # Initialize format-specific parsers
        self.parsers = [
            PDFParser(self.ocr_engine),
            ImageParser(self.ocr_engine),
            DOCXParser(self.ocr_engine),
            XLSXParser(),
        ]

        logger.info(f"Loaded {len(self.parsers)} document format handlers")

    def parse(self, file_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
        """
        Parse any supported document type.

        Args:
            file_path: Path to document

        Returns:
            List of (image, text, metadata) tuples
        """
        for parser in self.parsers:
            if parser.supports_format(file_path):
                logger.info(f"Using {parser.__class__.__name__} for {file_path.name}")
                return parser.parse(file_path)

        raise ValueError(
            f"No parser found for file type: {file_path.suffix}. "
            f"Supported formats: {Config.SUPPORTED_FORMATS}"
        )

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return Config.SUPPORTED_FORMATS


# Alias for backward compatibility with MinervParser interface
def parse_pdf(pdf_path: Path) -> List[Tuple[Image.Image, str, Dict]]:
    """
    Parse a PDF file (backward compatibility function).

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of (image, text, metadata) tuples
    """
    parser = RapidOCRParser()
    return parser.parse(pdf_path)
