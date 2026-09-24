"""Document parsing (PDF, DOCX, XLSX, image OCR) on the synthetic fixtures."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from conftest import FIXTURES_DIR

pytest.importorskip("rapidocr_onnxruntime")


@pytest.fixture(scope="module")
def parser():
    from src.rapidocr_parser import RapidOCRParser
    return RapidOCRParser()


def _single_page(pairs):
    assert len(pairs) == 1
    image, text, meta = pairs[0]
    assert isinstance(image, Image.Image)
    assert meta["page_number"] == 1
    return image, text, meta


def test_pdf_text_layer(parser):
    _, text, meta = _single_page(parser.parse(FIXTURES_DIR / "acme_annual_report.pdf"))
    assert meta["original_format"] == "pdf"
    assert meta["has_text_layer"] is True
    assert "Acme Widgets Pvt Ltd" in text
    assert "Director A (DIN 00000001)" in text
    assert "Beta Supplies LLP" in text


def test_docx_text_only_fallback(parser, monkeypatch):
    from src.rapidocr_parser import DOCXParser
    monkeypatch.setattr(DOCXParser, "_docx_to_images", lambda self, path: [])

    _, text, meta = _single_page(parser.parse(FIXTURES_DIR / "acme_board_minutes.docx"))
    assert meta["original_format"] == "docx"
    assert "warning" in meta
    assert "Acme Widgets Private Limited" in text
    assert "Beta Supplies LLP" in text


@pytest.mark.skipif(shutil.which("libreoffice") is None, reason="LibreOffice not installed")
def test_docx_rendered_with_libreoffice(parser):
    pairs = parser.parse(FIXTURES_DIR / "acme_board_minutes.docx")
    image, text, meta = _single_page(pairs)
    assert "warning" not in meta
    assert image.size[0] > 1000  # rendered page, not the blank placeholder
    assert "Director B" in text


def test_xlsx_sheet_text_and_render(parser):
    image, text, meta = _single_page(parser.parse(FIXTURES_DIR / "acme_related_parties.xlsx"))
    assert meta["original_format"] == "xlsx"
    assert meta["sheet_name"] == "Related Parties"
    assert meta["rows"] == 3 and meta["columns"] == 4
    assert text.startswith("SHEET: Related Parties")
    assert "Beta Supplies LLP" in text and "1200000" in text
    assert image.size[0] > 100 and image.size[1] > 100


def test_xlsx_does_not_need_matplotlib(parser, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    pairs = parser.parse(FIXTURES_DIR / "acme_related_parties.xlsx")
    assert len(pairs) == 1


def test_image_ocr(parser):
    _, text, meta = _single_page(parser.parse(FIXTURES_DIR / "acme_letterhead.png"))
    assert meta["is_image"] is True
    assert meta["ocr_confidence"] > 0.5
    lowered = text.lower()
    assert "director a" in lowered
    assert "acme" in lowered


def test_unsupported_extension(parser, tmp_path):
    bogus = tmp_path / "notes.xml"
    bogus.write_text("<note>Acme Widgets</note>")
    with pytest.raises(ValueError, match="No parser found"):
        parser.parse(bogus)
