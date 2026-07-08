import pytest

from perception.ocr import extract_ocr_text, is_ocr_available


@pytest.mark.asyncio
async def test_extract_ocr_text_returns_empty_string_for_empty_input():
    # Act
    text = await extract_ocr_text("")

    # Assert
    assert text == ""


@pytest.mark.asyncio
async def test_extract_ocr_text_does_not_raise_when_tesseract_unavailable(monkeypatch):
    # Arrange: simulate the common case in this environment (no tesseract binary).
    monkeypatch.setattr("perception.ocr._OCR_AVAILABLE", False)

    # Act
    text = await extract_ocr_text("aGVsbG8=")

    # Assert
    assert text == ""


def test_is_ocr_available_reports_a_bool():
    # Act / Assert
    assert isinstance(is_ocr_available(), bool)
