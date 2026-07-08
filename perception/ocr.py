from __future__ import annotations

import asyncio
import base64
import io

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:  # pytesseract / the tesseract binary isn't installed
    _OCR_AVAILABLE = False


def is_ocr_available() -> bool:
    return _OCR_AVAILABLE


def _ocr_sync(screenshot_b64: str) -> str:
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()


async def extract_ocr_text(screenshot_b64: str) -> str:
    """Best-effort OCR pass over the annotated screenshot, catching text
    baked into canvas/image content that the DOM/AX tree can't see (e.g.
    text rendered inside a <canvas> chart or an image-based banner).

    Returns "" when Tesseract isn't installed or OCR fails, rather than
    raising — OCR is a supplementary signal on top of DOM+AX+screenshot,
    not a required one, so its absence shouldn't break perception.
    """
    if not _OCR_AVAILABLE or not screenshot_b64:
        return ""
    try:
        return await asyncio.to_thread(_ocr_sync, screenshot_b64)
    except Exception:
        return ""
