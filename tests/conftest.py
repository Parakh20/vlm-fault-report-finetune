# tests/conftest.py
import functools
import http.server
import threading

import pytest
import pytest_asyncio
from playwright.async_api import async_playwright

FIXTURES_DIR = "tests/fixtures"


@pytest.fixture(scope="session")
def static_server():
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=FIXTURES_DIR
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest_asyncio.fixture
async def browser_page():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        yield page
        await browser.close()
