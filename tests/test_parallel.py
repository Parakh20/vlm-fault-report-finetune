import pytest
import pytest_asyncio
from playwright.async_api import async_playwright

from tools.parallel import extract_from_urls_parallel


@pytest_asyncio.fixture
async def context_page():
    # extract_from_urls_parallel opens sibling tabs via page.context.new_page(),
    # which requires a real multi-page BrowserContext (browser.new_context()),
    # not the single-owner context that browser.new_page() implicitly creates.
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        yield page
        await browser.close()


@pytest.mark.asyncio
async def test_extract_from_urls_parallel_fetches_multiple_pages(context_page, static_server):
    # Arrange
    await context_page.goto(f"{static_server}/sample_page.html")
    urls = [f"{static_server}/sample_page.html", f"{static_server}/sample_page.html"]

    # Act
    result = await extract_from_urls_parallel(context_page, urls)

    # Assert
    assert result.count(f"### {static_server}/sample_page.html") == 2


@pytest.mark.asyncio
async def test_extract_from_urls_parallel_reports_per_url_failures(context_page, static_server):
    # Arrange
    await context_page.goto(f"{static_server}/sample_page.html")
    urls = [f"{static_server}/sample_page.html", "http://127.0.0.1:1/does-not-exist"]

    # Act
    result = await extract_from_urls_parallel(context_page, urls)

    # Assert
    assert "failed" in result
    assert f"### {static_server}/sample_page.html" in result


@pytest.mark.asyncio
async def test_extract_from_urls_parallel_caps_at_max_urls(context_page, static_server):
    # Arrange
    await context_page.goto(f"{static_server}/sample_page.html")
    urls = [f"{static_server}/sample_page.html"] * 10

    # Act
    result = await extract_from_urls_parallel(context_page, urls)

    # Assert
    assert result.count("###") == 5


@pytest.mark.asyncio
async def test_extract_from_urls_parallel_handles_empty_list(context_page, static_server):
    # Arrange
    await context_page.goto(f"{static_server}/sample_page.html")

    # Act
    result = await extract_from_urls_parallel(context_page, [])

    # Assert
    assert "no urls" in result
