import pytest

from tools.search import search_web


@pytest.mark.asyncio
async def test_search_web_returns_exactly_top_5_results_when_more_exist(
    browser_page, static_server
):
    # Arrange
    url = f"{static_server}/search_results.html"

    # Act
    results = await search_web(browser_page, "irrelevant query", search_url=url)

    # Assert
    numbered_lines = [
        line
        for line in results.splitlines()
        if line and line[0].isdigit() and line[1:3] == ". "
    ]
    assert len(numbered_lines) == 5


@pytest.mark.asyncio
async def test_search_web_formats_each_result_with_title_url_and_snippet(
    browser_page, static_server
):
    # Arrange
    url = f"{static_server}/search_results.html"

    # Act
    results = await search_web(browser_page, "irrelevant query", search_url=url)

    # Assert
    assert "1. Example Result 1" in results
    assert "http://example.invalid/1" in results
    assert "Example Result 1 is a short snippet sentence about topic one." in results
    assert "5. Example Result 5" in results
    assert "http://example.invalid/5" in results
    # Result 6/7 should be truncated away by the top-5 limit.
    assert "Example Result 6" not in results
    assert "Example Result 7" not in results


@pytest.mark.asyncio
async def test_search_web_returns_no_results_found_when_zero_matches(
    browser_page, static_server
):
    # Arrange
    url = f"{static_server}/search_results_empty.html"

    # Act
    results = await search_web(browser_page, "irrelevant query", search_url=url)

    # Assert
    assert results == "No results found."
