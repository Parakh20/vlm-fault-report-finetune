# tools/search.py
from urllib.parse import quote

from playwright.async_api import Page

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"

TOP_N_RESULTS = 5


async def _parse_results(page: Page) -> str:
    """Scrape the currently-loaded search results page and format the top N.

    Operates purely on whatever page is already loaded, so it can be
    exercised against a local fixture without any network access.
    """
    results = await page.eval_on_selector_all(
        ".result",
        """nodes => nodes.map(n => {
            const title = n.querySelector('.result__title a');
            const snippet = n.querySelector('.result__snippet');
            return {
                title: title ? title.innerText.trim() : '',
                url: title ? title.href : '',
                snippet: snippet ? snippet.innerText.trim() : '',
            };
        })""",
    )

    top_results = results[:TOP_N_RESULTS]
    if not top_results:
        return "No results found."

    lines = [
        f"{i}. {result['title']}\n   {result['url']}\n   {result['snippet']}\n"
        for i, result in enumerate(top_results, start=1)
    ]
    return "\n".join(lines)


async def search_web(page: Page, query: str, search_url: str | None = None) -> str:
    """Navigate to a search results page and return the top 5 results.

    Args:
        page: An already-open Playwright page.
        query: The search query text.
        search_url: Optional override for the URL to navigate to. When
            omitted, defaults to a live DuckDuckGo HTML search for `query`.
            Tests pass a local fixture URL here so nothing hits the network.

    Returns:
        Formatted top-5 results, or "No results found." when there are none.
    """
    url = search_url or DUCKDUCKGO_HTML_URL.format(query=quote(query))
    await page.goto(url, wait_until="load")
    return await _parse_results(page)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        # Standalone smoke test: this path hits the real DuckDuckGo site and
        # is intentionally not exercised by the test suite (which is fully
        # local via the `static_server` fixture in tests/conftest.py).
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            print(await search_web(page, "Playwright Python"))
            await browser.close()

    asyncio.run(_demo())
