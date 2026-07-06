from playwright.async_api import Page


async def extract_accessibility_tree(page: Page) -> str:
    """
    Extract accessibility tree from the page using aria_snapshot.

    Returns an indented text tree with accessibility roles and names.
    Empty or invisible nodes are automatically excluded by aria_snapshot.
    """
    snapshot = await page.aria_snapshot()
    return snapshot if snapshot else ""


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await extract_accessibility_tree(page))
            await browser.close()

    asyncio.run(_demo())
