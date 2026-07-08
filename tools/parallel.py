import asyncio

from playwright.async_api import BrowserContext, Page

MAX_PARALLEL_URLS = 5
PER_PAGE_TIMEOUT_MS = 15000


async def _fetch_one(context: BrowserContext, url: str) -> dict:
    tab = await context.new_page()
    try:
        await tab.goto(url, wait_until="load", timeout=PER_PAGE_TIMEOUT_MS)
        text = await tab.evaluate("document.body.innerText")
        return {"url": url, "success": True, "text": text[:2000], "error": None}
    except Exception as exc:  # noqa: BLE001 - one bad URL shouldn't fail the batch
        return {"url": url, "success": False, "text": "", "error": str(exc)}
    finally:
        await tab.close()


async def extract_from_urls_parallel(page: Page, urls: list[str]) -> str:
    """Opens up to `MAX_PARALLEL_URLS` URLs concurrently in new tabs on the
    same browser context and extracts their visible text, for research-style
    tasks that need to compare several pages at once instead of visiting
    them one at a time through the main tab."""
    urls = urls[:MAX_PARALLEL_URLS]
    if not urls:
        return "(no urls provided)"
    results = await asyncio.gather(*(_fetch_one(page.context, url) for url in urls))

    sections = []
    for result in results:
        if result["success"]:
            sections.append(f"### {result['url']}\n{result['text']}")
        else:
            sections.append(f"### {result['url']} (failed: {result['error']})")
    return "\n\n".join(sections)


if __name__ == "__main__":

    async def _demo() -> None:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await extract_from_urls_parallel(page, ["https://example.com"]))
            await browser.close()

    asyncio.run(_demo())
