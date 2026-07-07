from playwright.async_api import Page


async def go_to_url(page: Page, url: str) -> None:
    await page.goto(url, wait_until="load")


async def go_back(page: Page) -> None:
    await page.go_back(wait_until="load")


async def go_forward(page: Page) -> None:
    await page.go_forward(wait_until="load")


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await go_to_url(page, "https://example.com")
            print(page.url)
            await browser.close()

    asyncio.run(_demo())
