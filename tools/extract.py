from playwright.async_api import Page


async def get_page_text(page: Page) -> str:
    return await page.evaluate("document.body.innerText")


async def extract_table(page: Page, element_id: str) -> str:
    rows = await page.eval_on_selector_all(
        f"{element_id} tr",
        "rows => rows.map(r => Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.trim()))",
    )
    if not rows:
        return ""

    header, *body = rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await get_page_text(page))
            await browser.close()

    asyncio.run(_demo())
