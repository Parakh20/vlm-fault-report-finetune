from playwright.async_api import Page

from agent.types import Element

INTERACTIVE_SELECTOR = (
    "a, button, input, select, textarea, "
    "[role=button], [role=link], [onclick]"
)
MAX_ELEMENTS = 100

_TAG_PREFIX = {
    "a": "link",
    "button": "btn",
    "input": "input",
    "select": "select",
    "textarea": "textarea",
}


async def extract_interactive_elements(page: Page) -> list[Element]:
    """
    Extract visible interactive elements from the page as a flat, capped list.

    Elements are matched via `INTERACTIVE_SELECTOR` (links, buttons, form
    controls, and elements with interactive roles/handlers), filtered to
    only those currently visible, and capped at `MAX_ELEMENTS`. Each element
    is assigned a stable id like "btn_0", "link_1", using a counter scoped
    per tag-category (unrecognized tags fall back to an "el" prefix).
    """
    handles = await page.query_selector_all(INTERACTIVE_SELECTOR)
    counters: dict[str, int] = {}
    elements: list[Element] = []

    for handle in handles:
        if len(elements) >= MAX_ELEMENTS:
            break
        if not await handle.is_visible():
            continue

        tag = (await handle.evaluate("el => el.tagName.toLowerCase()")) or ""
        prefix = _TAG_PREFIX.get(tag, "el")
        idx = counters.get(prefix, 0)
        counters[prefix] = idx + 1

        bbox = await handle.bounding_box() or {}
        text = (await handle.inner_text()).strip() if tag != "input" else ""
        role = (await handle.get_attribute("role")) or ""
        placeholder = await handle.get_attribute("placeholder")
        href = await handle.get_attribute("href")
        focused = await handle.evaluate("el => el === document.activeElement")

        elements.append(
            Element(
                id=f"{prefix}_{idx}",
                tag=tag,
                text=text,
                role=role,
                placeholder=placeholder,
                href=href,
                visible=True,
                bbox=bbox,
                focused=bool(focused),
            )
        )

    return elements


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            els = await extract_interactive_elements(page)
            for e in els:
                print(e)
            await browser.close()

    asyncio.run(_demo())
