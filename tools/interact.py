"""Interaction tools: the element-id-to-locator bridge.

`perception.dom_parser.extract_interactive_elements` assigns stable ids like
"btn_0", "link_1" by enumerating elements matched by `INTERACTIVE_SELECTOR`
in DOM order. This module re-derives the same Playwright `ElementHandle` for
a given id by re-running that same extraction/selector query immediately
before acting, relying on element order being stable between calls as long
as the page hasn't navigated or mutated in between.
"""

from playwright.async_api import ElementHandle, Page

from perception.dom_parser import INTERACTIVE_SELECTOR, extract_interactive_elements


async def _handle_for(page: Page, element_id: str) -> ElementHandle:
    """Resolve an element id to its current Playwright ElementHandle.

    Raises ValueError if no element with `element_id` exists on the page.
    """
    elements = await extract_interactive_elements(page)
    index = next((i for i, e in enumerate(elements) if e.id == element_id), None)
    if index is None:
        raise ValueError(f"Element id '{element_id}' not found on current page")

    handles = await page.query_selector_all(INTERACTIVE_SELECTOR)
    visible_handles = []
    for handle in handles:
        if await handle.is_visible():
            visible_handles.append(handle)
    return visible_handles[index]


async def click(page: Page, element_id: str) -> None:
    """Click the element identified by `element_id`."""
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.click()
    await page.wait_for_load_state("domcontentloaded", timeout=5000)


async def type_text(page: Page, element_id: str, text: str, clear_first: bool = True) -> None:
    """Type `text` into the element identified by `element_id`.

    Clears the existing value first unless `clear_first` is False.
    """
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.click()
    if clear_first:
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
    await page.keyboard.type(text)


async def scroll(page: Page, direction: str, amount: int = 300) -> None:
    """Scroll the page in `direction` ("up", "down", "top", "bottom") by `amount` pixels."""
    deltas = {"up": (0, -amount), "down": (0, amount)}
    if direction in deltas:
        dx, dy = deltas[direction]
        await page.mouse.wheel(dx, dy)
        # The wheel event is dispatched asynchronously in headless Chromium;
        # give the compositor a moment to apply the resulting scroll offset.
        await page.wait_for_timeout(100)
    elif direction == "top":
        await page.evaluate("window.scrollTo(0, 0)")
    elif direction == "bottom":
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    else:
        raise ValueError(f"Unknown scroll direction: {direction}")


async def select_option(page: Page, element_id: str, value: str) -> None:
    """Select `value` in the <select> element identified by `element_id`."""
    handle = await _handle_for(page, element_id)
    await handle.select_option(value=value)


async def press_key(page: Page, key: str) -> None:
    """Press a keyboard key (e.g. "Enter", "Backspace") on the currently focused element."""
    await page.keyboard.press(key)


async def hover(page: Page, element_id: str) -> None:
    """Hover over the element identified by `element_id`."""
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.hover()


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print("demo ready")
            await browser.close()

    asyncio.run(_demo())
