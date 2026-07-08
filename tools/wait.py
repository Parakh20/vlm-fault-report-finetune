import asyncio

from playwright.async_api import Page

MAX_WAIT_SECONDS = 5.0


async def wait(seconds: float) -> None:
    await asyncio.sleep(min(seconds, MAX_WAIT_SECONDS))


async def wait_for_load(page: Page) -> None:
    await page.wait_for_load_state("load")


if __name__ == "__main__":
    asyncio.run(wait(1.0))
