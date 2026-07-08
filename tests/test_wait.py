import time

import pytest
from tools.wait import wait, wait_for_load


@pytest.mark.asyncio
async def test_wait_sleeps_for_requested_duration():
    # Arrange
    start = time.monotonic()

    # Act
    await wait(0.2)

    # Assert
    assert time.monotonic() - start >= 0.2


@pytest.mark.asyncio
async def test_wait_clamps_to_max_5_seconds():
    # Arrange
    start = time.monotonic()

    # Act
    await wait(100)

    # Assert
    elapsed = time.monotonic() - start
    assert elapsed <= 5.5


@pytest.mark.asyncio
async def test_wait_for_load_resolves_after_navigation(browser_page, static_server):
    # Act
    await browser_page.goto(f"{static_server}/sample_page.html", wait_until="commit")
    await wait_for_load(browser_page)

    # Assert
    assert await browser_page.title() == "Fixture Page"
