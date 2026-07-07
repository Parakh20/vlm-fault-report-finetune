import pytest

from agent.browser import BrowserSession


@pytest.mark.asyncio
async def test_start_and_get_page_state_returns_url_and_elements(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        state = await session.get_page_state()

        # Assert
        assert state.url.endswith("sample_page.html")
        assert state.title == "Fixture Page"
        assert any(e.id == "btn_0" for e in state.interactive_elements)
        assert state.screenshot_b64 != ""
        assert state.dialog_visible is False
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_stop_closes_browser_cleanly(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    # Act / Assert (no exception means clean shutdown)
    await session.stop()
