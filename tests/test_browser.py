import os

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


def test_rejects_unsupported_browser_engine():
    # Act / Assert
    with pytest.raises(ValueError, match="Unsupported browser engine"):
        BrowserSession(engine="netscape")


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["chromium", "firefox", "webkit"])
async def test_start_works_with_each_supported_engine(engine, static_server):
    # Arrange
    session = BrowserSession(engine=engine)

    # Act
    await session.start(headless=True)
    try:
        await session.page.goto(f"{static_server}/sample_page.html")
        state = await session.get_page_state()

        # Assert
        assert state.url.endswith("sample_page.html")
        assert session.engine == engine
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_tracing_writes_a_trace_zip(static_server, tmp_path):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)
    trace_path = str(tmp_path / "trace.zip")

    try:
        await session.start_tracing()
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        await session.stop_tracing(trace_path)

        # Assert
        assert os.path.exists(trace_path)
        assert os.path.getsize(trace_path) > 0
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_start_with_record_video_dir_produces_a_video_file(static_server, tmp_path):
    # Arrange
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir, exist_ok=True)
    session = BrowserSession()

    # Act
    await session.start(headless=True, record_video_dir=video_dir)
    try:
        await session.page.goto(f"{static_server}/sample_page.html")
    finally:
        await session.stop()

    # Assert: video is flushed to disk once the context/browser closes
    videos = os.listdir(video_dir)
    assert len(videos) >= 1
