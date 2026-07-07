import pytest

from agent.actions import ACTION_SCHEMAS, _TEXT_RETURNING, dispatch_action
from agent.browser import BrowserSession


def test_action_schemas_include_all_required_actions():
    # Arrange
    names = {schema["name"] for schema in ACTION_SCHEMAS}

    # Assert
    expected = {
        "navigate_to",
        "click",
        "type_text",
        "scroll",
        "select_option",
        "press_key",
        "hover",
        "wait",
        "get_page_text",
        "search_web",
        "extract_table",
        "go_back",
        "task_complete",
        "task_failed",
    }
    assert expected.issubset(names)


def test_text_returning_actions_are_exported_for_reasoning_loop():
    # Assert
    assert _TEXT_RETURNING == {"get_page_text", "search_web", "extract_table"}


@pytest.mark.asyncio
async def test_dispatch_navigate_to_succeeds_and_returns_screenshot(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        # Act
        result = await dispatch_action(
            session, "navigate_to", {"url": f"{static_server}/sample_page.html"}
        )

        # Assert
        assert result.success is True
        assert result.new_url.endswith("sample_page.html")
        assert result.error is None
        assert result.screenshot_b64 != ""
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_dispatch_get_page_text_returns_payload_in_error_field(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        result = await dispatch_action(session, "get_page_text", {})

        # Assert
        assert result.success is True
        assert result.new_url.endswith("sample_page.html")
        assert "Test Fixture" in result.error
        assert result.screenshot_b64 != ""
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_dispatch_extract_table_returns_markdown_payload(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        result = await dispatch_action(session, "extract_table", {"selector": "#data-table"})

        # Assert
        assert result.success is True
        assert "| Name | Price |" in result.error
        assert "| Widget | $10 |" in result.error
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_dispatch_click_unknown_element_returns_failure_result(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        result = await dispatch_action(session, "click", {"element_id": "btn_99"})

        # Assert
        assert result.success is False
        assert "btn_99" in result.error
        assert result.screenshot_b64 != ""
    finally:
        await session.stop()
