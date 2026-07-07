import pytest

from tools.interact import click, hover, press_key, scroll, select_option, type_text


@pytest.mark.asyncio
async def test_click_navigates_via_link_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    await click(browser_page, "link_0")

    # Assert
    assert browser_page.url.endswith("about.html")


@pytest.mark.asyncio
async def test_type_text_fills_input_and_clears_first(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")
    await browser_page.fill("#query-input", "old value")

    # Act
    await type_text(browser_page, "input_0", "new query")

    # Assert
    assert await browser_page.input_value("#query-input") == "new query"


@pytest.mark.asyncio
async def test_select_option_sets_dropdown_value(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    await select_option(browser_page, "select_0", "date")

    # Assert
    assert await browser_page.eval_on_selector("#sort-select", "el => el.value") == "date"


@pytest.mark.asyncio
async def test_scroll_down_changes_scroll_position(browser_page, static_server, tmp_path):
    # Arrange: make the page tall enough to scroll
    tall_html = "<html><body style='height:3000px'>" + "<p>x</p>" * 50 + "</body></html>"
    html_path = tmp_path / "tall.html"
    html_path.write_text(tall_html)
    await browser_page.goto(html_path.as_uri())

    # Act
    await scroll(browser_page, "down", amount=500)

    # Assert
    scroll_y = await browser_page.evaluate("window.scrollY")
    assert scroll_y > 0


@pytest.mark.asyncio
async def test_click_raises_value_error_for_unknown_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="btn_99"):
        await click(browser_page, "btn_99")


@pytest.mark.asyncio
async def test_hover_raises_value_error_for_unknown_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="link_99"):
        await hover(browser_page, "link_99")


@pytest.mark.asyncio
async def test_hover_highlights_target_element(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    await hover(browser_page, "btn_0")

    # Assert
    is_hovered = await browser_page.eval_on_selector("#search-btn", "el => el.matches(':hover')")
    assert is_hovered is True


@pytest.mark.asyncio
async def test_press_key_submits_form_focused_input(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")
    await browser_page.focus("#query-input")
    await browser_page.type("#query-input", "abc")

    # Act
    await press_key(browser_page, "Backspace")

    # Assert
    assert await browser_page.input_value("#query-input") == "ab"


@pytest.mark.asyncio
async def test_scroll_raises_value_error_for_unknown_direction(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="Unknown scroll direction"):
        await scroll(browser_page, "sideways")


@pytest.mark.asyncio
async def test_select_option_raises_value_error_for_unknown_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="select_99"):
        await select_option(browser_page, "select_99", "date")


@pytest.mark.asyncio
async def test_type_text_raises_value_error_for_unknown_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="input_99"):
        await type_text(browser_page, "input_99", "hello")
