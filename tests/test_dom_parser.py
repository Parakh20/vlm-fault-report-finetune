import pytest

from perception.dom_parser import MAX_ELEMENTS, extract_interactive_elements


@pytest.mark.asyncio
async def test_extracts_visible_interactive_elements(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    elements = await extract_interactive_elements(browser_page)
    ids = [e.id for e in elements]
    by_id = {e.id: e for e in elements}

    # Assert
    assert "link_0" in ids
    assert "btn_0" in ids
    assert "input_0" in ids
    assert "select_0" in ids
    assert by_id["link_0"].href == "/about.html"
    assert by_id["input_0"].placeholder == "Search query"


@pytest.mark.asyncio
async def test_excludes_hidden_elements(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    elements = await extract_interactive_elements(browser_page)
    texts = [e.text for e in elements]

    # Assert
    assert "Hidden text" not in texts


@pytest.mark.asyncio
async def test_excludes_hidden_interactive_element(browser_page, tmp_path):
    # Arrange
    html = (
        "<html><body>"
        '<button id="hidden-btn" style="display:none;">Hidden Button</button>'
        '<button id="visible-btn">Visible Button</button>'
        "</body></html>"
    )
    html_path = tmp_path / "hidden_interactive.html"
    html_path.write_text(html)
    await browser_page.goto(html_path.as_uri())

    # Act
    elements = await extract_interactive_elements(browser_page)
    texts = [e.text for e in elements]

    # Assert
    assert "Hidden Button" not in texts
    assert "Visible Button" in texts
    assert len(elements) == 1


@pytest.mark.asyncio
async def test_assigns_el_prefix_fallback_id_for_non_native_interactive_tags(
    browser_page, tmp_path
):
    # Arrange
    html = (
        "<html><body>"
        '<div role="button" id="div-role-btn">Div Button</div>'
        '<span onclick="void(0)" id="span-onclick">Span Click</span>'
        "</body></html>"
    )
    html_path = tmp_path / "fallback_prefix.html"
    html_path.write_text(html)
    await browser_page.goto(html_path.as_uri())

    # Act
    elements = await extract_interactive_elements(browser_page)
    ids = [e.id for e in elements]

    # Assert
    assert len(elements) == 2
    assert all(id_.startswith("el_") for id_ in ids)


@pytest.mark.asyncio
async def test_truncates_to_max_elements(browser_page, static_server, tmp_path):
    # Arrange
    many_buttons = "".join(f'<button id="b{i}">Btn {i}</button>' for i in range(150))
    html_path = tmp_path / "many.html"
    html_path.write_text(f"<html><body>{many_buttons}</body></html>")
    await browser_page.goto(html_path.as_uri())

    # Act
    elements = await extract_interactive_elements(browser_page)

    # Assert
    assert len(elements) == MAX_ELEMENTS


@pytest.mark.asyncio
async def test_marks_the_focused_element(browser_page, tmp_path):
    # Arrange: don't rely on <input autofocus> — Chromium only honors it
    # with a real user gesture, so focus explicitly instead.
    html_path = tmp_path / "focus.html"
    html_path.write_text('<html><body><button id="a">A</button><input id="b"></body></html>')
    await browser_page.goto(html_path.as_uri())
    await browser_page.focus("#b")

    # Act
    elements = await extract_interactive_elements(browser_page)

    # Assert
    focused = [e for e in elements if e.focused]
    assert len(focused) == 1
    assert focused[0].tag == "input"
