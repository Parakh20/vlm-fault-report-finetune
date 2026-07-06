import pytest
from perception.accessibility import extract_accessibility_tree


@pytest.mark.asyncio
async def test_returns_indented_text_tree_with_roles_and_names(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)

    # Assert
    assert "button" in tree
    assert "Search" in tree
    assert "link" in tree


@pytest.mark.asyncio
async def test_drops_nodes_with_no_name_and_no_children(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)

    # Assert: no line is just an empty bracket pair with nothing else
    for line in tree.splitlines():
        assert line.strip() != "[]"
