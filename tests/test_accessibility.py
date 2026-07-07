import re

import pytest
from perception.accessibility import extract_accessibility_tree

# Matches the contract line shape: [role] "name" extra="value"
BRACKET_NODE_PATTERN = re.compile(r'^\[[a-zA-Z][\w-]*\](\s+"[^"]*")?(\s+\w+="[^"]*")*$')


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


@pytest.mark.asyncio
async def test_every_line_matches_bracket_role_name_format(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)
    lines = [line for line in tree.splitlines() if line.strip()]

    # Assert: every non-empty line follows `[role] "name" extra="value"`,
    # not Playwright's raw YAML-like aria_snapshot syntax (e.g. `- link "Home":`)
    assert lines, "expected at least one node line"
    for line in lines:
        stripped = line.strip()
        assert not stripped.startswith("-"), f"line still in raw aria_snapshot syntax: {line!r}"
        assert BRACKET_NODE_PATTERN.match(stripped), f"line does not match bracket format: {line!r}"


@pytest.mark.asyncio
async def test_button_line_has_bracket_role_and_quoted_name(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)

    # Assert: the Search button is rendered in the specified bracket shape
    assert '[button] "Search"' in tree
