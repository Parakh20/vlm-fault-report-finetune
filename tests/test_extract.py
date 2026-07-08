import pytest
from tools.extract import extract_table, get_page_text


@pytest.mark.asyncio
async def test_get_page_text_returns_visible_text(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    text = await get_page_text(browser_page)

    # Assert
    assert "This is sample text used by perception tests." in text
    assert "Hidden text" not in text


@pytest.mark.asyncio
async def test_extract_table_returns_markdown(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    markdown = await extract_table(browser_page, "#data-table")

    # Assert
    assert "| Name | Price |" in markdown
    assert "| Widget | $10 |" in markdown
    assert "| Gadget | $20 |" in markdown
