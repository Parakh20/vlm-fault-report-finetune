import pytest
from tools.navigate import go_back, go_forward, go_to_url


@pytest.mark.asyncio
async def test_go_to_url_navigates_and_waits_for_load(browser_page, static_server):
    # Act
    await go_to_url(browser_page, f"{static_server}/sample_page.html")

    # Assert
    assert browser_page.url.endswith("sample_page.html")
    assert await browser_page.title() == "Fixture Page"


@pytest.mark.asyncio
async def test_go_back_and_forward_traverse_history(browser_page, static_server):
    # Arrange
    await go_to_url(browser_page, f"{static_server}/sample_page.html")
    await browser_page.click("#about-link")
    assert browser_page.url.endswith("about.html")

    # Act
    await go_back(browser_page)
    # Assert
    assert browser_page.url.endswith("sample_page.html")

    # Act
    await go_forward(browser_page)
    # Assert
    assert browser_page.url.endswith("about.html")
