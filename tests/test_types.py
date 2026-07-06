from agent.types import ActionResult, Element, PageState


def test_element_holds_stable_id_and_bbox():
    # Arrange / Act
    el = Element(
        id="btn_0", tag="button", text="Search", role="button",
        placeholder=None, href=None, visible=True,
        bbox={"x": 1.0, "y": 2.0, "width": 50.0, "height": 20.0},
    )

    # Assert
    assert el.id == "btn_0"
    assert el.bbox["width"] == 50.0


def test_page_state_holds_elements_and_metadata():
    # Arrange
    el = Element("link_0", "a", "About", "link", None, "/about", True, {})

    # Act
    state = PageState(
        url="http://x/", title="X", screenshot_b64="", interactive_elements=[el],
        accessibility_tree="", scroll_y=0, page_height=1000,
        dialog_visible=False, dialog_text=None,
    )

    # Assert
    assert state.interactive_elements[0].href == "/about"


def test_action_result_carries_error_on_failure():
    # Arrange / Act
    result = ActionResult(success=False, new_url="http://x/", error="not found", screenshot_b64="")

    # Assert
    assert result.success is False
    assert result.error == "not found"
