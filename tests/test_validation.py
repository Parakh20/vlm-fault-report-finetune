from agent.types import Element, PageState
from agent.validation import validate_action


def _state_with_elements(elements: list[Element]) -> PageState:
    return PageState(
        url="http://example.com",
        title="Test",
        screenshot_b64="",
        interactive_elements=elements,
        accessibility_tree="",
        scroll_y=0,
        page_height=1000,
        dialog_visible=False,
        dialog_text=None,
    )


def _element(element_id: str) -> Element:
    return Element(
        id=element_id,
        tag="button",
        text="Submit",
        role="button",
        placeholder=None,
        href=None,
        visible=True,
        bbox={"x": 0, "y": 0, "width": 10, "height": 10},
    )


def test_validate_action_passes_for_known_element_id():
    # Arrange
    state = _state_with_elements([_element("btn_3")])

    # Act
    result = validate_action("click", {"element_id": "btn_3"}, state)

    # Assert
    assert result.ok is True


def test_validate_action_fails_for_unknown_element_id():
    # Arrange
    state = _state_with_elements([_element("btn_3")])

    # Act
    result = validate_action("click", {"element_id": "btn_99"}, state)

    # Assert
    assert result.ok is False
    assert "btn_99" in result.reason


def test_validate_action_fails_for_missing_required_argument():
    # Arrange
    state = _state_with_elements([])

    # Act
    result = validate_action("type_text", {"element_id": "input_0"}, state)

    # Assert
    assert result.ok is False
    assert "text" in result.reason


def test_validate_action_fails_for_unknown_action_name():
    # Arrange
    state = _state_with_elements([])

    # Act
    result = validate_action("teleport", {}, state)

    # Assert
    assert result.ok is False
    assert "Unknown action" in result.reason


def test_validate_action_passes_for_actions_without_element_id():
    # Arrange
    state = _state_with_elements([])

    # Act
    result = validate_action("get_page_text", {}, state)

    # Assert
    assert result.ok is True


def test_validate_action_passes_for_navigate_to_with_url():
    # Arrange
    state = _state_with_elements([])

    # Act
    result = validate_action("navigate_to", {"url": "https://example.com"}, state)

    # Assert
    assert result.ok is True
