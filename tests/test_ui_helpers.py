from agent.types import ActionResult, Step
from ui.app import format_step_line


def test_format_step_line_includes_step_number_action_and_status():
    # Arrange
    result = ActionResult(
        success=True,
        new_url="http://x/",
        error=None,
        screenshot_b64="",
    )
    step = Step(
        number=2,
        url="http://x/",
        action_type="click",
        action_input={"element_id": "btn_0"},
        action_result=result,
        reasoning_text="clicking search button",
        tokens_used=80,
    )

    # Act
    line = format_step_line(step)

    # Assert
    assert "[Step 2]" in line
    assert "click" in line
    assert "OK" in line
