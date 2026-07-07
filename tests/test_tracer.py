import json
import os

from agent.types import ActionResult
from observability.tracer import Tracer


def test_log_step_appends_one_json_line_per_call(tmp_path):
    # Arrange
    tracer = Tracer(session_id="test-session", base_dir=str(tmp_path))
    result = ActionResult(success=True, new_url="http://x/", error=None, screenshot_b64="")

    # Act
    tracer.log_step(1, "http://x/", "navigate_to", {"url": "http://x/"}, result, "thinking", 50)
    tracer.log_step(2, "http://x/", "click", {"element_id": "btn_0"}, result, "thinking more", 60)

    # Assert
    with open(tracer.trace_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["action_type"] == "navigate_to"
    assert lines[0]["gemini_reasoning"] == "thinking"
    assert lines[1]["tokens_used"] == 60


def test_log_step_saves_screenshot_when_present(tmp_path):
    # Arrange
    tracer = Tracer(session_id="test-session", base_dir=str(tmp_path))
    fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    result = ActionResult(success=True, new_url="http://x/", error=None, screenshot_b64=fake_b64)

    # Act
    tracer.log_step(1, "http://x/", "navigate_to", {}, result, "thinking", 10)

    # Assert
    screenshot_path = os.path.join(str(tmp_path), "test-session", "step_1.png")
    assert os.path.exists(screenshot_path)


def test_log_step_prints_live_summary(tmp_path, capsys):
    # Arrange
    tracer = Tracer(session_id="test-session", base_dir=str(tmp_path))
    result = ActionResult(success=False, new_url="http://x/", error="missing element", screenshot_b64="")

    # Act
    tracer.log_step(3, "http://x/", "click", {"element_id": "btn_404"}, result, "try button", 12)

    # Assert
    output = capsys.readouterr().out
    assert "Step 3" in output
    assert "click" in output
    assert "FAIL" in output
