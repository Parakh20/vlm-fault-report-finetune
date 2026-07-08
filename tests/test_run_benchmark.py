from agent.types import AgentRun
from scripts.run_benchmark import classify_failure_mode


def test_classifies_max_steps_reached():
    run = AgentRun(task="x", success=False, result="", total_actions=25)
    assert classify_failure_mode(run) == "max_steps_reached"


def test_classifies_login_required():
    run = AgentRun(
        task="x",
        success=False,
        result="Login required to view this page",
        total_actions=4,
    )
    assert classify_failure_mode(run) == "login_required"


def test_classifies_captcha_blocked():
    run = AgentRun(
        task="x",
        success=False,
        result="Blocked by a CAPTCHA challenge",
        total_actions=2,
    )
    assert classify_failure_mode(run) == "captcha_blocked"


def test_classifies_navigation_error_as_fallback():
    run = AgentRun(
        task="x",
        success=False,
        result="Could not find the requested element",
        total_actions=10,
    )
    assert classify_failure_mode(run) == "navigation_error"


def test_returns_none_for_successful_run():
    run = AgentRun(task="x", success=True, result="done", total_actions=5)
    assert classify_failure_mode(run) is None
