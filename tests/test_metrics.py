import os

from agent.types import ActionResult, AgentRun, Step
from evaluation.judge import JudgeScore
from evaluation.metrics import aggregate_metrics, write_summary


def _result(task_id, category, success, steps, tokens, seconds, failure_mode=None, step_list=None):
    run = AgentRun(
        task=f"task for {task_id}", success=success, result="result text",
        total_actions=steps, total_tokens=tokens, duration_seconds=seconds,
        steps=step_list or [],
    )
    score = JudgeScore(task_completed=success, accuracy=4, completeness=4, efficiency=4)
    return {"task_id": task_id, "category": category, "run": run, "judge_score": score, "failure_mode": failure_mode}


def _step(action_type, success):
    result = ActionResult(success=success, new_url="http://x/", error=None, screenshot_b64="")
    return Step(1, "http://x/", action_type, {}, result, "thinking", 10)


def test_aggregate_metrics_computes_overall_and_per_category_rates():
    # Arrange
    results = [
        _result("ie_01", "information_extraction", True, 4, 1000, 10.0),
        _result("ie_02", "information_extraction", False, 25, 5000, 60.0, failure_mode="max_steps_reached"),
        _result("nav_01", "multi_step_navigation", True, 6, 1200, 15.0),
    ]

    # Act
    metrics = aggregate_metrics(results)

    # Assert
    assert metrics["overall_success_rate"] == 2 / 3
    assert metrics["by_category"]["information_extraction"] == {"passed": 1, "total": 2}
    assert metrics["by_category"]["multi_step_navigation"] == {"passed": 1, "total": 1}
    assert metrics["avg_steps"] == (4 + 25 + 6) / 3
    assert metrics["failure_modes"] == {"max_steps_reached": 1}


def test_aggregate_metrics_computes_recovery_and_retries():
    # Arrange: task succeeded, but only after one failed step.
    results = [
        _result(
            "nav_01", "multi_step_navigation", True, 3, 500, 5.0,
            step_list=[_step("click", False), _step("click", True), _step("task_complete", True)],
        ),
        _result("ie_01", "information_extraction", True, 2, 500, 5.0, step_list=[_step("task_complete", True)]),
    ]

    # Act
    metrics = aggregate_metrics(results)

    # Assert
    assert metrics["avg_retries"] == 0.5  # 1 failed step across 2 tasks
    assert metrics["recovery_rate"] == 0.5  # only the first task recovered from a failure


def test_aggregate_metrics_detects_hallucinated_completion():
    # Arrange: verifier rejected the first task_complete claim, agent tried again.
    results = [
        _result(
            "ie_01", "information_extraction", True, 2, 500, 5.0,
            step_list=[_step("task_complete", True), _step("task_complete", True)],
        ),
        _result("ie_02", "information_extraction", True, 1, 500, 5.0, step_list=[_step("task_complete", True)]),
    ]

    # Act
    metrics = aggregate_metrics(results)

    # Assert
    assert metrics["hallucination_rate"] == 0.5


def test_write_summary_produces_readable_text_file(tmp_path):
    # Arrange
    results = [_result("ie_01", "information_extraction", True, 4, 1000, 10.0)]
    metrics = aggregate_metrics(results)
    out_path = str(tmp_path / "summary.txt")

    # Act
    write_summary(metrics, best_run=results[0], out_path=out_path)

    # Assert
    assert os.path.exists(out_path)
    content = open(out_path).read()
    assert "BENCHMARK RESULTS" in content
    assert "Overall success rate" in content
    assert "MOST IMPRESSIVE SUCCESSFUL TASK" in content
