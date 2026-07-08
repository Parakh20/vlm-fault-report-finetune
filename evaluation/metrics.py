# evaluation/metrics.py
"""Aggregate per-task judge scores and agent runs into benchmark summary metrics."""

CATEGORY_LABELS = {
    "information_extraction": "Information Extraction",
    "multi_step_navigation": "Multi-step Navigation",
    "form_interaction": "Form Interaction",
    "data_aggregation": "Data Aggregation",
    "reasoning_over_content": "Reasoning Over Content",
    "shopping": "Shopping",
    "wikipedia": "Wikipedia",
    "tables": "Tables",
    "downloads": "Downloads",
    "authentication": "Authentication",
    "pagination": "Pagination",
    "infinite_scroll": "Infinite Scroll",
    "captcha_detection": "CAPTCHA Detection",
    "dynamic_content": "Dynamic Content",
    "file_uploads": "File Uploads",
}


def _failed_step_count(run) -> int:
    return sum(
        1
        for step in run.steps
        if step.action_result is not None and not step.action_result.success
    )


def _recovered_from_a_failure(run) -> bool:
    """True if the task ultimately succeeded despite at least one failed
    step along the way — evidence the retry/repair/validation layers did
    their job instead of the agent just getting lucky on the first try."""
    return run.success and _failed_step_count(run) > 0


def _hallucinated_completion(run) -> bool:
    """True if the agent called task_complete more than once — the earlier
    claim(s) were rejected by the Verifier before one was finally accepted
    (or the run ended without ever being accepted)."""
    task_complete_attempts = sum(1 for step in run.steps if step.action_type == "task_complete")
    return task_complete_attempts > 1


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute overall and per-category success rates, efficiency averages, and failure modes.

    Each result dict must have: task_id, category, run (AgentRun), judge_score (JudgeScore),
    and an optional failure_mode.
    """
    total = len(results)
    passed = sum(1 for r in results if r["run"].success)

    by_category: dict[str, dict] = {}
    failure_modes: dict[str, int] = {}

    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"passed": 0, "total": 0})
        by_category[cat]["total"] += 1
        if r["run"].success:
            by_category[cat]["passed"] += 1
        if r.get("failure_mode"):
            failure_modes[r["failure_mode"]] = failure_modes.get(r["failure_mode"], 0) + 1

    recovered = sum(1 for r in results if _recovered_from_a_failure(r["run"]))
    hallucinated = sum(1 for r in results if _hallucinated_completion(r["run"]))
    total_retries = sum(_failed_step_count(r["run"]) for r in results)

    return {
        "overall_success_rate": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "by_category": by_category,
        "avg_steps": sum(r["run"].total_actions for r in results) / total if total else 0.0,
        "avg_tokens": sum(r["run"].total_tokens for r in results) / total if total else 0.0,
        "avg_time": sum(r["run"].duration_seconds for r in results) / total if total else 0.0,
        "avg_retries": total_retries / total if total else 0.0,
        "recovery_rate": recovered / total if total else 0.0,
        "hallucination_rate": hallucinated / total if total else 0.0,
        "failure_modes": failure_modes,
    }


def write_summary(metrics: dict, best_run: dict, out_path: str) -> None:
    """Write a human-readable benchmark summary report to out_path."""
    lines = []
    lines.append(f"BENCHMARK RESULTS ({metrics['total']} tasks):")
    lines.append(f"  Overall success rate:   {metrics['passed']} / {metrics['total']} "
                 f"({metrics['overall_success_rate'] * 100:.0f}%)\n")
    lines.append("  By category:")
    for cat, label in CATEGORY_LABELS.items():
        stats = metrics["by_category"].get(cat, {"passed": 0, "total": 0})
        lines.append(f"    {label}: {stats['passed']}/{stats['total']}")
    lines.append("")
    lines.append("  Efficiency:")
    lines.append(f"    Avg steps per task:      {metrics['avg_steps']:.1f}")
    lines.append(f"    Avg tokens per task:     {metrics['avg_tokens']:.0f}")
    lines.append(f"    Avg time per task:       {metrics['avg_time']:.0f}s")
    lines.append(f"    Avg retries per task:    {metrics['avg_retries']:.1f}\n")
    lines.append("  Reliability:")
    lines.append(f"    Recovery rate:           {metrics['recovery_rate'] * 100:.0f}% "
                 "(succeeded despite a mid-run failure)")
    lines.append(f"    Hallucination rate:      {metrics['hallucination_rate'] * 100:.0f}% "
                 "(claimed done, verifier rejected at least once)\n")
    lines.append("  Failure modes:")
    if metrics["failure_modes"]:
        for mode, count in metrics["failure_modes"].items():
            lines.append(f"    {mode}: {count} tasks")
    else:
        lines.append("    none")
    lines.append("")
    lines.append("MOST IMPRESSIVE SUCCESSFUL TASK:")
    lines.append(f"  Task: {best_run['run'].task}")
    lines.append(f"  Result: {best_run['run'].result}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("Run scripts/run_benchmark.py to generate real metrics.")
