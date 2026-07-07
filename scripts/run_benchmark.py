import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from agent.browser import BrowserSession
from agent.config import Settings, load_settings
from agent.reasoning import WebAgentReasoner
from agent.types import AgentRun
from evaluation.judge import LLMJudge
from evaluation.metrics import aggregate_metrics, write_summary
from evaluation.tasks import BENCHMARK_TASKS
from observability.tracer import Tracer

RESULTS_DIR = "results/benchmark"


def classify_failure_mode(run: AgentRun) -> str | None:
    if run.success:
        return None
    if run.total_actions >= 25:
        return "max_steps_reached"

    text = run.result.lower()
    if "login" in text or "sign in" in text:
        return "login_required"
    if "captcha" in text:
        return "captcha_blocked"
    return "navigation_error"


async def run_single_task(
    task: dict,
    settings: Settings,
    headless: bool = True,
) -> dict:
    session = BrowserSession()
    await session.start(headless=headless)
    tracer = Tracer(session_id=task["id"])
    try:
        reasoner = WebAgentReasoner(settings)
        run = await reasoner.run(task["prompt"], session, tracer=tracer)
    finally:
        await session.stop()

    judge = LLMJudge(settings)
    judge_score = judge.score(run)
    return {
        "task_id": task["id"],
        "category": task["category"],
        "run": run,
        "judge_score": judge_score,
        "failure_mode": classify_failure_mode(run),
    }


async def run_benchmark(headless: bool = True) -> None:
    settings = load_settings()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    results = []

    for task in BENCHMARK_TASKS:
        print(f"\n=== Running {task['id']} ({task['category']}) ===")
        result = await run_single_task(task, settings, headless=headless)
        results.append(result)
        out_path = os.path.join(RESULTS_DIR, f"{task['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_result_to_json_payload(result), f, indent=2, default=str)

    metrics = aggregate_metrics(results)
    best_run = max(
        results,
        key=lambda result: (
            result["run"].success,
            result["judge_score"].accuracy + result["judge_score"].completeness,
        ),
    )

    df = pd.DataFrame(
        [
            {
                "task_id": result["task_id"],
                "category": result["category"],
                "success": result["run"].success,
                "steps": result["run"].total_actions,
                "tokens": result["run"].total_tokens,
                "duration_s": result["run"].duration_seconds,
                "failure_mode": result["failure_mode"],
            }
            for result in results
        ]
    )
    df.to_csv("results/benchmark_summary.csv", index=False)
    write_summary(metrics, best_run, out_path="results/summary.txt")
    print("\nBenchmark complete. See results/summary.txt and results/benchmark_summary.csv")


def _result_to_json_payload(result: dict) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "category": result["category"],
        "run": _to_plain_data(result["run"]),
        "judge_score": _to_plain_data(result["judge_score"]),
        "failure_mode": result["failure_mode"],
    }


def _to_plain_data(value):
    if is_dataclass(value):
        return asdict(value)
    return value


if __name__ == "__main__":
    asyncio.run(run_benchmark(headless=True))
