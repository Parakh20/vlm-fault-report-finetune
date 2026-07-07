# Task 21 Report

## Branch

- `task-21`

## Changed Paths

- `scripts/run_benchmark.py`
- `tests/test_run_benchmark.py`
- `.superpowers/sdd/task-21-report.md`

## Red Phase

Command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/test_run_benchmark.py -v
```

Result:

```text
ERROR tests/test_run_benchmark.py
ModuleNotFoundError: No module named 'scripts.run_benchmark'
1 error in 0.05s
```

This matched the expected failure after adding the classification tests first.

## Green Phase

Focused command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/test_run_benchmark.py -v
```

Result:

```text
5 passed in 0.67s
```

Full suite command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/ -v
```

Result:

```text
73 passed in 23.42s
```

Import smoke:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python - <<'PY'
from scripts.run_benchmark import classify_failure_mode
from agent.types import AgentRun
print(classify_failure_mode(AgentRun(task='x', success=True, result='ok')))
PY
```

Result:

```text
None
```

## Live Smoke / Benchmark Status

Attempted the brief's one-task benchmark smoke with a simple `https://example.com` task and the main checkout's `.env` sourced. The run reached Gemini but failed with:

```text
429 RESOURCE_EXHAUSTED ... Quota exceeded for metric:
generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 20, model: gemini-2.5-flash
```

The full 25-task benchmark was not run. No benchmark results were fabricated; live benchmark results remain pending a funded or non-exhausted `GEMINI_API_KEY`.

## Review Notes

- `classify_failure_mode(...)` implements the requested max-step/login/CAPTCHA/navigation heuristics.
- `run_single_task(...)` starts a headless browser session, attaches a tracer, runs `WebAgentReasoner`, scores with `LLMJudge`, and returns the result shape consumed by `aggregate_metrics`.
- `run_benchmark(...)` loops all benchmark tasks, writes per-task JSON, writes `results/benchmark_summary.csv`, and writes `results/summary.txt`.
- Direct script execution is supported with a project-root path guard.
