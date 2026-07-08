# Task 15 Report

## Branch

- `task-15`

## Changed Paths

- `scripts/run_task.py`
- `tests/test_run_task.py`
- `.superpowers/sdd/task-15-report.md`

## Red Phase

Command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/test_run_task.py -v
```

Result:

```text
ERROR tests/test_run_task.py
ModuleNotFoundError: No module named 'scripts.run_task'
1 error in 0.05s
```

This matched the expected failure after adding the parser tests first.

## Green Phase

Focused command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/test_run_task.py -v
```

Result:

```text
2 passed in 0.37s
```

Full suite command:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python -m pytest tests/ -v
```

Result:

```text
68 passed in 22.45s
```

CLI help smoke:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python scripts/run_task.py --help
```

Result: printed the expected argparse usage.

## Live Smoke

The task worktree does not contain the ignored `.env`, so the first live smoke failed before Gemini with `RuntimeError: GEMINI_API_KEY is not set`. Re-running with the main checkout's `.env` sourced succeeded:

```bash
/home/parakh/Desktop/Project/AI_ML_SDE/auto_web_agent/.venv/bin/python scripts/run_task.py "Go to https://example.com and report the page title" --headless --max-steps 5
```

Result:

```text
Success: True
Result: Example Domain
Steps taken: 2
Tokens used: 2766
Duration: 6.0s
Final URL: https://example.com/
```

## Review Notes

- `run_task(...)` loads settings, starts `BrowserSession`, runs `WebAgentReasoner`, and always stops the session.
- `build_arg_parser()` supports required task text, `--headless`, and `--max-steps`.
- `main()` prints success/result/actions/tokens/duration/final URL.
- Direct script execution is supported with a project-root path guard.
