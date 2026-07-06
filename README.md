# Autonomous Web Agent

An autonomous agent that controls a real Chromium browser via Playwright and
uses Google Gemini (`gemini-2.5-flash`) as its reasoning engine in a ReAct
loop: perceive the page, decide the next action, execute it, repeat until
the task is done or 25 steps are exhausted.

## Architecture

```
Task → ReAct Loop → [Perceive page (DOM + a11y tree + screenshot)
                      | Reason with Gemini (function calling)
                      | Execute action via Playwright]
                  → repeat → Result
```

- **Perception** (`perception/`): extracts visible interactive elements, a
  simplified accessibility tree, and an annotated screenshot (numbered boxes
  over every clickable/typeable element) on every step.
- **Reasoning** (`agent/reasoning.py`): sends page state + screenshot to
  Gemini with a fixed action schema (`agent/actions.py`); Gemini picks one
  function call per turn.
- **Memory** (`agent/memory.py`): a scratchpad + recent action history kept
  in every prompt so the agent doesn't lose context across steps, plus loop
  detection on repeated URLs.
- **Observability** (`observability/`): every step is logged to a JSONL
  trace with its screenshot; `replay.py` turns a trace into an animated GIF.
- **Evaluation** (`evaluation/`): a 25-task benchmark across 5 categories,
  scored by an LLM-as-judge, aggregated into `results/summary.txt`.

## Benchmark Results

See `results/summary.txt` after running `python scripts/run_benchmark.py`.

## Example: running a single task

```bash
source .venv/bin/activate
python scripts/run_task.py "Go to Hacker News and summarize today's #1 story" --max-steps 15
```

Watch a Chromium window perform the task live (headless is off by default).
A trace with one screenshot per step is saved under `results/traces/`.

## Technical Highlights

- Vision-augmented perception: screenshot + DOM + accessibility tree on
  every step, not just the DOM.
- Annotated screenshots give the model a stable, numbered way to refer to
  elements (`click("btn_3")`) instead of guessing brittle selectors.
- Scratchpad memory prevents context loss across long multi-step tasks.
- JSONL tracing + GIF replay make any run fully demoable after the fact.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env   # then add your GEMINI_API_KEY
```

## Run the benchmark

```bash
python scripts/run_benchmark.py
```

## Run the UI

```bash
streamlit run ui/app.py
```

## Known Limitations

- Sites behind a login wall or CAPTCHA cause `task_failed`; the agent does
  not attempt to bypass either.
- DuckDuckGo's HTML search endpoint can change markup; `tools/search.py`'s
  CSS selectors may need updating if results suddenly come back empty.
- Benchmark tasks hit live third-party sites — results can vary run-to-run
  as those sites change their layout or content.
