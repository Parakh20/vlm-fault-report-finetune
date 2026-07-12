# Autonomous Web Agent

An autonomous agent that controls a real browser (Chromium, Firefox, or
WebKit) via Playwright and reasons in a ReAct loop: perceive the page,
decide the next action, execute it, repeat until the task is done or the
step cap is hit. A Planner sketches subtasks up front and a Verifier checks
every completion claim against the actual final page before accepting it.

**Status:** Core agent, provider layer, memory/validation/perception
upgrades, multi-agent split, multi-tab/multi-browser support, and
observability are implemented and covered by 138 passing tests. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram and
per-component detail — this file is the quickstart.

## Architecture (short version)

```
Task → Planner (subtask sketch)
     → Executor ReAct loop:
         Perceive (DOM + a11y tree + screenshot + OCR + viewport)
         → Validate the proposed action against the page
         → Reason via Provider Router (Groq → Gemini → OpenRouter → Ollama)
         → Execute via Playwright
     → Verifier checks task_complete claims against final page state
     → Result
```

- **LLM layer** (`agent/llm/`): a `BaseLLM` interface with Groq, Gemini,
  OpenRouter, and Ollama providers behind an `LLMRouter` that retries with
  exponential backoff and falls back to the next configured provider on
  429/503/timeout. One provider going down doesn't stop the agent.
- **Planner / Executor / Verifier** (`agent/planner.py`,
  `agent/reasoning.py`, `agent/verifier.py`): the Executor still runs the
  step-by-step loop, but a Planner sketches non-binding subtasks up front
  and a Verifier checks `task_complete` claims against the real final page
  before accepting them — rejected claims feed back into the loop instead
  of ending the run.
- **Memory** (`agent/memory.py`): goal, scratchpad, extracted facts, and a
  dedicated failures list (separate from general action history) used to
  detect repeated dead ends and loops.
- **Validation** (`agent/validation.py`): checks a proposed action's
  `element_id` and required arguments against the current page *before* it
  reaches the browser — a hallucinated element id becomes a cheap repair
  note instead of a wasted round trip.
- **Perception** (`perception/`): DOM + accessibility tree + annotated
  screenshot + optional OCR pass + viewport/scroll/focused-element signals,
  capped at 100 elements to keep prompts compact.
- **Browser** (`agent/browser.py`): Chromium, Firefox, or WebKit via
  Playwright; multi-tab parallel extraction via `tools/parallel.py`.
- **Observability** (`observability/`): per-step JSONL trace with
  screenshot + DOM diff; Playwright `trace.zip` and session video capture
  wired into the benchmark runner.
- **Evaluation** (`evaluation/`): 35 tasks across 15 categories (original
  5 plus shopping, Wikipedia, tables, downloads, authentication,
  pagination, infinite scroll, CAPTCHA-detection, dynamic content, file
  uploads), scored by an LLM-as-judge, with success rate, efficiency,
  recovery rate, and hallucination rate rolled up into
  `results/summary.txt`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium firefox webkit
cp .env.example .env   # then add at least one provider key (see below)
```

`requirements.txt` includes `pytesseract` for OCR; it also needs the
`tesseract-ocr` system package (`sudo apt-get install tesseract-ocr` on
Debian/Ubuntu). OCR degrades to an empty string if either piece is missing
— the agent still works fine without it.

### Configuring LLM providers

At least one of these must be set in `.env`. They're tried in
`LLM_PROVIDER_ORDER` (default `groq,gemini,openrouter,ollama`), falling
back left-to-right on failure:

| Provider | Env vars | Notes |
|---|---|---|
| Groq | `GROQ_API_KEY`, `GROQ_MODEL` | |
| Gemini | `GEMINI_API_KEY`, `GEMINI_MODEL` | |
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | one key, dozens of models |
| Ollama | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | local, no key needed; requires `ollama serve` running |

`BROWSER_ENGINE` selects `chromium` (default), `firefox`, or `webkit`.

## Example: running a single task

```bash
source .venv/bin/activate
python scripts/run_task.py "Go to Hacker News and summarize today's #1 story" --max-steps 15
```

Watch a browser window perform the task live (headless is off by default).
The CLI prints the final result, step count, token count, duration, and
final URL.

## Run interactively

```bash
python scripts/interactive.py
```

Keeps one browser session open and takes instructions one at a time from
the terminal until you type `exit`.

## Run the benchmark

```bash
python scripts/run_benchmark.py
```

Runs all 35 tasks, writes per-task JSON + a `trace.zip` + session video
under `results/traces/<task_id>/`, and rolls everything up into
`results/summary.txt` and `results/benchmark_summary.csv`.

## Run the UI

```bash
streamlit run ui/app.py
```

## Known Limitations

- CAPTCHAs are detected and reported via `task_failed`, never bypassed —
  see the `captcha_detection` benchmark category, which explicitly grades
  on graceful failure rather than solving the challenge.
- DuckDuckGo's HTML search endpoint can change markup; `tools/search.py`'s
  CSS selectors may need updating if results suddenly come back empty.
- Benchmark tasks hit live/public third-party sites — results can vary
  run-to-run as those sites change their layout or content.
- `perception/screenshot.py` currently returns a mocked blank image due to
  a prior "headless VM crash" workaround unrelated to this session's work;
  see `ARCHITECTURE.md`'s "Known gaps" section.
