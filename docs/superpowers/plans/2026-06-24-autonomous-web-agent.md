# Autonomous Web Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality autonomous web agent that controls a real Chromium browser via Playwright, uses Gemini as the reasoning engine in a ReAct loop, and is evaluated against a 25-task benchmark with observability (tracing/replay) and a Streamlit UI.

**Architecture:** `Task → ReAct Loop → [Perceive page (DOM + a11y tree + screenshot) | Reason with Gemini (function calling) | Execute action via Playwright] → repeat → task_complete/task_failed`. Perception, action execution, reasoning, and memory are separate modules wired together by `agent/reasoning.py`. Evaluation and observability are built on top of the core loop without modifying it.

**Tech Stack:** Python 3.12, `playwright` (async API) for browser control, `google-genai` SDK with `gemini-2.5-flash` for reasoning + vision + function calling, `python-dotenv` for secrets, `pytest` + `pytest-asyncio` for tests, `streamlit` for the UI, `Pillow` for screenshot annotation, `imageio` for replay GIFs.

## Global Constraints

- Python 3.11+ (repo has 3.12.3), async throughout — every browser-touching function is `async def`.
- Reasoning engine is Google Gemini (`gemini-2.5-flash`), **not** Anthropic Claude — this project deliberately deviates from the original Claude-based spec because the user only has a Gemini API key. Every place the original spec said "Claude"/"claude-sonnet-4-6" maps to Gemini in this plan.
- All secrets in `.env` (already created, contains `GEMINI_API_KEY=...`), loaded via `python-dotenv`. Never hardcode the key.
- Max 25 steps per task — hard guardrail enforced in `agent/reasoning.py`, not configurable past that ceiling.
- Never fill forms with real personal data — any form-interaction code/tests use obviously-fake placeholder data (e.g. `"Test User"`, `"test@example.com"`).
- `requirements.txt` must pin exact versions installed during Task 1.
- Every module under `agent/`, `tools/`, `perception/`, `evaluation/`, `observability/` must be runnable standalone via `if __name__ == "__main__":` guard for a quick manual smoke test.
- `headless=False` is the default for `BrowserSession.start()`; benchmark runner explicitly passes `headless=True`.
- Tests must not depend on live external websites (flaky, rate-limited, ToS risk). Perception/action/integration tests run against a local static HTML fixture served by a local HTTP server started in `tests/conftest.py`. Only the benchmark suite itself (Task 21+) touches real sites, and that's an evaluation run, not a unit/integration test.
- AAA test structure, descriptive test names, 80%+ coverage target on `agent/`, `perception/`, `tools/`, `evaluation/` (UI and scripts are thin glue, excluded from the coverage target).
- Minimum test coverage 80%, TDD red→green→refactor for every task below.

---

## Shared Interfaces Reference

These types are defined once (Task 2) and consumed by nearly every later task. Listed here so every task's author knows the exact shape without re-deriving it.

```python
# agent/types.py
from dataclasses import dataclass, field

@dataclass
class Element:
    id: str                 # stable short id e.g. "btn_3", "link_12", "input_0"
    tag: str                # "button", "a", "input", "select", "textarea", ...
    text: str               # visible text / label
    role: str                # accessibility role, "" if unknown
    placeholder: str | None
    href: str | None
    visible: bool
    bbox: dict               # {"x": float, "y": float, "width": float, "height": float}

@dataclass
class PageState:
    url: str
    title: str
    screenshot_b64: str
    interactive_elements: list[Element]
    accessibility_tree: str
    scroll_y: int
    page_height: int
    dialog_visible: bool
    dialog_text: str | None

@dataclass
class ActionResult:
    success: bool
    new_url: str
    error: str | None
    screenshot_b64: str

@dataclass
class Step:
    number: int
    url: str
    action_type: str
    action_input: dict
    action_result: ActionResult
    reasoning_text: str
    tokens_used: int

@dataclass
class AgentRun:
    task: str
    success: bool
    result: str
    steps: list[Step] = field(default_factory=list)
    total_actions: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    final_url: str = ""
```

---

### Task 1: Project scaffolding, dependencies, config loader

**Files:**
- Create: `requirements.txt`
- Create: `agent/__init__.py`, `tools/__init__.py`, `perception/__init__.py`, `evaluation/__init__.py`, `observability/__init__.py`, `ui/__init__.py`, `scripts/__init__.py`
- Create: `agent/config.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/fixtures/sample_page.html`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `agent.config.Settings` dataclass with fields `gemini_api_key: str`, `gemini_model: str = "gemini-2.5-flash"`, `max_steps: int = 25`, `headless_default: bool = False`; `agent.config.load_settings() -> Settings` (raises `RuntimeError` if `GEMINI_API_KEY` missing).
- Produces: `tests/conftest.py` fixture `static_server` (yields base URL of a local `http.server` serving `tests/fixtures/`) used by every later Playwright-based test.

- [ ] **Step 1: Write `requirements.txt`**

```text
playwright==1.59.0
google-genai==1.50.1
python-dotenv==1.1.0
pytest==8.3.4
pytest-asyncio==0.25.2
pillow==11.1.0
imageio==2.36.1
streamlit==1.41.1
pandas==2.2.3
```

- [ ] **Step 2: Install dependencies and Chromium**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```
Expected: all packages install cleanly; `playwright install chromium` reports already-downloaded or installs successfully.

- [ ] **Step 3: Create package `__init__.py` files (empty) for `agent`, `tools`, `perception`, `evaluation`, `observability`, `ui`, `scripts`, `tests`**

```bash
for d in agent tools perception evaluation observability ui scripts tests; do
  mkdir -p "$d"
  touch "$d/__init__.py"
done
```

- [ ] **Step 4: Write the failing test for config loading**

```python
# tests/test_config.py
import pytest
from agent.config import load_settings

def test_load_settings_reads_gemini_api_key_from_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Act
    settings = load_settings()

    # Assert
    assert settings.gemini_api_key == "fake-key-123"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.max_steps == 25
    assert settings.headless_default is False


def test_load_settings_raises_when_key_missing(monkeypatch):
    # Arrange
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_settings()
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.config'`

- [ ] **Step 6: Implement `agent/config.py`**

```python
# agent/config.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    max_steps: int = 25
    headless_default: bool = False


def load_settings() -> Settings:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return Settings(gemini_api_key=api_key, gemini_model=model)


if __name__ == "__main__":
    print(load_settings())
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 8: Create the static test fixture page**

```html
<!-- tests/fixtures/sample_page.html -->
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Fixture Page</title></head>
<body>
  <h1>Test Fixture</h1>
  <p id="intro">This is sample text used by perception tests.</p>
  <a id="about-link" href="/about.html">About</a>
  <button id="search-btn">Search</button>
  <input id="query-input" type="text" placeholder="Search query" />
  <select id="sort-select">
    <option value="relevance">Relevance</option>
    <option value="date">Date</option>
  </select>
  <table id="data-table">
    <tr><th>Name</th><th>Price</th></tr>
    <tr><td>Widget</td><td>$10</td></tr>
    <tr><td>Gadget</td><td>$20</td></tr>
  </table>
  <div style="display:none;" id="hidden-div">Hidden text</div>
</body>
</html>
```

```html
<!-- tests/fixtures/about.html -->
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>About</title></head>
<body><h1>About Page</h1><p>You navigated successfully.</p></body></html>
```

- [ ] **Step 9: Write `tests/conftest.py` with the static server and Playwright fixtures**

```python
# tests/conftest.py
import asyncio
import functools
import http.server
import threading

import pytest
import pytest_asyncio
from playwright.async_api import async_playwright

FIXTURES_DIR = "tests/fixtures"


@pytest.fixture(scope="session")
def static_server():
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=FIXTURES_DIR
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest_asyncio.fixture
async def browser_page():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        yield page
        await browser.close()
```

- [ ] **Step 10: Run the full test suite to confirm fixtures load**

Run: `pytest tests/ -v`
Expected: 2 passed (config tests only; no other tests exist yet)

- [ ] **Step 11: Commit**

```bash
git add requirements.txt agent/__init__.py agent/config.py tools/__init__.py \
  perception/__init__.py evaluation/__init__.py observability/__init__.py \
  ui/__init__.py scripts/__init__.py tests/__init__.py tests/conftest.py \
  tests/test_config.py tests/fixtures/
git commit -m "feat: project scaffolding, config loader, and test fixtures"
```

---

### Task 2: Shared dataclasses (`agent/types.py`)

**Files:**
- Create: `agent/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Produces: `Element`, `PageState`, `ActionResult`, `Step`, `AgentRun` exactly as defined in "Shared Interfaces Reference" above.

- [ ] **Step 1: Write failing test**

```python
# tests/test_types.py
from agent.types import ActionResult, Element, PageState


def test_element_holds_stable_id_and_bbox():
    # Arrange / Act
    el = Element(
        id="btn_0", tag="button", text="Search", role="button",
        placeholder=None, href=None, visible=True,
        bbox={"x": 1.0, "y": 2.0, "width": 50.0, "height": 20.0},
    )

    # Assert
    assert el.id == "btn_0"
    assert el.bbox["width"] == 50.0


def test_page_state_holds_elements_and_metadata():
    # Arrange
    el = Element("link_0", "a", "About", "link", None, "/about", True, {})

    # Act
    state = PageState(
        url="http://x/", title="X", screenshot_b64="", interactive_elements=[el],
        accessibility_tree="", scroll_y=0, page_height=1000,
        dialog_visible=False, dialog_text=None,
    )

    # Assert
    assert state.interactive_elements[0].href == "/about"


def test_action_result_carries_error_on_failure():
    # Arrange / Act
    result = ActionResult(success=False, new_url="http://x/", error="not found", screenshot_b64="")

    # Assert
    assert result.success is False
    assert result.error == "not found"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.types'`

- [ ] **Step 3: Implement `agent/types.py`** (copy verbatim from "Shared Interfaces Reference" section above — all five dataclasses: `Element`, `PageState`, `ActionResult`, `Step`, `AgentRun`)

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_types.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add agent/types.py tests/test_types.py
git commit -m "feat: shared dataclasses for page state, actions, and agent runs"
```

---

### Task 3: DOM parser — interactive element extraction (`perception/dom_parser.py`)

**Files:**
- Create: `perception/dom_parser.py`
- Test: `tests/test_dom_parser.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page` (a live page), `agent.types.Element`
- Produces: `async def extract_interactive_elements(page) -> list[Element]` — visible-only, max 50, stable ids like `btn_0`, `link_0`, `input_0`, `select_0`, `textarea_0` (counter per tag-category).

- [ ] **Step 1: Write failing test**

```python
# tests/test_dom_parser.py
import pytest
from perception.dom_parser import extract_interactive_elements


@pytest.mark.asyncio
async def test_extracts_visible_interactive_elements(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    elements = await extract_interactive_elements(browser_page)
    ids = [e.id for e in elements]
    by_id = {e.id: e for e in elements}

    # Assert
    assert "link_0" in ids
    assert "btn_0" in ids
    assert "input_0" in ids
    assert "select_0" in ids
    assert by_id["link_0"].href == "/about.html"
    assert by_id["input_0"].placeholder == "Search query"


@pytest.mark.asyncio
async def test_excludes_hidden_elements(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    elements = await extract_interactive_elements(browser_page)
    texts = [e.text for e in elements]

    # Assert
    assert "Hidden text" not in texts


@pytest.mark.asyncio
async def test_truncates_to_max_50_elements(browser_page, static_server, tmp_path):
    # Arrange
    many_buttons = "".join(f'<button id="b{i}">Btn {i}</button>' for i in range(80))
    html_path = tmp_path / "many.html"
    html_path.write_text(f"<html><body>{many_buttons}</body></html>")
    await browser_page.goto(html_path.as_uri())

    # Act
    elements = await extract_interactive_elements(browser_page)

    # Assert
    assert len(elements) == 50
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_dom_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.dom_parser'`

- [ ] **Step 3: Implement `perception/dom_parser.py`**

```python
# perception/dom_parser.py
from playwright.async_api import Page

from agent.types import Element

INTERACTIVE_SELECTOR = (
    "a, button, input, select, textarea, "
    "[role=button], [role=link], [onclick]"
)
MAX_ELEMENTS = 50

_TAG_PREFIX = {
    "a": "link",
    "button": "btn",
    "input": "input",
    "select": "select",
    "textarea": "textarea",
}


async def extract_interactive_elements(page: Page) -> list[Element]:
    handles = await page.query_selector_all(INTERACTIVE_SELECTOR)
    counters: dict[str, int] = {}
    elements: list[Element] = []

    for handle in handles:
        if len(elements) >= MAX_ELEMENTS:
            break
        if not await handle.is_visible():
            continue

        tag = (await handle.evaluate("el => el.tagName.toLowerCase()")) or ""
        prefix = _TAG_PREFIX.get(tag, "el")
        idx = counters.get(prefix, 0)
        counters[prefix] = idx + 1

        bbox = await handle.bounding_box() or {}
        text = (await handle.inner_text()).strip() if tag != "input" else ""
        role = (await handle.get_attribute("role")) or ""
        placeholder = await handle.get_attribute("placeholder")
        href = await handle.get_attribute("href")

        elements.append(
            Element(
                id=f"{prefix}_{idx}",
                tag=tag,
                text=text,
                role=role,
                placeholder=placeholder,
                href=href,
                visible=True,
                bbox=bbox,
            )
        )

    return elements


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            els = await extract_interactive_elements(page)
            for e in els:
                print(e)
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_dom_parser.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add perception/dom_parser.py tests/test_dom_parser.py
git commit -m "feat: extract visible interactive elements from the DOM"
```

---

### Task 4: Accessibility tree extraction (`perception/accessibility.py`)

**Files:**
- Create: `perception/accessibility.py`
- Test: `tests/test_accessibility.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`
- Produces: `async def extract_accessibility_tree(page) -> str` — indented text tree, format `[role] "name" extra="value"` per line, invisible/empty nodes dropped.

- [ ] **Step 1: Write failing test**

```python
# tests/test_accessibility.py
import pytest
from perception.accessibility import extract_accessibility_tree


@pytest.mark.asyncio
async def test_returns_indented_text_tree_with_roles_and_names(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)

    # Assert
    assert "button" in tree
    assert "Search" in tree
    assert "link" in tree


@pytest.mark.asyncio
async def test_drops_nodes_with_no_name_and_no_children(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    tree = await extract_accessibility_tree(browser_page)

    # Assert: no line is just an empty bracket pair with nothing else
    for line in tree.splitlines():
        assert line.strip() != "[]"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_accessibility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.accessibility'`

- [ ] **Step 3: Implement `perception/accessibility.py`**

```python
# perception/accessibility.py
from playwright.async_api import Page


def _format_node(node: dict, depth: int) -> list[str]:
    name = (node.get("name") or "").strip()
    role = node.get("role") or ""
    children = node.get("children") or []

    lines: list[str] = []
    if role not in ("", "generic", "none") and (name or children):
        indent = "  " * depth
        label = f'{indent}[{role}]'
        if name:
            label += f' "{name}"'
        lines.append(label)
        depth += 1

    for child in children:
        lines.extend(_format_node(child, depth))

    return lines


async def extract_accessibility_tree(page: Page) -> str:
    snapshot = await page.accessibility.snapshot(interesting_only=True)
    if not snapshot:
        return ""
    lines = _format_node(snapshot, 0)
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await extract_accessibility_tree(page))
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_accessibility.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add perception/accessibility.py tests/test_accessibility.py
git commit -m "feat: simplified accessibility tree extraction"
```

---

### Task 5: Annotated screenshot capture (`perception/screenshot.py`)

**Files:**
- Create: `perception/screenshot.py`
- Test: `tests/test_screenshot.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`, `list[Element]` (from Task 3)
- Produces: `async def capture_annotated_screenshot(page, elements: list[Element]) -> str` (base64 PNG with numbered boxes drawn at each element's bbox); `def save_screenshot(b64: str, path: str) -> None`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_screenshot.py
import base64
import io

import pytest
from PIL import Image

from perception.dom_parser import extract_interactive_elements
from perception.screenshot import capture_annotated_screenshot, save_screenshot


@pytest.mark.asyncio
async def test_capture_annotated_screenshot_returns_valid_base64_png(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")
    elements = await extract_interactive_elements(browser_page)

    # Act
    b64 = await capture_annotated_screenshot(browser_page, elements)

    # Assert
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    assert img.width > 0 and img.height > 0


@pytest.mark.asyncio
async def test_save_screenshot_writes_png_file(browser_page, static_server, tmp_path):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")
    elements = await extract_interactive_elements(browser_page)
    b64 = await capture_annotated_screenshot(browser_page, elements)
    out_path = tmp_path / "step_0.png"

    # Act
    save_screenshot(b64, str(out_path))

    # Assert
    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_screenshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.screenshot'`

- [ ] **Step 3: Implement `perception/screenshot.py`**

```python
# perception/screenshot.py
import base64
import io

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

from agent.types import Element


async def capture_annotated_screenshot(page: Page, elements: list[Element]) -> str:
    raw_png = await page.screenshot(full_page=False)
    image = Image.open(io.BytesIO(raw_png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    colors = ["red", "blue", "green", "orange", "purple"]
    for i, el in enumerate(elements):
        bbox = el.bbox
        if not bbox:
            continue
        x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
        if w <= 0 or h <= 0:
            continue
        color = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x, max(0, y - 10)), el.id, fill=color, font=font)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_screenshot(b64: str, path: str) -> None:
    raw = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(raw)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    from perception.dom_parser import extract_interactive_elements

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            els = await extract_interactive_elements(page)
            b64 = await capture_annotated_screenshot(page, els)
            save_screenshot(b64, "/tmp/demo_screenshot.png")
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_screenshot.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add perception/screenshot.py tests/test_screenshot.py
git commit -m "feat: annotated screenshot capture with numbered element overlays"
```

---

### Task 6: Browser wrapper — `BrowserSession` (`agent/browser.py`)

**Files:**
- Create: `agent/browser.py`
- Test: `tests/test_browser.py`

**Interfaces:**
- Consumes: `perception.dom_parser.extract_interactive_elements`, `perception.accessibility.extract_accessibility_tree`, `perception.screenshot.capture_annotated_screenshot`, `agent.types.PageState`
- Produces: `class BrowserSession` with `async start(headless: bool = False) -> None`, `async stop() -> None`, `async get_page_state() -> PageState`, `.page` (raw Playwright `Page`, used by `agent/actions.py` dispatch in Task 12).

- [ ] **Step 1: Write failing test**

```python
# tests/test_browser.py
import pytest
from agent.browser import BrowserSession


@pytest.mark.asyncio
async def test_start_and_get_page_state_returns_url_and_elements(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        state = await session.get_page_state()

        # Assert
        assert state.url.endswith("sample_page.html")
        assert state.title == "Fixture Page"
        assert any(e.id == "btn_0" for e in state.interactive_elements)
        assert state.screenshot_b64 != ""
        assert state.dialog_visible is False
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_stop_closes_browser_cleanly(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    # Act / Assert (no exception means clean shutdown)
    await session.stop()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_browser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.browser'`

- [ ] **Step 3: Implement `agent/browser.py`**

```python
# agent/browser.py
from playwright.async_api import Browser, Page, Playwright, async_playwright

from agent.types import PageState
from perception.accessibility import extract_accessibility_tree
from perception.dom_parser import extract_interactive_elements
from perception.screenshot import capture_annotated_screenshot

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class BrowserSession:
    def __init__(self) -> None:
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self.page: Page | None = None
        self._dialog_text: str | None = None

    async def start(self, headless: bool = False) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=headless)
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="Asia/Kolkata",
        )
        self.page = await context.new_page()
        self.page.on("dialog", self._on_dialog)

    def _on_dialog(self, dialog) -> None:
        self._dialog_text = dialog.message

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def get_page_state(self) -> PageState:
        assert self.page is not None, "call start() first"
        elements = await extract_interactive_elements(self.page)
        tree = await extract_accessibility_tree(self.page)
        screenshot_b64 = await capture_annotated_screenshot(self.page, elements)
        scroll_y = await self.page.evaluate("window.scrollY")
        page_height = await self.page.evaluate("document.body.scrollHeight")

        return PageState(
            url=self.page.url,
            title=await self.page.title(),
            screenshot_b64=screenshot_b64,
            interactive_elements=elements,
            accessibility_tree=tree,
            scroll_y=int(scroll_y),
            page_height=int(page_height),
            dialog_visible=self._dialog_text is not None,
            dialog_text=self._dialog_text,
        )


if __name__ == "__main__":
    import asyncio

    async def _demo():
        session = BrowserSession()
        await session.start(headless=False)
        await session.page.goto("https://example.com")
        state = await session.get_page_state()
        print(state.url, state.title, len(state.interactive_elements))
        await session.stop()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_browser.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add agent/browser.py tests/test_browser.py
git commit -m "feat: BrowserSession wrapper with full page-state perception"
```

---

### Task 7: Navigation tools (`tools/navigate.py`)

**Files:**
- Create: `tools/navigate.py`
- Test: `tests/test_navigate.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`
- Produces: `async def go_to_url(page, url: str) -> None`, `async def go_back(page) -> None`, `async def go_forward(page) -> None` — every function waits for `"load"` state before returning.

- [ ] **Step 1: Write failing test**

```python
# tests/test_navigate.py
import pytest
from tools.navigate import go_back, go_forward, go_to_url


@pytest.mark.asyncio
async def test_go_to_url_navigates_and_waits_for_load(browser_page, static_server):
    # Act
    await go_to_url(browser_page, f"{static_server}/sample_page.html")

    # Assert
    assert browser_page.url.endswith("sample_page.html")
    assert await browser_page.title() == "Fixture Page"


@pytest.mark.asyncio
async def test_go_back_and_forward_traverse_history(browser_page, static_server):
    # Arrange
    await go_to_url(browser_page, f"{static_server}/sample_page.html")
    await browser_page.click("#about-link")
    assert browser_page.url.endswith("about.html")

    # Act
    await go_back(browser_page)
    # Assert
    assert browser_page.url.endswith("sample_page.html")

    # Act
    await go_forward(browser_page)
    # Assert
    assert browser_page.url.endswith("about.html")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_navigate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.navigate'`

- [ ] **Step 3: Implement `tools/navigate.py`**

```python
# tools/navigate.py
from playwright.async_api import Page


async def go_to_url(page: Page, url: str) -> None:
    await page.goto(url, wait_until="load")


async def go_back(page: Page) -> None:
    await page.go_back(wait_until="load")


async def go_forward(page: Page) -> None:
    await page.go_forward(wait_until="load")


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await go_to_url(page, "https://example.com")
            print(page.url)
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_navigate.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tools/navigate.py tests/test_navigate.py
git commit -m "feat: navigation tools (go_to_url, go_back, go_forward)"
```

---

### Task 8: Interaction tools (`tools/interact.py`)

**Files:**
- Create: `tools/interact.py`
- Test: `tests/test_interact.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`, element ids produced by `perception.dom_parser` (CSS lookup uses Playwright's own element handles cached per-page via `id` attribute injected by `tools.interact._locator_for`)
- Produces: `async def click(page, element_id: str) -> None`, `async def type_text(page, element_id: str, text: str, clear_first: bool = True) -> None`, `async def scroll(page, direction: str, amount: int = 300) -> None`, `async def select_option(page, element_id: str, value: str) -> None`, `async def press_key(page, key: str) -> None`, `async def hover(page, element_id: str) -> None`. All raise `ValueError` with a clear message if `element_id` is not found.

This task introduces the **element-id-to-locator bridge**: `dom_parser.extract_interactive_elements` assigns ids like `btn_0` by enumerating matched handles in DOM order. `tools/interact.py` re-derives the same locator by calling `extract_interactive_elements` again and indexing into the *same* `INTERACTIVE_SELECTOR` query — this works because element order is stable between calls on an unchanged page (re-querying immediately before acting, not caching stale handles across navigations).

- [ ] **Step 1: Write failing test**

```python
# tests/test_interact.py
import pytest
from perception.dom_parser import extract_interactive_elements
from tools.interact import click, hover, press_key, scroll, select_option, type_text


@pytest.mark.asyncio
async def test_click_navigates_via_link_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    await click(browser_page, "link_0")

    # Assert
    assert browser_page.url.endswith("about.html")


@pytest.mark.asyncio
async def test_type_text_fills_input_and_clears_first(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")
    await browser_page.fill("#query-input", "old value")

    # Act
    await type_text(browser_page, "input_0", "new query")

    # Assert
    assert await browser_page.input_value("#query-input") == "new query"


@pytest.mark.asyncio
async def test_select_option_sets_dropdown_value(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    await select_option(browser_page, "select_0", "date")

    # Assert
    assert await browser_page.eval_on_selector("#sort-select", "el => el.value") == "date"


@pytest.mark.asyncio
async def test_scroll_down_changes_scroll_position(browser_page, static_server, tmp_path):
    # Arrange: make the page tall enough to scroll
    tall_html = "<html><body style='height:3000px'>" + "<p>x</p>" * 50 + "</body></html>"
    html_path = tmp_path / "tall.html"
    html_path.write_text(tall_html)
    await browser_page.goto(html_path.as_uri())

    # Act
    await scroll(browser_page, "down", amount=500)

    # Assert
    scroll_y = await browser_page.evaluate("window.scrollY")
    assert scroll_y > 0


@pytest.mark.asyncio
async def test_click_raises_value_error_for_unknown_element_id(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act / Assert
    with pytest.raises(ValueError, match="btn_99"):
        await click(browser_page, "btn_99")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_interact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.interact'`

- [ ] **Step 3: Implement `tools/interact.py`**

```python
# tools/interact.py
from playwright.async_api import ElementHandle, Page

from perception.dom_parser import INTERACTIVE_SELECTOR, extract_interactive_elements


async def _handle_for(page: Page, element_id: str) -> ElementHandle:
    elements = await extract_interactive_elements(page)
    index = next((i for i, e in enumerate(elements) if e.id == element_id), None)
    if index is None:
        raise ValueError(f"Element id '{element_id}' not found on current page")

    handles = await page.query_selector_all(INTERACTIVE_SELECTOR)
    visible_handles = []
    for h in handles:
        if await h.is_visible():
            visible_handles.append(h)
    return visible_handles[index]


async def click(page: Page, element_id: str) -> None:
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.click()
    await page.wait_for_load_state("domcontentloaded", timeout=5000)


async def type_text(page: Page, element_id: str, text: str, clear_first: bool = True) -> None:
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.click()
    if clear_first:
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
    await page.keyboard.type(text)


async def scroll(page: Page, direction: str, amount: int = 300) -> None:
    deltas = {"up": (0, -amount), "down": (0, amount)}
    if direction in deltas:
        dx, dy = deltas[direction]
        await page.mouse.wheel(dx, dy)
    elif direction == "top":
        await page.evaluate("window.scrollTo(0, 0)")
    elif direction == "bottom":
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    else:
        raise ValueError(f"Unknown scroll direction: {direction}")


async def select_option(page: Page, element_id: str, value: str) -> None:
    handle = await _handle_for(page, element_id)
    await handle.select_option(value=value)


async def press_key(page: Page, key: str) -> None:
    await page.keyboard.press(key)


async def hover(page: Page, element_id: str) -> None:
    handle = await _handle_for(page, element_id)
    await handle.scroll_into_view_if_needed()
    await handle.hover()


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print("demo ready")
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_interact.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tools/interact.py tests/test_interact.py
git commit -m "feat: interaction tools (click, type, scroll, select, hover, press_key)"
```

---

### Task 9: Extraction tools (`tools/extract.py`)

**Files:**
- Create: `tools/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`, element-id bridge from Task 8 (`tools.interact._handle_for`)
- Produces: `async def get_page_text(page) -> str`, `async def extract_table(page, element_id: str) -> str` (markdown table).

- [ ] **Step 1: Write failing test**

```python
# tests/test_extract.py
import pytest
from tools.extract import extract_table, get_page_text


@pytest.mark.asyncio
async def test_get_page_text_returns_visible_text(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    text = await get_page_text(browser_page)

    # Assert
    assert "This is sample text used by perception tests." in text
    assert "Hidden text" not in text


@pytest.mark.asyncio
async def test_extract_table_returns_markdown(browser_page, static_server):
    # Arrange
    await browser_page.goto(f"{static_server}/sample_page.html")

    # Act
    markdown = await extract_table(browser_page, "#data-table")

    # Assert
    assert "| Name | Price |" in markdown
    assert "| Widget | $10 |" in markdown
    assert "| Gadget | $20 |" in markdown
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.extract'`

- [ ] **Step 3: Implement `tools/extract.py`**

```python
# tools/extract.py
from playwright.async_api import Page


async def get_page_text(page: Page) -> str:
    return await page.evaluate("document.body.innerText")


async def extract_table(page: Page, element_id: str) -> str:
    rows = await page.eval_on_selector_all(
        f"{element_id} tr",
        "rows => rows.map(r => Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.trim()))",
    )
    if not rows:
        return ""

    header, *body = rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await get_page_text(page))
            await browser.close()

    asyncio.run(_demo())
```

Note: `extract_table` takes a CSS selector string directly (`"#data-table"`), not an agent element id — table elements aren't part of `INTERACTIVE_SELECTOR`, so the `tools.interact` bridge doesn't apply. `agent/actions.py` (Task 12) passes through whatever selector-like string Gemini supplies for this one tool, documented in its schema description.

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_extract.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tools/extract.py tests/test_extract.py
git commit -m "feat: page text and markdown table extraction tools"
```

---

### Task 10: Wait tool (`tools/wait.py`)

**Files:**
- Create: `tools/wait.py`
- Test: `tests/test_wait.py`

**Interfaces:**
- Produces: `async def wait(seconds: float) -> None` (clamps to max 5.0s), `async def wait_for_load(page) -> None`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_wait.py
import time

import pytest
from tools.wait import wait, wait_for_load


@pytest.mark.asyncio
async def test_wait_sleeps_for_requested_duration():
    # Arrange
    start = time.monotonic()

    # Act
    await wait(0.2)

    # Assert
    assert time.monotonic() - start >= 0.2


@pytest.mark.asyncio
async def test_wait_clamps_to_max_5_seconds():
    # Arrange
    start = time.monotonic()

    # Act
    await wait(100)

    # Assert
    elapsed = time.monotonic() - start
    assert elapsed <= 5.5


@pytest.mark.asyncio
async def test_wait_for_load_resolves_after_navigation(browser_page, static_server):
    # Act
    await browser_page.goto(f"{static_server}/sample_page.html", wait_until="commit")
    await wait_for_load(browser_page)

    # Assert
    assert await browser_page.title() == "Fixture Page"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_wait.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.wait'`

- [ ] **Step 3: Implement `tools/wait.py`**

```python
# tools/wait.py
import asyncio

from playwright.async_api import Page

MAX_WAIT_SECONDS = 5.0


async def wait(seconds: float) -> None:
    await asyncio.sleep(min(seconds, MAX_WAIT_SECONDS))


async def wait_for_load(page: Page) -> None:
    await page.wait_for_load_state("load")


if __name__ == "__main__":
    asyncio.run(wait(1.0))
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_wait.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/wait.py tests/test_wait.py
git commit -m "feat: wait tool with 5-second safety clamp"
```

---

### Task 11: Web search tool (`tools/search.py`)

**Files:**
- Create: `tools/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `playwright.async_api.Page`
- Produces: `async def search_web(page, query: str) -> str` — navigates to DuckDuckGo HTML search, returns top 5 results formatted as `"1. {title}\n   {url}\n   {snippet}\n"`.

This is the one tool that touches a real external site; its test is marked so it can be skipped offline/CI without skipping the rest of the suite.

- [ ] **Step 1: Write failing test**

```python
# tests/test_search.py
import os

import pytest
from tools.search import search_web

requires_network = pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS") == "1",
    reason="network access disabled",
)


@requires_network
@pytest.mark.asyncio
async def test_search_web_returns_top_5_formatted_results(browser_page):
    # Act
    results = await search_web(browser_page, "Playwright Python documentation")

    # Assert
    assert results.count("\n1.") <= 1  # exactly one numbered "1." entry
    lines = [l for l in results.splitlines() if l.strip().startswith(tuple("12345"))]
    assert 1 <= len(lines) <= 5
    assert "http" in results
```

- [ ] **Step 2: Run to confirm failure**

Run: `SKIP_NETWORK_TESTS=0 pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.search'`

- [ ] **Step 3: Implement `tools/search.py`**

```python
# tools/search.py
from playwright.async_api import Page

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"


async def search_web(page: Page, query: str) -> str:
    from urllib.parse import quote

    url = DUCKDUCKGO_HTML_URL.format(query=quote(query))
    await page.goto(url, wait_until="load")

    results = await page.eval_on_selector_all(
        ".result",
        """nodes => nodes.slice(0, 5).map(n => {
            const title = n.querySelector('.result__title a');
            const snippet = n.querySelector('.result__snippet');
            return {
                title: title ? title.innerText.trim() : '',
                url: title ? title.href : '',
                snippet: snippet ? snippet.innerText.trim() : '',
            };
        })""",
    )

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")
    return "\n".join(lines) if lines else "No results found."


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            print(await search_web(page, "Playwright Python"))
            await browser.close()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_search.py -v`
Expected: 1 passed (requires network)

- [ ] **Step 5: Commit**

```bash
git add tools/search.py tests/test_search.py
git commit -m "feat: DuckDuckGo web search tool"
```

---

### Task 12: Action dispatch and Gemini tool schemas (`agent/actions.py`)

**Files:**
- Create: `agent/actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: every `tools/*` function from Tasks 7-11, `agent.types.ActionResult`, `agent.browser.BrowserSession`
- Produces: `ACTION_SCHEMAS: list[dict]` (Gemini `function_declarations` format, one entry per action including `task_complete`/`task_failed`), `async def dispatch_action(session: BrowserSession, name: str, args: dict) -> ActionResult` — catches all exceptions from the underlying tool and returns `ActionResult(success=False, error=str(e), ...)` instead of raising, screenshots after every action.

- [ ] **Step 1: Write failing test**

```python
# tests/test_actions.py
import pytest
from agent.actions import ACTION_SCHEMAS, dispatch_action
from agent.browser import BrowserSession


def test_action_schemas_include_all_required_actions():
    # Arrange
    names = {schema["name"] for schema in ACTION_SCHEMAS}

    # Assert
    expected = {
        "navigate_to", "click", "type_text", "scroll", "select_option",
        "press_key", "hover", "wait", "get_page_text", "search_web",
        "extract_table", "go_back", "task_complete", "task_failed",
    }
    assert expected.issubset(names)


@pytest.mark.asyncio
async def test_dispatch_navigate_to_succeeds_and_returns_screenshot(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        # Act
        result = await dispatch_action(
            session, "navigate_to", {"url": f"{static_server}/sample_page.html"}
        )

        # Assert
        assert result.success is True
        assert result.new_url.endswith("sample_page.html")
        assert result.screenshot_b64 != ""
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_dispatch_click_unknown_element_returns_failure_result(static_server):
    # Arrange
    session = BrowserSession()
    await session.start(headless=True)

    try:
        await session.page.goto(f"{static_server}/sample_page.html")

        # Act
        result = await dispatch_action(session, "click", {"element_id": "btn_99"})

        # Assert
        assert result.success is False
        assert "btn_99" in result.error
    finally:
        await session.stop()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_actions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.actions'`

- [ ] **Step 3: Implement `agent/actions.py`**

```python
# agent/actions.py
from agent.browser import BrowserSession
from agent.types import ActionResult
from perception.dom_parser import extract_interactive_elements
from perception.screenshot import capture_annotated_screenshot
from tools.extract import extract_table, get_page_text
from tools.interact import click, hover, press_key, scroll, select_option, type_text
from tools.navigate import go_back, go_to_url
from tools.search import search_web
from tools.wait import wait

ACTION_SCHEMAS: list[dict] = [
    {
        "name": "navigate_to",
        "description": "Navigate the browser directly to a URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "click",
        "description": "Click an interactive element by its id (e.g. 'btn_3', 'link_0').",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "string"}},
            "required": ["element_id"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input or textarea element by id, clearing it first.",
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "text": {"type": "string"},
                "clear_first": {"type": "boolean"},
            },
            "required": ["element_id", "text"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up, down, to top, or to bottom.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"]},
                "amount": {"type": "integer"},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option by value in a <select> element by id.",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "string"}, "value": {"type": "string"}},
            "required": ["element_id", "value"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key, e.g. 'Enter', 'Tab', 'Escape'.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "hover",
        "description": "Hover over an interactive element by id to reveal menus/tooltips.",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "string"}},
            "required": ["element_id"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for up to 5 seconds for a slow page to settle.",
        "parameters": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "get_page_text",
        "description": "Return all visible text on the current page.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_web",
        "description": "Search DuckDuckGo and return the top 5 results (title, url, snippet).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "extract_table",
        "description": "Extract an HTML table as markdown using a CSS selector, e.g. '#data-table'.",
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in history.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "task_complete",
        "description": "Signal the task is fully complete and report the final result.",
        "parameters": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    },
    {
        "name": "task_failed",
        "description": "Signal the task cannot be completed and explain why.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

_TEXT_RETURNING = {"get_page_text", "search_web", "extract_table"}


async def dispatch_action(session: BrowserSession, name: str, args: dict) -> ActionResult:
    page = session.page
    assert page is not None, "BrowserSession not started"

    try:
        text_payload = ""
        if name == "navigate_to":
            await go_to_url(page, args["url"])
        elif name == "click":
            await click(page, args["element_id"])
        elif name == "type_text":
            await type_text(page, args["element_id"], args["text"], args.get("clear_first", True))
        elif name == "scroll":
            await scroll(page, args["direction"], args.get("amount", 300))
        elif name == "select_option":
            await select_option(page, args["element_id"], args["value"])
        elif name == "press_key":
            await press_key(page, args["key"])
        elif name == "hover":
            await hover(page, args["element_id"])
        elif name == "wait":
            await wait(args["seconds"])
        elif name == "get_page_text":
            text_payload = await get_page_text(page)
        elif name == "search_web":
            text_payload = await search_web(page, args["query"])
        elif name == "extract_table":
            text_payload = await extract_table(page, args["selector"])
        elif name == "go_back":
            await go_back(page)
        else:
            raise ValueError(f"Unknown action: {name}")

        elements = await extract_interactive_elements(page)
        screenshot_b64 = await capture_annotated_screenshot(page, elements)
        payload = text_payload or None if name in _TEXT_RETURNING else None
        return ActionResult(success=True, new_url=page.url, error=payload, screenshot_b64=screenshot_b64)

    except Exception as e:  # noqa: BLE001 - intentionally broad: any tool failure becomes a recoverable ActionResult
        screenshot_b64 = ""
        try:
            elements = await extract_interactive_elements(page)
            screenshot_b64 = await capture_annotated_screenshot(page, elements)
        except Exception:
            pass
        return ActionResult(success=False, new_url=page.url, error=str(e), screenshot_b64=screenshot_b64)


if __name__ == "__main__":
    import asyncio

    async def _demo():
        session = BrowserSession()
        await session.start(headless=False)
        result = await dispatch_action(session, "navigate_to", {"url": "https://example.com"})
        print(result.success, result.new_url)
        await session.stop()

    asyncio.run(_demo())
```

Note on the `error` field misuse above for text-returning actions: this is a deliberate, narrow convention — `get_page_text`/`search_web`/`extract_table` are read-only "return data" actions, and `ActionResult` (Task 2) has no separate `data` field. Task 14's reasoning loop checks `name in _TEXT_RETURNING` (re-imported from `agent.actions`) and reads `result.error` as the payload instead of an error message when `result.success` is `True`. This is documented again in Task 14 so the reasoning loop implementer doesn't miss it.

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_actions.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add agent/actions.py tests/test_actions.py
git commit -m "feat: action dispatch table and Gemini function-calling schemas"
```

---

### Task 13: Task memory (`agent/memory.py`)

**Files:**
- Create: `agent/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `class TaskMemory` with fields `scratchpad: str`, `visited_urls: list[str]`, `extracted_data: dict`, `action_history: list[dict]`; methods `update_scratchpad(note: str) -> None`, `record_action(action: str, result: str, url: str) -> None`, `is_looping(threshold: int = 3) -> bool` (True if current URL appears `>= threshold` times in `visited_urls`), `recent_history_text(n: int = 5) -> str`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_memory.py
from agent.memory import TaskMemory


def test_update_scratchpad_appends_timestamped_note():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.update_scratchpad("Found the price: $42")

    # Assert
    assert "Found the price: $42" in memory.scratchpad


def test_record_action_tracks_url_and_history():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.record_action("click", "success", "http://example.com/page1")

    # Assert
    assert memory.visited_urls == ["http://example.com/page1"]
    assert memory.action_history[-1]["action"] == "click"
    assert memory.action_history[-1]["url"] == "http://example.com/page1"


def test_is_looping_true_when_url_repeats_past_threshold():
    # Arrange
    memory = TaskMemory()
    for _ in range(3):
        memory.record_action("click", "success", "http://example.com/loop")

    # Act / Assert
    assert memory.is_looping() is True


def test_is_looping_false_when_url_visited_once():
    # Arrange
    memory = TaskMemory()
    memory.record_action("click", "success", "http://example.com/once")

    # Act / Assert
    assert memory.is_looping() is False


def test_recent_history_text_returns_last_n_entries_only():
    # Arrange
    memory = TaskMemory()
    for i in range(8):
        memory.record_action(f"action_{i}", "success", f"http://x/{i}")

    # Act
    text = memory.recent_history_text(n=5)

    # Assert
    assert "action_7" in text
    assert "action_2" not in text
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.memory'`

- [ ] **Step 3: Implement `agent/memory.py`**

```python
# agent/memory.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskMemory:
    scratchpad: str = ""
    visited_urls: list[str] = field(default_factory=list)
    extracted_data: dict = field(default_factory=dict)
    action_history: list[dict] = field(default_factory=list)

    def update_scratchpad(self, note: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scratchpad += f"[{timestamp}] {note}\n"

    def record_action(self, action: str, result: str, url: str) -> None:
        self.visited_urls.append(url)
        self.action_history.append({"action": action, "result": result, "url": url})

    def is_looping(self, threshold: int = 3) -> bool:
        if not self.visited_urls:
            return False
        current = self.visited_urls[-1]
        return self.visited_urls.count(current) >= threshold

    def recent_history_text(self, n: int = 5) -> str:
        recent = self.action_history[-n:]
        lines = [f"- {h['action']} on {h['url']} -> {h['result']}" for h in recent]
        return "\n".join(lines)


if __name__ == "__main__":
    m = TaskMemory()
    m.update_scratchpad("starting task")
    m.record_action("navigate_to", "success", "https://example.com")
    print(m.scratchpad)
    print(m.recent_history_text())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_memory.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add agent/memory.py tests/test_memory.py
git commit -m "feat: task memory with scratchpad, loop detection, and action history"
```

---

### Task 14: ReAct reasoning loop with Gemini (`agent/reasoning.py`)

**Files:**
- Create: `agent/reasoning.py`
- Test: `tests/test_reasoning.py`

**Interfaces:**
- Consumes: `agent.browser.BrowserSession`, `agent.actions.ACTION_SCHEMAS`, `agent.actions.dispatch_action`, `agent.actions._TEXT_RETURNING`, `agent.memory.TaskMemory`, `agent.types.{Step, AgentRun}`, `agent.config.Settings`
- Produces: `class WebAgentReasoner` with `__init__(self, settings: Settings)` and `async def run(self, task: str, session: BrowserSession, max_steps: int | None = None, tracer=None) -> AgentRun`. The Gemini client itself (`google.genai.Client`) is injectable via a constructor kwarg `client=None` (defaults to a real client) so tests can pass a fake.

This is the most behavior-dense module. Tests use a **fake Gemini client** (a small stub implementing the same `.models.generate_content(...)` call signature) so the loop's control flow — not Gemini's actual reasoning — is what's under test. The real `google-genai` wiring is verified manually in Task 15's smoke test against a live task, per the spec's "after each step: smoke test" constraint.

- [ ] **Step 1: Write failing test**

```python
# tests/test_reasoning.py
import pytest
from agent.browser import BrowserSession
from agent.config import Settings
from agent.reasoning import WebAgentReasoner


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakePart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeUsage:
    total_token_count = 100


class _FakeResponse:
    def __init__(self, parts):
        self.candidates = [_FakeCandidate(parts)]
        self.usage_metadata = _FakeUsage()


class _ScriptedGeminiClient:
    """Replays a fixed sequence of responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.models = self
        self.call_count = 0

    def generate_content(self, model, contents, config):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def _response_for(action_name: str, args: dict, reasoning_text: str = "thinking..."):
    return _FakeResponse([
        _FakePart(text=reasoning_text),
        _FakePart(function_call=_FakeFunctionCall(action_name, args)),
    ])


@pytest.mark.asyncio
async def test_run_executes_navigate_then_task_complete(static_server):
    # Arrange
    fake_client = _ScriptedGeminiClient([
        _response_for("navigate_to", {"url": f"{static_server}/sample_page.html"}),
        _response_for("task_complete", {"result": "Found the fixture page."}),
    ])
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
    session = BrowserSession()
    await session.start(headless=True)

    try:
        # Act
        run = await reasoner.run("Go to the fixture page", session, max_steps=5)

        # Assert
        assert run.success is True
        assert run.result == "Found the fixture page."
        assert run.total_actions == 2
        assert run.final_url.endswith("sample_page.html")
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_stops_at_max_steps_if_never_completes(static_server):
    # Arrange: scripted client returns the same harmless action forever
    responses = [_response_for("get_page_text", {}) for _ in range(10)]
    fake_client = _ScriptedGeminiClient(responses)
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Never-ending task", session, max_steps=3)

        # Assert
        assert run.success is False
        assert len(run.steps) == 3
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_records_task_failed(static_server):
    # Arrange
    fake_client = _ScriptedGeminiClient([
        _response_for("task_failed", {"reason": "Login required"}),
    ])
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Log into a private dashboard", session, max_steps=5)

        # Assert
        assert run.success is False
        assert "Login required" in run.result
    finally:
        await session.stop()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_reasoning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.reasoning'`

- [ ] **Step 3: Implement `agent/reasoning.py`**

```python
# agent/reasoning.py
import time

from google import genai
from google.genai import types

from agent.actions import ACTION_SCHEMAS, _TEXT_RETURNING, dispatch_action
from agent.browser import BrowserSession
from agent.config import Settings
from agent.memory import TaskMemory
from agent.types import AgentRun, Step

SYSTEM_PROMPT = """You are an autonomous web agent. You control a real browser
to complete tasks. At each step you see:
- The current URL and page title
- A list of interactive elements on the page (with IDs)
- The accessibility tree of the page
- A screenshot with elements annotated

Your job: decide the single best next action to make progress
toward the task. Think step by step before acting:
1. Where am I? What does this page show?
2. What progress have I made so far?
3. What is the next logical step toward the goal?
4. Which action achieves that step?

Rules:
- One action per turn, no batching
- If lost: go_back or search_web to reorient
- If a page is slow: use wait() before interacting
- If task requires information: use get_page_text() to read it
- When task is fully complete: call task_complete with result
- If task is impossible (login required, CAPTCHA, etc.): task_failed
- Never loop on the same action more than 3 times"""


def _elements_as_text(elements) -> str:
    lines = []
    for e in elements:
        bits = [f'id="{e.id}"', f'tag="{e.tag}"']
        if e.text:
            bits.append(f'text="{e.text[:60]}"')
        if e.placeholder:
            bits.append(f'placeholder="{e.placeholder}"')
        if e.href:
            bits.append(f'href="{e.href}"')
        lines.append("  " + " ".join(bits))
    return "\n".join(lines) if lines else "  (no interactive elements found)"


class WebAgentReasoner:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.tools = [types.Tool(function_declarations=ACTION_SCHEMAS)]

    def _build_prompt(self, task: str, step_n: int, max_steps: int, memory: TaskMemory, state) -> str:
        return f"""Task: {task}
Step: {step_n} of {max_steps}
Scratchpad:
{memory.scratchpad or '(empty)'}

Recent actions:
{memory.recent_history_text() or '(none yet)'}

Current URL: {state.url}
Page title: {state.title}

Interactive elements:
{_elements_as_text(state.interactive_elements)}

Accessibility tree:
{state.accessibility_tree or '(empty)'}

What is your next action?"""

    async def run(self, task: str, session: BrowserSession, max_steps: int | None = None, tracer=None) -> AgentRun:
        max_steps = min(max_steps or self.settings.max_steps, self.settings.max_steps)
        memory = TaskMemory()
        steps: list[Step] = []
        start_time = time.monotonic()
        success = False
        result_text = ""

        for step_n in range(1, max_steps + 1):
            state = await session.get_page_state()
            prompt = self._build_prompt(task, step_n, max_steps, memory, state)

            image_part = types.Part.from_bytes(
                data=__import__("base64").b64decode(state.screenshot_b64), mime_type="image/png"
            )
            contents = [types.Content(role="user", parts=[types.Part(text=prompt), image_part])]
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(tools=self.tools, system_instruction=SYSTEM_PROMPT),
            )

            parts = response.candidates[0].content.parts
            reasoning_text = next((p.text for p in parts if getattr(p, "text", None)), "")
            function_call = next((p.function_call for p in parts if getattr(p, "function_call", None)), None)
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            if function_call is None:
                memory.update_scratchpad("No action returned by model; retrying next step.")
                continue

            action_name = function_call.name
            action_args = dict(function_call.args or {})

            if action_name == "task_complete":
                result_text = action_args.get("result", "")
                success = True
                action_result_for_step = None
                steps.append(Step(step_n, state.url, action_name, action_args, action_result_for_step, reasoning_text, tokens_used))
                if tracer:
                    tracer.log_step(step_n, state.url, action_name, action_args, None, reasoning_text, tokens_used)
                break

            if action_name == "task_failed":
                result_text = action_args.get("reason", "")
                success = False
                steps.append(Step(step_n, state.url, action_name, action_args, None, reasoning_text, tokens_used))
                if tracer:
                    tracer.log_step(step_n, state.url, action_name, action_args, None, reasoning_text, tokens_used)
                break

            action_result = await dispatch_action(session, action_name, action_args)
            steps.append(Step(step_n, state.url, action_name, action_args, action_result, reasoning_text, tokens_used))
            if tracer:
                tracer.log_step(step_n, state.url, action_name, action_args, action_result, reasoning_text, tokens_used)

            if action_name in _TEXT_RETURNING and action_result.success:
                memory.update_scratchpad(f"{action_name} returned: {action_result.error}")
            elif not action_result.success:
                memory.update_scratchpad(f"{action_name} failed: {action_result.error}")

            memory.record_action(action_name, "success" if action_result.success else "failed", action_result.new_url)

            if memory.is_looping():
                memory.update_scratchpad("Detected repeated visits to the same URL — try a different approach.")

        duration = time.monotonic() - start_time
        final_url = session.page.url if session.page else ""

        return AgentRun(
            task=task,
            success=success,
            result=result_text,
            steps=steps,
            total_actions=len(steps),
            total_tokens=sum(s.tokens_used for s in steps),
            duration_seconds=duration,
            final_url=final_url,
        )


if __name__ == "__main__":
    import asyncio

    from agent.config import load_settings

    async def _demo():
        settings = load_settings()
        session = BrowserSession()
        await session.start(headless=False)
        reasoner = WebAgentReasoner(settings)
        run = await reasoner.run("Go to https://example.com and report the page title", session, max_steps=5)
        print(run)
        await session.stop()

    asyncio.run(_demo())
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_reasoning.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add agent/reasoning.py tests/test_reasoning.py
git commit -m "feat: ReAct reasoning loop driving Gemini function calling"
```

---

### Task 15: CLI single-task runner + manual smoke test (`scripts/run_task.py`)

**Files:**
- Create: `scripts/run_task.py`
- Test: `tests/test_run_task.py`

**Interfaces:**
- Consumes: `agent.config.load_settings`, `agent.browser.BrowserSession`, `agent.reasoning.WebAgentReasoner`
- Produces: `async def run_task(task: str, headless: bool = False, max_steps: int | None = None) -> AgentRun`, plus `argparse`-based `main()` CLI entrypoint (`python scripts/run_task.py "task text" [--headless] [--max-steps N]`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_run_task.py
import pytest
from scripts.run_task import build_arg_parser


def test_arg_parser_requires_task_positional():
    # Arrange
    parser = build_arg_parser()

    # Act
    args = parser.parse_args(["do something", "--headless", "--max-steps", "10"])

    # Assert
    assert args.task == "do something"
    assert args.headless is True
    assert args.max_steps == 10


def test_arg_parser_defaults_headless_false_and_max_steps_none():
    # Arrange
    parser = build_arg_parser()

    # Act
    args = parser.parse_args(["do something"])

    # Assert
    assert args.headless is False
    assert args.max_steps is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_run_task.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_task'`

- [ ] **Step 3: Implement `scripts/run_task.py`**

```python
# scripts/run_task.py
import argparse
import asyncio

from agent.browser import BrowserSession
from agent.config import load_settings
from agent.reasoning import WebAgentReasoner
from agent.types import AgentRun


async def run_task(task: str, headless: bool = False, max_steps: int | None = None) -> AgentRun:
    settings = load_settings()
    session = BrowserSession()
    await session.start(headless=headless)
    try:
        reasoner = WebAgentReasoner(settings)
        return await reasoner.run(task, session, max_steps=max_steps)
    finally:
        await session.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single autonomous web agent task")
    parser.add_argument("task", help="Natural-language task description")
    parser.add_argument("--headless", action="store_true", help="Run Chromium headless")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max step count")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run = asyncio.run(run_task(args.task, headless=args.headless, max_steps=args.max_steps))
    print(f"\nSuccess: {run.success}")
    print(f"Result: {run.result}")
    print(f"Steps taken: {run.total_actions}")
    print(f"Tokens used: {run.total_tokens}")
    print(f"Duration: {run.duration_seconds:.1f}s")
    print(f"Final URL: {run.final_url}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_run_task.py -v`
Expected: 2 passed

- [ ] **Step 5: Manual smoke test against a real site (per spec's "after each step: smoke test" constraint)**

Run: `python scripts/run_task.py "Go to https://example.com and report the page title" --max-steps 5`
Expected: Chromium window opens (headless=False default), agent navigates, prints `Success: True` and a result mentioning "Example Domain". If this fails, stop and debug `agent/reasoning.py`'s Gemini wiring before proceeding to Task 16 — this is the first point the real Gemini API and real browser are exercised together.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_task.py tests/test_run_task.py
git commit -m "feat: CLI entrypoint to run a single agent task end-to-end"
```

---

### Task 16: Step tracer (`observability/tracer.py`)

**Files:**
- Create: `observability/tracer.py`
- Test: `tests/test_tracer.py`

**Interfaces:**
- Consumes: `agent.types.ActionResult`
- Produces: `class Tracer` with `__init__(self, session_id: str, base_dir: str = "results/traces")`, `.trace_path` (str, points at `{base_dir}/{session_id}/trace.jsonl`), `log_step(step_n: int, url: str, action_type: str, action_input: dict, action_result, reasoning_text: str, tokens_used: int) -> None` (appends one JSON line, also saves the action's screenshot to `{base_dir}/{session_id}/step_{n}.png` if `action_result` has a `screenshot_b64`, and prints a one-line live summary to stdout).

- [ ] **Step 1: Write failing test**

```python
# tests/test_tracer.py
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
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert lines[0]["action_type"] == "navigate_to"
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_tracer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'observability.tracer'`

- [ ] **Step 3: Implement `observability/tracer.py`**

```python
# observability/tracer.py
import json
import os
from datetime import datetime

from perception.screenshot import save_screenshot


class Tracer:
    def __init__(self, session_id: str, base_dir: str = "results/traces"):
        self.session_id = session_id
        self.dir_path = os.path.join(base_dir, session_id)
        os.makedirs(self.dir_path, exist_ok=True)
        self.trace_path = os.path.join(self.dir_path, "trace.jsonl")

    def log_step(self, step_n, url, action_type, action_input, action_result, reasoning_text, tokens_used) -> None:
        screenshot_path = None
        if action_result is not None and getattr(action_result, "screenshot_b64", ""):
            screenshot_path = os.path.join(self.dir_path, f"step_{step_n}.png")
            save_screenshot(action_result.screenshot_b64, screenshot_path)

        record = {
            "step": step_n,
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "action_type": action_type,
            "action_input": action_input,
            "action_result": {
                "success": getattr(action_result, "success", None),
                "error": getattr(action_result, "error", None),
            } if action_result is not None else None,
            "claude_reasoning": reasoning_text,
            "tokens_used": tokens_used,
            "screenshot_path": screenshot_path,
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        status = "OK" if record["action_result"] is None or record["action_result"]["success"] else "FAIL"
        print(f"Step {step_n} | {action_type:14s} | {json.dumps(action_input)[:60]:60s} | {status}")


if __name__ == "__main__":
    t = Tracer(session_id="demo")
    print(t.trace_path)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_tracer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add observability/tracer.py tests/test_tracer.py
git commit -m "feat: structured JSONL step tracer with live terminal summary"
```

---

### Task 17: Trace replay GIF generator (`observability/replay.py`)

**Files:**
- Create: `observability/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: a `trace.jsonl` file as written by Task 16's `Tracer`
- Produces: `def build_replay_gif(trace_dir: str, out_path: str, frame_duration_ms: int = 1200) -> str` — reads each `step_N.png` referenced in the trace, overlays step number + action + reasoning text, writes an animated GIF, returns `out_path`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_replay.py
import json
import os

from PIL import Image

from observability.replay import build_replay_gif


def _write_trace(trace_dir, n_steps):
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(trace_dir, "trace.jsonl"), "w") as f:
        for i in range(1, n_steps + 1):
            img_path = os.path.join(trace_dir, f"step_{i}.png")
            Image.new("RGB", (100, 80), color=(i * 10 % 255, 0, 0)).save(img_path)
            record = {
                "step": i, "action_type": "click", "claude_reasoning": f"reason {i}",
                "screenshot_path": img_path,
            }
            f.write(json.dumps(record) + "\n")


def test_build_replay_gif_creates_file_with_one_frame_per_step(tmp_path):
    # Arrange
    trace_dir = str(tmp_path / "session")
    _write_trace(trace_dir, n_steps=3)
    out_path = str(tmp_path / "replay.gif")

    # Act
    result_path = build_replay_gif(trace_dir, out_path)

    # Assert
    assert result_path == out_path
    assert os.path.exists(out_path)
    with Image.open(out_path) as gif:
        frame_count = gif.n_frames
    assert frame_count == 3
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'observability.replay'`

- [ ] **Step 3: Implement `observability/replay.py`**

```python
# observability/replay.py
import json
import os

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def build_replay_gif(trace_dir: str, out_path: str, frame_duration_ms: int = 1200) -> str:
    trace_path = os.path.join(trace_dir, "trace.jsonl")
    frames = []
    font = ImageFont.load_default()

    with open(trace_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            screenshot_path = record.get("screenshot_path")
            if not screenshot_path or not os.path.exists(screenshot_path):
                continue

            image = Image.open(screenshot_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            caption = f"Step {record['step']}: {record['action_type']} — {record.get('claude_reasoning', '')[:80]}"
            draw.rectangle([0, 0, image.width, 20], fill="black")
            draw.text((4, 4), caption, fill="white", font=font)
            frames.append(image)

    imageio.mimsave(out_path, [f.copy() for f in frames], duration=frame_duration_ms / 1000, loop=0)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python observability/replay.py <trace_dir> <out_path.gif>")
    else:
        print(build_replay_gif(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_replay.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add observability/replay.py tests/test_replay.py
git commit -m "feat: replay GIF generator for recorded trace sessions"
```

---

### Task 18: Benchmark task definitions (`evaluation/tasks.py`)

**Files:**
- Create: `evaluation/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `BENCHMARK_TASKS: list[dict]`, each `{"id": str, "category": str, "prompt": str}`. Exactly 25 entries, 5 per category, categories exactly `{"information_extraction", "multi_step_navigation", "form_interaction", "data_aggregation", "reasoning_over_content"}`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_tasks.py
from evaluation.tasks import BENCHMARK_TASKS

EXPECTED_CATEGORIES = {
    "information_extraction",
    "multi_step_navigation",
    "form_interaction",
    "data_aggregation",
    "reasoning_over_content",
}


def test_exactly_25_tasks_defined():
    assert len(BENCHMARK_TASKS) == 25


def test_each_category_has_exactly_5_tasks():
    counts = {}
    for t in BENCHMARK_TASKS:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert set(counts.keys()) == EXPECTED_CATEGORIES
    assert all(v == 5 for v in counts.values())


def test_every_task_has_unique_id_and_nonempty_prompt():
    ids = [t["id"] for t in BENCHMARK_TASKS]
    assert len(ids) == len(set(ids))
    assert all(t["prompt"].strip() for t in BENCHMARK_TASKS)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation.tasks'`

- [ ] **Step 3: Implement `evaluation/tasks.py`** (the 25 tasks verbatim from the spec's Step 7, tagged with category keys)

```python
# evaluation/tasks.py

BENCHMARK_TASKS: list[dict] = [
    # --- Category 1: Information Extraction ---
    {
        "id": "ie_01", "category": "information_extraction",
        "prompt": "Find the current price of Reliance Industries stock on NSE and return the price, change %, and 52-week high",
    },
    {
        "id": "ie_02", "category": "information_extraction",
        "prompt": "Go to arxiv.org and find the 3 most recent papers on 'retrieval augmented generation', return titles + abstracts",
    },
    {
        "id": "ie_03", "category": "information_extraction",
        "prompt": "Find the weather forecast for Mumbai for the next 5 days",
    },
    {
        "id": "ie_04", "category": "information_extraction",
        "prompt": "Find the top 5 results for 'best budget laptops India 2026' and extract product names and prices",
    },
    {
        "id": "ie_05", "category": "information_extraction",
        "prompt": "Go to IITB.ac.in and find the academic calendar for the current semester",
    },
    # --- Category 2: Multi-step Navigation ---
    {
        "id": "nav_01", "category": "multi_step_navigation",
        "prompt": "Go to HackerNews, find today's #1 story, open it, and summarize the article in 3 bullet points",
    },
    {
        "id": "nav_02", "category": "multi_step_navigation",
        "prompt": "Search for 'YC S25 batch' on Google, find the YC company list page, and return the first 10 company names and their one-line descriptions",
    },
    {
        "id": "nav_03", "category": "multi_step_navigation",
        "prompt": "Go to GitHub, search for 'FastMCP python', open the top result, and return the README summary and star count",
    },
    {
        "id": "nav_04", "category": "multi_step_navigation",
        "prompt": "Find the Wikipedia page for 'Limit order book', navigate to the 'Market microstructure' linked article, and return its first paragraph",
    },
    {
        "id": "nav_05", "category": "multi_step_navigation",
        "prompt": "Go to producthunt.com, find today's #1 product, and return its name, tagline, and upvote count",
    },
    # --- Category 3: Form Interaction ---
    {
        "id": "form_01", "category": "form_interaction",
        "prompt": "Go to DuckDuckGo, search for 'IIT Bombay CSE faculty', then refine the search to show only recent results from the last year",
    },
    {
        "id": "form_02", "category": "form_interaction",
        "prompt": "Go to Google Flights (no login), search for flights from Mumbai to Delhi tomorrow, return the cheapest 3 options with times and prices",
    },
    {
        "id": "form_03", "category": "form_interaction",
        "prompt": "Go to Wolfram Alpha and compute the integral of x^2 * sin(x) from 0 to pi. Return the result.",
    },
    {
        "id": "form_04", "category": "form_interaction",
        "prompt": "Use Google Scholar to find papers by 'Sunita Sarawagi' and return her h-index",
    },
    {
        "id": "form_05", "category": "form_interaction",
        "prompt": "Go to translate.google.com, translate 'The quick brown fox jumps over the lazy dog' to Hindi and return the translation",
    },
    # --- Category 4: Data Aggregation ---
    {
        "id": "agg_01", "category": "data_aggregation",
        "prompt": "Find the top 10 trending repositories on GitHub today, return name, stars, and description for each",
    },
    {
        "id": "agg_02", "category": "data_aggregation",
        "prompt": "Go to NSE India website, find the top 5 gainers and top 5 losers for today, return the full table",
    },
    {
        "id": "agg_03", "category": "data_aggregation",
        "prompt": "Find the current USD to INR, EUR to INR, and GBP to INR exchange rates from a financial site",
    },
    {
        "id": "agg_04", "category": "data_aggregation",
        "prompt": "Go to Cricbuzz, find the scorecard of the most recent completed international match, return full scores",
    },
    {
        "id": "agg_05", "category": "data_aggregation",
        "prompt": "Find the 5 most upvoted questions tagged 'python' on StackOverflow from the last week",
    },
    # --- Category 5: Reasoning Over Web Content ---
    {
        "id": "reason_01", "category": "reasoning_over_content",
        "prompt": "Search for 'India drone regulations 2025', read the top 3 results, and summarize the key rules in bullet points",
    },
    {
        "id": "reason_02", "category": "reasoning_over_content",
        "prompt": "Find recent news about Anthropic from the last month, read 3 articles, and identify the 3 biggest developments",
    },
    {
        "id": "reason_03", "category": "reasoning_over_content",
        "prompt": "Search for 'best MCP servers 2025', compile a list of the top 10 recommended servers with their use cases",
    },
    {
        "id": "reason_04", "category": "reasoning_over_content",
        "prompt": "Find the latest SEBI circular on F&O regulations, summarize what changed and who it affects",
    },
    {
        "id": "reason_05", "category": "reasoning_over_content",
        "prompt": "Research 'options trading strategies for beginners', read 2-3 sources, and produce a structured comparison of covered calls vs cash-secured puts",
    },
]

if __name__ == "__main__":
    for t in BENCHMARK_TASKS:
        print(t["id"], "-", t["category"])
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_tasks.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/tasks.py tests/test_tasks.py
git commit -m "feat: 25-task benchmark suite across 5 categories"
```

---

### Task 19: LLM-as-judge scoring (`evaluation/judge.py`)

**Files:**
- Create: `evaluation/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `agent.config.Settings`, `agent.types.AgentRun`
- Produces: `@dataclass JudgeScore(task_completed: bool, accuracy: int, completeness: int, efficiency: int)`, `class LLMJudge` with `__init__(self, settings: Settings, client=None)`, `def score(self, run: AgentRun) -> JudgeScore` (sync — judging happens after the run, no need for async). Sends the task, the run's `result`, and `total_actions` to Gemini with `response_mime_type="application/json"` and a strict JSON schema so parsing is deterministic.

- [ ] **Step 1: Write failing test**

```python
# tests/test_judge.py
import json

from agent.config import Settings
from agent.types import AgentRun
from evaluation.judge import JudgeScore, LLMJudge


class _FakeJudgeResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)


class _FakeJudgeClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.models = self

    def generate_content(self, model, contents, config):
        return _FakeJudgeResponse(self._payload)


def test_score_parses_json_response_into_judge_score():
    # Arrange
    payload = {"task_completed": True, "accuracy": 4, "completeness": 5, "efficiency": 3}
    fake_client = _FakeJudgeClient(payload)
    settings = Settings(gemini_api_key="fake-key")
    judge = LLMJudge(settings, client=fake_client)
    run = AgentRun(task="do x", success=True, result="did x", total_actions=3)

    # Act
    score = judge.score(run)

    # Assert
    assert isinstance(score, JudgeScore)
    assert score.task_completed is True
    assert score.accuracy == 4
    assert score.completeness == 5
    assert score.efficiency == 3


def test_score_handles_task_not_completed():
    # Arrange
    payload = {"task_completed": False, "accuracy": 1, "completeness": 1, "efficiency": 1}
    fake_client = _FakeJudgeClient(payload)
    settings = Settings(gemini_api_key="fake-key")
    judge = LLMJudge(settings, client=fake_client)
    run = AgentRun(task="do x", success=False, result="gave up", total_actions=25)

    # Act
    score = judge.score(run)

    # Assert
    assert score.task_completed is False
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation.judge'`

- [ ] **Step 3: Implement `evaluation/judge.py`**

```python
# evaluation/judge.py
import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from agent.config import Settings
from agent.types import AgentRun

JUDGE_PROMPT = """You are grading an autonomous web agent's attempt at a task.

Task: {task}
Agent's final result: {result}
Number of actions taken: {total_actions}

Score the attempt as JSON with exactly these fields:
- task_completed: boolean, did the agent actually accomplish the task?
- accuracy: integer 1-5, is the reported information correct/plausible?
- completeness: integer 1-5, did it cover everything the task asked for?
- efficiency: integer 1-5, was the number of actions reasonable (5=efficient, 1=wasteful)?
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "task_completed": {"type": "boolean"},
        "accuracy": {"type": "integer"},
        "completeness": {"type": "integer"},
        "efficiency": {"type": "integer"},
    },
    "required": ["task_completed", "accuracy", "completeness", "efficiency"],
}


@dataclass
class JudgeScore:
    task_completed: bool
    accuracy: int
    completeness: int
    efficiency: int


class LLMJudge:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or genai.Client(api_key=settings.gemini_api_key)

    def score(self, run: AgentRun) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(task=run.task, result=run.result, total_actions=run.total_actions)
        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        payload = json.loads(response.text)
        return JudgeScore(**payload)


if __name__ == "__main__":
    from agent.config import load_settings

    settings = load_settings()
    judge = LLMJudge(settings)
    demo_run = AgentRun(task="report the page title of example.com", success=True, result="Example Domain", total_actions=2)
    print(judge.score(demo_run))
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_judge.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/judge.py tests/test_judge.py
git commit -m "feat: LLM-as-judge scoring for completed benchmark runs"
```

---

### Task 20: Metrics aggregation (`evaluation/metrics.py`)

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `agent.types.AgentRun`, `evaluation.judge.JudgeScore`, `evaluation.tasks.BENCHMARK_TASKS` (for category lookup by task id)
- Produces: `def aggregate_metrics(results: list[dict]) -> dict` where each input dict is `{"task_id": str, "category": str, "run": AgentRun, "judge_score": JudgeScore, "failure_mode": str | None}`. Returns a dict with `overall_success_rate`, `by_category: dict[str, dict]` (`{"passed": int, "total": int}`), `avg_steps`, `avg_tokens`, `avg_time`, `failure_modes: dict[str, int]`. `def write_summary(metrics: dict, best_run: dict, out_path: str) -> None` writes the `results/summary.txt` format from the spec's Step 11.

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import os

from agent.types import AgentRun
from evaluation.judge import JudgeScore
from evaluation.metrics import aggregate_metrics, write_summary


def _result(task_id, category, success, steps, tokens, seconds, failure_mode=None):
    run = AgentRun(
        task=f"task for {task_id}", success=success, result="result text",
        total_actions=steps, total_tokens=tokens, duration_seconds=seconds,
    )
    score = JudgeScore(task_completed=success, accuracy=4, completeness=4, efficiency=4)
    return {"task_id": task_id, "category": category, "run": run, "judge_score": score, "failure_mode": failure_mode}


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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation.metrics'`

- [ ] **Step 3: Implement `evaluation/metrics.py`**

```python
# evaluation/metrics.py

CATEGORY_LABELS = {
    "information_extraction": "Information Extraction",
    "multi_step_navigation": "Multi-step Navigation",
    "form_interaction": "Form Interaction",
    "data_aggregation": "Data Aggregation",
    "reasoning_over_content": "Reasoning Over Content",
}


def aggregate_metrics(results: list[dict]) -> dict:
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

    return {
        "overall_success_rate": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "by_category": by_category,
        "avg_steps": sum(r["run"].total_actions for r in results) / total if total else 0.0,
        "avg_tokens": sum(r["run"].total_tokens for r in results) / total if total else 0.0,
        "avg_time": sum(r["run"].duration_seconds for r in results) / total if total else 0.0,
        "failure_modes": failure_modes,
    }


def write_summary(metrics: dict, best_run: dict, out_path: str) -> None:
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
    lines.append(f"    Avg time per task:       {metrics['avg_time']:.0f}s\n")
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
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: benchmark metrics aggregation and summary report writer"
```

---

### Task 21: Full benchmark runner (`scripts/run_benchmark.py`)

**Files:**
- Create: `scripts/run_benchmark.py`
- Test: `tests/test_run_benchmark.py`

**Interfaces:**
- Consumes: `evaluation.tasks.BENCHMARK_TASKS`, `agent.browser.BrowserSession`, `agent.reasoning.WebAgentReasoner`, `evaluation.judge.LLMJudge`, `evaluation.metrics.{aggregate_metrics, write_summary}`, `observability.tracer.Tracer`
- Produces: `def classify_failure_mode(run: AgentRun) -> str | None` (returns `"max_steps_reached"`, `"login_required"`, `"captcha_blocked"`, `"navigation_error"`, or `None` for success — heuristic string match against `run.result` for failed runs), `async def run_single_task(task: dict, settings, headless: bool = True) -> dict` (returns the per-task result dict shape consumed by `aggregate_metrics`), `async def run_benchmark(headless: bool = True) -> None` (loops all 25 tasks, saves each `AgentRun` to `results/benchmark/{task_id}.json`, writes `results/benchmark_summary.csv` via `pandas`, calls `write_summary`).

- [ ] **Step 1: Write failing test for the pure classification function (the only fully unit-testable piece — the rest is an integration entrypoint exercised by the manual smoke test in Step 6)**

```python
# tests/test_run_benchmark.py
from agent.types import AgentRun
from scripts.run_benchmark import classify_failure_mode


def test_classifies_max_steps_reached():
    run = AgentRun(task="x", success=False, result="", total_actions=25)
    assert classify_failure_mode(run) == "max_steps_reached"


def test_classifies_login_required():
    run = AgentRun(task="x", success=False, result="Login required to view this page", total_actions=4)
    assert classify_failure_mode(run) == "login_required"


def test_classifies_captcha_blocked():
    run = AgentRun(task="x", success=False, result="Blocked by a CAPTCHA challenge", total_actions=2)
    assert classify_failure_mode(run) == "captcha_blocked"


def test_classifies_navigation_error_as_fallback():
    run = AgentRun(task="x", success=False, result="Could not find the requested element", total_actions=10)
    assert classify_failure_mode(run) == "navigation_error"


def test_returns_none_for_successful_run():
    run = AgentRun(task="x", success=True, result="done", total_actions=5)
    assert classify_failure_mode(run) is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_run_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_benchmark'`

- [ ] **Step 3: Implement `scripts/run_benchmark.py`**

```python
# scripts/run_benchmark.py
import asyncio
import json
import os
from dataclasses import asdict

import pandas as pd

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


async def run_single_task(task: dict, settings: Settings, headless: bool = True) -> dict:
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
    failure_mode = classify_failure_mode(run)

    return {
        "task_id": task["id"],
        "category": task["category"],
        "run": run,
        "judge_score": judge_score,
        "failure_mode": failure_mode,
    }


async def run_benchmark(headless: bool = True) -> None:
    settings = load_settings()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for task in BENCHMARK_TASKS:
        print(f"\n=== Running {task['id']} ({task['category']}) ===")
        result = await run_single_task(task, settings, headless=headless)
        results.append(result)

        with open(os.path.join(RESULTS_DIR, f"{task['id']}.json"), "w") as f:
            json.dump({**asdict(result["run"]), "judge_score": asdict(result["judge_score"])}, f, default=str, indent=2)

    metrics = aggregate_metrics(results)
    best_run = max(results, key=lambda r: (r["run"].success, r["judge_score"].accuracy + r["judge_score"].completeness))

    df = pd.DataFrame([
        {
            "task_id": r["task_id"], "category": r["category"], "success": r["run"].success,
            "steps": r["run"].total_actions, "tokens": r["run"].total_tokens,
            "duration_s": r["run"].duration_seconds, "failure_mode": r["failure_mode"],
        }
        for r in results
    ])
    df.to_csv("results/benchmark_summary.csv", index=False)
    write_summary(metrics, best_run, out_path="results/summary.txt")

    print("\nBenchmark complete. See results/summary.txt and results/benchmark_summary.csv")


if __name__ == "__main__":
    asyncio.run(run_benchmark(headless=True))
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_run_benchmark.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke test against 1-2 real tasks before running the full 25 (per the spec's safety/cost concerns — full run costs real Gemini tokens and takes a long time)**

Temporarily edit `BENCHMARK_TASKS` usage for a quick check:
```bash
python3 -c "
import asyncio
from agent.config import load_settings
from evaluation.tasks import BENCHMARK_TASKS
from scripts.run_benchmark import run_single_task

async def _smoke():
    settings = load_settings()
    result = await run_single_task(BENCHMARK_TASKS[0], settings, headless=True)
    print(result['task_id'], result['run'].success, result['run'].result[:200])

asyncio.run(_smoke())
"
```
Expected: prints a task id, a success bool, and a non-empty result snippet without raising. If it raises on the Gemini call, fix `agent/reasoning.py` wiring before running the full benchmark.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_benchmark.py tests/test_run_benchmark.py
git commit -m "feat: full 25-task benchmark runner with judge scoring and CSV export"
```

---

### Task 22: Streamlit UI (`ui/app.py`)

**Files:**
- Create: `ui/app.py`
- Test: `tests/test_ui_helpers.py`

**Interfaces:**
- Consumes: `scripts.run_task.run_task`, `agent.types.AgentRun`, `observability.replay.build_replay_gif`
- Produces: `def format_step_line(step) -> str` (pure helper, unit-tested), `def run_task_sync(task: str, headless: bool, max_steps: int | None) -> AgentRun` (wraps `asyncio.run` for Streamlit's sync execution model), and the Streamlit page itself (not unit-tested — verified by manual run in Step 4).

Streamlit scripts execute top-to-bottom on every interaction and aren't natively async-test-friendly, so only the pure formatting/wrapper helpers get unit tests; the page layout is verified by actually running it.

- [ ] **Step 1: Write failing test for the pure helper**

```python
# tests/test_ui_helpers.py
from agent.types import ActionResult, Step
from ui.app import format_step_line


def test_format_step_line_includes_step_number_action_and_status():
    # Arrange
    result = ActionResult(success=True, new_url="http://x/", error=None, screenshot_b64="")
    step = Step(
        number=2, url="http://x/", action_type="click", action_input={"element_id": "btn_0"},
        action_result=result, reasoning_text="clicking search button", tokens_used=80,
    )

    # Act
    line = format_step_line(step)

    # Assert
    assert "[Step 2]" in line
    assert "click" in line
    assert "OK" in line
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_ui_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.app'`

- [ ] **Step 3: Implement `ui/app.py`**

```python
# ui/app.py
import asyncio
import os

import streamlit as st

from agent.types import AgentRun, Step
from observability.replay import build_replay_gif
from scripts.run_task import run_task


def format_step_line(step: Step) -> str:
    status = "OK" if (step.action_result is None or step.action_result.success) else "FAIL"
    return f"[Step {step.number}] {step.action_type}: {step.action_input} -> {status}"


def run_task_sync(task: str, headless: bool, max_steps: int | None) -> AgentRun:
    return asyncio.run(run_task(task, headless=headless, max_steps=max_steps))


def _render_page() -> None:
    st.set_page_config(page_title="Autonomous Web Agent", layout="wide")
    st.title("Autonomous Web Agent")

    if "history" not in st.session_state:
        st.session_state.history = []

    task_text = st.text_area("Task", placeholder="e.g. Find today's #1 Hacker News story and summarize it")
    headless = st.checkbox("Headless", value=True)
    max_steps = st.number_input("Max steps", min_value=1, max_value=25, value=25)

    if st.button("Run") and task_text.strip():
        with st.spinner("Agent working..."):
            run = run_task_sync(task_text, headless=headless, max_steps=int(max_steps))
        st.session_state.history.insert(0, run)

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.subheader("Result")
        st.write(f"**Success:** {latest.success}")
        st.write(f"**Answer:** {latest.result}")
        st.write(f"Steps: {latest.total_actions} | Tokens: {latest.total_tokens} | "
                 f"Time: {latest.duration_seconds:.1f}s | Final URL: {latest.final_url}")

        st.subheader("Step-by-step trace")
        for step in latest.steps:
            with st.expander(format_step_line(step)):
                st.text(step.reasoning_text)
                if step.action_result and step.action_result.screenshot_b64:
                    st.image(f"data:image/png;base64,{step.action_result.screenshot_b64}")

        if st.button("View full trace (raw JSON)"):
            st.json([s.__dict__ for s in latest.steps])

        st.subheader("History")
        for i, run in enumerate(st.session_state.history[:10]):
            st.write(f"{i + 1}. {'✅' if run.success else '❌'} {run.task[:80]}")


if __name__ == "__main__":
    _render_page()
```

- [ ] **Step 4: Run to confirm test passes, then manually launch the UI**

Run: `pytest tests/test_ui_helpers.py -v`
Expected: 1 passed

Run: `streamlit run ui/app.py`
Expected: browser tab opens with the task input form; submit a simple task (e.g. "Go to https://example.com and report the page title") and confirm the live result, step trace, and history render without errors.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py tests/test_ui_helpers.py
git commit -m "feat: Streamlit UI for submitting tasks and watching live agent runs"
```

---

### Task 23: README

**Files:**
- Modify: `README.md` (replace stale placeholder content entirely)

- [ ] **Step 1: Replace `README.md`**

```markdown
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
```

- [ ] **Step 2: Create `.env.example` referenced by the README**

```bash
# .env.example
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash
```

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: project README with architecture, usage, and limitations"
```

---

## Final Validation

- [ ] Run the full test suite: `pytest tests/ -v` — all tests pass (network-dependent search test may be skipped with `SKIP_NETWORK_TESTS=1` if offline).
- [ ] Run `python scripts/run_task.py "Go to https://example.com and report the page title"` and confirm a visible Chromium window completes the task.
- [ ] Run `streamlit run ui/app.py` and submit one task through the browser UI.
- [ ] Run `python scripts/run_benchmark.py` (costs real Gemini tokens and takes real wall-clock time against live sites — confirm with the user before running the full 25-task suite) and confirm `results/summary.txt` and `results/benchmark_summary.csv` are produced.

