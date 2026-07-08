import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from agent.browser import BrowserSession
from agent.types import ActionResult
from perception.dom_parser import extract_interactive_elements
from perception.screenshot import capture_annotated_screenshot
from tools.extract import extract_table, get_page_text
from tools.interact import click, hover, press_key, scroll, select_option, type_text
from tools.navigate import go_back, go_to_url
from tools.parallel import extract_from_urls_parallel
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
        "description": "Click an interactive element by its id, such as 'btn_3' or 'link_0'.",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "string"}},
            "required": ["element_id"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input or textarea element by id, clearing it first by default.",
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
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "top", "bottom"],
                },
                "amount": {"type": "integer"},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option by value in a select element by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["element_id", "value"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key, such as 'Enter', 'Tab', or 'Escape'.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "hover",
        "description": "Hover over an interactive element by id to reveal menus or tooltips.",
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
        "description": "Search DuckDuckGo and return the top 5 results with title, URL, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "extract_table",
        "description": "Extract an HTML table as markdown using a CSS selector, such as '#data-table'.",
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in browser history.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "extract_from_urls",
        "description": (
            "Open up to 5 URLs concurrently in new tabs and extract their visible text. "
            "Useful for research/comparison tasks where several pages need to be read at "
            "once instead of one at a time, e.g. comparing search results or product pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["urls"],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal that the task is fully complete and report the final result.",
        "parameters": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    },
    {
        "name": "task_failed",
        "description": "Signal that the task cannot be completed and explain why.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

_TEXT_RETURNING = {"get_page_text", "search_web", "extract_table", "extract_from_urls"}

_ActionHandler = Callable[[Any, dict], Awaitable[str | None]]


async def _navigate_to(page: Any, args: dict) -> None:
    await go_to_url(page, args["url"])


async def _click(page: Any, args: dict) -> None:
    await click(page, args["element_id"])


async def _type_text(page: Any, args: dict) -> None:
    await type_text(
        page,
        args["element_id"],
        args["text"],
        args.get("clear_first", True),
    )


async def _scroll(page: Any, args: dict) -> None:
    await scroll(page, args["direction"], args.get("amount", 300))


async def _select_option(page: Any, args: dict) -> None:
    await select_option(page, args["element_id"], args["value"])


async def _press_key(page: Any, args: dict) -> None:
    await press_key(page, args["key"])


async def _hover(page: Any, args: dict) -> None:
    await hover(page, args["element_id"])


async def _wait(_: Any, args: dict) -> None:
    await wait(args["seconds"])


async def _get_page_text(page: Any, _: dict) -> str:
    return await get_page_text(page)


async def _search_web(page: Any, args: dict) -> str:
    return await search_web(page, args["query"], args.get("search_url"))


async def _extract_table(page: Any, args: dict) -> str:
    return await extract_table(page, args["selector"])


async def _go_back(page: Any, _: dict) -> None:
    await go_back(page)


async def _extract_from_urls(page: Any, args: dict) -> str:
    return await extract_from_urls_parallel(page, args.get("urls", []))


async def _task_complete(_: Any, args: dict) -> str | None:
    return args.get("result")


async def _task_failed(_: Any, args: dict) -> str | None:
    return args.get("reason")


_ACTION_HANDLERS: dict[str, _ActionHandler] = {
    "navigate_to": _navigate_to,
    "click": _click,
    "type_text": _type_text,
    "scroll": _scroll,
    "select_option": _select_option,
    "press_key": _press_key,
    "hover": _hover,
    "wait": _wait,
    "get_page_text": _get_page_text,
    "search_web": _search_web,
    "extract_table": _extract_table,
    "go_back": _go_back,
    "extract_from_urls": _extract_from_urls,
    "task_complete": _task_complete,
    "task_failed": _task_failed,
}


async def _capture_action_screenshot(page: Any) -> str:
    elements = await extract_interactive_elements(page)
    return await capture_annotated_screenshot(page, elements)


async def dispatch_action(session: BrowserSession, name: str, args: dict) -> ActionResult:
    page = session.page
    assert page is not None, "BrowserSession not started"

    try:
        handler = _ACTION_HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unknown action: {name}")

        text_payload = await handler(page, args)
        screenshot_b64 = await _capture_action_screenshot(page)
        payload = text_payload if name in _TEXT_RETURNING else None
        return ActionResult(
            success=True,
            new_url=page.url,
            error=payload,
            screenshot_b64=screenshot_b64,
        )
    except Exception as exc:  # noqa: BLE001 - action failures are recoverable loop state.
        screenshot_b64 = ""
        try:
            screenshot_b64 = await _capture_action_screenshot(page)
        except Exception:
            pass
        return ActionResult(
            success=False,
            new_url=page.url,
            error=str(exc),
            screenshot_b64=screenshot_b64,
        )


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        session = BrowserSession()
        await session.start(headless=True)
        try:
            result = await dispatch_action(
                session,
                "navigate_to",
                {"url": "data:text/html,<title>Smoke</title><h1>Smoke</h1>"},
            )
            print(result.success, result.new_url)
        finally:
            await session.stop()

    asyncio.run(_demo())
