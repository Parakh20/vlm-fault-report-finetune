from __future__ import annotations

import json
import re

from agent.config import Settings, build_router
from agent.llm.router import LLMRouter

PLANNER_SYSTEM_PROMPT = """You are a planning assistant for an autonomous web agent.
Given a task, break it into a short ordered list of concrete subtasks (3-6 steps).
Return ONLY a JSON array of strings, e.g. ["Go to the site", "Search for X", "Extract Y"].
No other text, no markdown fences."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class PlannerAgent:
    """Produces a one-shot, non-binding subtask breakdown that gets folded
    into the Executor's prompt as guidance. The Executor still reasons step
    by step and can deviate from the plan — this isn't a rigid script."""

    def __init__(self, settings: Settings, router: LLMRouter | None = None):
        self.settings = settings
        self.router = router or build_router(settings)

    async def plan(self, task: str) -> list[str]:
        response = await self.router.plan(PLANNER_SYSTEM_PROMPT, task, None, [])
        steps = _parse_step_list(response.text)
        return steps if steps else [task]


def _parse_step_list(text: str) -> list[str]:
    candidates = [text.strip()]
    match = _JSON_ARRAY_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed:
            return [str(item) for item in parsed]
    return []
