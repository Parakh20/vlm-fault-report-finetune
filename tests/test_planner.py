import pytest

from agent.llm.base import LLMResponse
from agent.planner import PlannerAgent


class _FakeRouter:
    def __init__(self, text: str):
        self.text = text

    async def plan(self, system_prompt, user_prompt, screenshot_b64, tools):
        return LLMResponse(text=self.text, tool_call=None, tokens_used=10)


@pytest.mark.asyncio
async def test_plan_parses_json_array_of_steps():
    # Arrange
    router = _FakeRouter('["Go to site", "Search for X", "Extract Y"]')
    planner = PlannerAgent(settings=None, router=router)

    # Act
    steps = await planner.plan("do the thing")

    # Assert
    assert steps == ["Go to site", "Search for X", "Extract Y"]


@pytest.mark.asyncio
async def test_plan_extracts_json_array_embedded_in_extra_text():
    # Arrange
    router = _FakeRouter('Sure, here is the plan:\n["Step one", "Step two"]\nHope that helps!')
    planner = PlannerAgent(settings=None, router=router)

    # Act
    steps = await planner.plan("do the thing")

    # Assert
    assert steps == ["Step one", "Step two"]


@pytest.mark.asyncio
async def test_plan_falls_back_to_the_task_itself_when_unparseable():
    # Arrange
    router = _FakeRouter("I refuse to produce JSON today.")
    planner = PlannerAgent(settings=None, router=router)

    # Act
    steps = await planner.plan("do the thing")

    # Assert
    assert steps == ["do the thing"]
