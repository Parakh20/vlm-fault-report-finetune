import pytest

from agent.llm.base import LLMResponse
from agent.verifier import VerifierAgent


class _FakeRouter:
    def __init__(self, text: str):
        self.text = text

    async def plan(self, system_prompt, user_prompt, screenshot_b64, tools):
        return LLMResponse(text=self.text, tool_call=None, tokens_used=10)


@pytest.mark.asyncio
async def test_verify_accepts_a_credible_claim():
    # Arrange
    router = _FakeRouter('{"verified": true, "reason": "price is visible in the evidence"}')
    verifier = VerifierAgent(settings=None, router=router)

    # Act
    result = await verifier.verify("find the price", "$42", "Evidence: price shown as $42")

    # Assert
    assert result.verified is True
    assert "price" in result.reason


@pytest.mark.asyncio
async def test_verify_rejects_an_uncorroborated_claim():
    # Arrange
    router = _FakeRouter('{"verified": false, "reason": "no price appears anywhere in the evidence"}')
    verifier = VerifierAgent(settings=None, router=router)

    # Act
    result = await verifier.verify("find the price", "$42", "Evidence: page shows a 404 error")

    # Assert
    assert result.verified is False


@pytest.mark.asyncio
async def test_verify_fails_open_when_response_is_unparseable():
    # Arrange
    router = _FakeRouter("not json at all")
    verifier = VerifierAgent(settings=None, router=router)

    # Act
    result = await verifier.verify("find the price", "$42", "some evidence")

    # Assert: a broken verifier should not be able to block every run
    assert result.verified is True
