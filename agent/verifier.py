from __future__ import annotations

import json
from dataclasses import dataclass

from agent.config import Settings, build_router
from agent.llm.router import LLMRouter

VERIFIER_SYSTEM_PROMPT = """You are verifying whether an autonomous web agent actually
completed its assigned task, or is hallucinating a plausible-sounding result.
You will be given the task, the agent's claimed result, and evidence from the final
page state. Decide whether the claim is credible given that evidence.
Return ONLY JSON: {"verified": true or false, "reason": "short explanation"}.
No other text, no markdown fences."""


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    reason: str


class VerifierAgent:
    """Second opinion on the Executor's task_complete claim, checked against
    the actual final page state rather than trusting the model's own report
    of success. Fails open (verified=True) if its own response can't be
    parsed, since a broken verifier shouldn't be able to block every run."""

    def __init__(self, settings: Settings, router: LLMRouter | None = None):
        self.settings = settings
        self.router = router or build_router(settings)

    async def verify(self, task: str, claimed_result: str, final_state_summary: str) -> VerificationResult:
        prompt = (
            f"Task: {task}\n"
            f"Agent's claimed result: {claimed_result}\n\n"
            f"Evidence from the final page state:\n{final_state_summary}\n\n"
            "Is this claim credible?"
        )
        response = await self.router.plan(VERIFIER_SYSTEM_PROMPT, prompt, None, [])
        try:
            payload = json.loads(response.text.strip())
            return VerificationResult(
                verified=bool(payload.get("verified")),
                reason=str(payload.get("reason", "")),
            )
        except (json.JSONDecodeError, AttributeError):
            return VerificationResult(
                verified=True, reason="verifier response unparseable; defaulting to accept"
            )
