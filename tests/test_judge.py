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
