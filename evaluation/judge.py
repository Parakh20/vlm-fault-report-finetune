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
