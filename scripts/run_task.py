import argparse
import asyncio
import os
import sys

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from agent.browser import BrowserSession
from agent.config import load_settings
from agent.reasoning import WebAgentReasoner
from agent.types import AgentRun


async def run_task(
    task: str,
    headless: bool = False,
    max_steps: int | None = None,
) -> AgentRun:
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
    run = asyncio.run(
        run_task(args.task, headless=args.headless, max_steps=args.max_steps)
    )
    print(f"\nSuccess: {run.success}")
    print(f"Result: {run.result}")
    print(f"Steps taken: {run.total_actions}")
    print(f"Tokens used: {run.total_tokens}")
    print(f"Duration: {run.duration_seconds:.1f}s")
    print(f"Final URL: {run.final_url}")


if __name__ == "__main__":
    main()
