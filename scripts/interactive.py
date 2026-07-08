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

async def interactive_loop(headless: bool = False, max_steps: int = 10):
    settings = load_settings()
    session = BrowserSession(engine=settings.browser_engine)
    
    print("Starting browser session...")
    await session.start(headless=headless)
    
    try:
        reasoner = WebAgentReasoner(settings)
        print("\n=== Autonomous Web Agent (Interactive Mode) ===")
        print("Type 'exit' or 'quit' to stop the agent and close the browser.")
        
        while True:
            try:
                task = input("\n> Enter your next instruction: ").strip()
                if not task:
                    continue
                if task.lower() in ("exit", "quit"):
                    break
                    
                print(f"Running agent for: '{task}'...")
                run = await reasoner.run(task, session, max_steps=max_steps)
                
                print(f"\n--- Task Finished ---")
                print(f"Success: {run.success}")
                print(f"Result: {run.result}")
                print(f"Steps taken: {run.total_actions}")
                print(f"Tokens used: {run.total_tokens}")
                print(f"Final URL: {run.final_url}")
                
            except KeyboardInterrupt:
                print("\nTask interrupted. You can enter a new instruction or type 'exit' to quit.")
                continue
                
    finally:
        print("Closing browser...")
        await session.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the web agent in an interactive continuous loop")
    parser.add_argument("--headless", action="store_true", help="Run Chromium headless")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per instruction")
    args = parser.parse_args()
    
    asyncio.run(interactive_loop(headless=args.headless, max_steps=args.max_steps))
