import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from perception.screenshot import save_screenshot


class Tracer:
    def __init__(self, session_id: str, base_dir: str = "results/traces"):
        self.session_id = session_id
        self.dir_path = os.path.join(base_dir, session_id)
        os.makedirs(self.dir_path, exist_ok=True)
        self.trace_path = os.path.join(self.dir_path, "trace.jsonl")

    def log_step(
        self,
        step_n: int,
        url: str,
        action_type: str,
        action_input: dict,
        action_result,
        reasoning_text: str,
        tokens_used: int,
    ) -> None:
        screenshot_path = None
        if action_result is not None and getattr(action_result, "screenshot_b64", ""):
            screenshot_path = os.path.join(self.dir_path, f"step_{step_n}.png")
            save_screenshot(action_result.screenshot_b64, screenshot_path)

        action_result_record = None
        if action_result is not None:
            action_result_record = {
                "success": getattr(action_result, "success", None),
                "new_url": getattr(action_result, "new_url", None),
                "error": getattr(action_result, "error", None),
            }

        record = {
            "step": step_n,
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "action_type": action_type,
            "action_input": action_input,
            "action_result": action_result_record,
            "gemini_reasoning": reasoning_text,
            "tokens_used": tokens_used,
            "screenshot_path": screenshot_path,
        }

        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        success = action_result_record is None or action_result_record["success"]
        status = "OK" if success else "FAIL"
        action_summary = json.dumps(action_input)[:60]
        print(f"Step {step_n} | {action_type:14s} | {action_summary:60s} | {status}")


if __name__ == "__main__":
    tracer = Tracer(session_id="demo")
    print(tracer.trace_path)
