import json
import os

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def build_replay_gif(trace_dir: str, out_path: str, frame_duration_ms: int = 1200) -> str:
    trace_path = os.path.join(trace_dir, "trace.jsonl")
    frames = []
    font = ImageFont.load_default()

    with open(trace_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            screenshot_path = record.get("screenshot_path")
            if not screenshot_path or not os.path.exists(screenshot_path):
                continue

            image = Image.open(screenshot_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            caption = f"Step {record['step']}: {record['action_type']} — {record.get('claude_reasoning', '')[:80]}"
            draw.rectangle([0, 0, image.width, 20], fill="black")
            draw.text((4, 4), caption, fill="white", font=font)
            frames.append(image)

    imageio.mimsave(out_path, [f.copy() for f in frames], duration=frame_duration_ms / 1000, loop=0)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python observability/replay.py <trace_dir> <out_path.gif>")
    else:
        print(build_replay_gif(sys.argv[1], sys.argv[2]))
