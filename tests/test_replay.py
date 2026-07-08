import json
import os

from PIL import Image

from observability.replay import build_replay_gif


def _write_trace(trace_dir, n_steps):
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(trace_dir, "trace.jsonl"), "w") as f:
        for i in range(1, n_steps + 1):
            img_path = os.path.join(trace_dir, f"step_{i}.png")
            Image.new("RGB", (100, 80), color=(i * 10 % 255, 0, 0)).save(img_path)
            record = {
                "step": i, "action_type": "click", "claude_reasoning": f"reason {i}",
                "screenshot_path": img_path,
            }
            f.write(json.dumps(record) + "\n")


def test_build_replay_gif_creates_file_with_one_frame_per_step(tmp_path):
    # Arrange
    trace_dir = str(tmp_path / "session")
    _write_trace(trace_dir, n_steps=3)
    out_path = str(tmp_path / "replay.gif")

    # Act
    result_path = build_replay_gif(trace_dir, out_path)

    # Assert
    assert result_path == out_path
    assert os.path.exists(out_path)
    with Image.open(out_path) as gif:
        frame_count = gif.n_frames
    assert frame_count == 3


def test_build_replay_gif_accepts_gemini_reasoning_field(tmp_path):
    # Arrange
    trace_dir = str(tmp_path / "session")
    os.makedirs(trace_dir, exist_ok=True)
    img_path = os.path.join(trace_dir, "step_1.png")
    Image.new("RGB", (100, 80), color=(20, 0, 0)).save(img_path)
    with open(os.path.join(trace_dir, "trace.jsonl"), "w") as f:
        record = {
            "step": 1,
            "action_type": "click",
            "gemini_reasoning": "reason 1",
            "screenshot_path": img_path,
        }
        f.write(json.dumps(record) + "\n")
    out_path = str(tmp_path / "replay.gif")

    # Act
    result_path = build_replay_gif(trace_dir, out_path)

    # Assert
    assert result_path == out_path
    assert os.path.exists(out_path)
