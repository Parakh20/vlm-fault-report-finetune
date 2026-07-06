from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskMemory:
    scratchpad: str = ""
    visited_urls: list[str] = field(default_factory=list)
    extracted_data: dict = field(default_factory=dict)
    action_history: list[dict] = field(default_factory=list)

    def update_scratchpad(self, note: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scratchpad += f"[{timestamp}] {note}\n"

    def record_action(self, action: str, result: str, url: str) -> None:
        self.visited_urls.append(url)
        self.action_history.append({"action": action, "result": result, "url": url})

    def is_looping(self, threshold: int = 3) -> bool:
        if not self.visited_urls:
            return False
        current = self.visited_urls[-1]
        return self.visited_urls.count(current) >= threshold

    def recent_history_text(self, n: int = 5) -> str:
        recent = self.action_history[-n:]
        lines = [f"- {h['action']} on {h['url']} -> {h['result']}" for h in recent]
        return "\n".join(lines)


if __name__ == "__main__":
    m = TaskMemory()
    m.update_scratchpad("starting task")
    m.record_action("navigate_to", "success", "https://example.com")
    print(m.scratchpad)
    print(m.recent_history_text())
