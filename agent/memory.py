from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskMemory:
    goal: str = ""
    scratchpad: str = ""
    visited_urls: list[str] = field(default_factory=list)
    extracted_data: dict = field(default_factory=dict)
    action_history: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def update_scratchpad(self, note: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scratchpad += f"[{timestamp}] {note}\n"

    def record_action(self, action: str, result: str, url: str, error: str | None = None) -> None:
        self.visited_urls.append(url)
        entry = {"action": action, "result": result, "url": url}
        self.action_history.append(entry)
        if result != "success":
            self.failures.append({**entry, "error": error})

    def record_fact(self, key: str, value) -> None:
        """Persist a piece of information the agent extracted (page text,
        table data, search results) so later steps can reference it without
        re-deriving it from the scratchpad's free text."""
        self.extracted_data[key] = value

    def is_looping(self, threshold: int = 3) -> bool:
        if not self.visited_urls:
            return False
        current = self.visited_urls[-1]
        return self.visited_urls.count(current) >= threshold

    def is_repeating_failure(self, action: str, threshold: int = 2) -> bool:
        """True once the same action has failed `threshold`+ times in a
        row, so the agent can be nudged toward a different approach instead
        of retrying a dead end."""
        recent_failures = self.failures[-threshold:]
        if len(recent_failures) < threshold:
            return False
        return all(f["action"] == action for f in recent_failures)

    def recent_history_text(self, n: int = 5) -> str:
        recent = self.action_history[-n:]
        lines = [f"- {h['action']} on {h['url']} -> {h['result']}" for h in recent]
        return "\n".join(lines)

    def failures_text(self, n: int = 5) -> str:
        recent = self.failures[-n:]
        lines = [f"- {f['action']} on {f['url']} failed: {f.get('error') or 'unknown error'}" for f in recent]
        return "\n".join(lines) if lines else ""

    def facts_text(self) -> str:
        if not self.extracted_data:
            return ""
        lines = [f"- {key}: {value}" for key, value in self.extracted_data.items()]
        return "\n".join(lines)


if __name__ == "__main__":
    m = TaskMemory(goal="find the price")
    m.update_scratchpad("starting task")
    m.record_action("navigate_to", "success", "https://example.com")
    m.record_fact("price", "$42")
    print(m.scratchpad)
    print(m.recent_history_text())
    print(m.facts_text())
