from agent.memory import TaskMemory


def test_update_scratchpad_appends_timestamped_note():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.update_scratchpad("Found the price: $42")

    # Assert
    assert "Found the price: $42" in memory.scratchpad


def test_record_action_tracks_url_and_history():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.record_action("click", "success", "http://example.com/page1")

    # Assert
    assert memory.visited_urls == ["http://example.com/page1"]
    assert memory.action_history[-1]["action"] == "click"
    assert memory.action_history[-1]["url"] == "http://example.com/page1"


def test_is_looping_true_when_url_repeats_past_threshold():
    # Arrange
    memory = TaskMemory()
    for _ in range(3):
        memory.record_action("click", "success", "http://example.com/loop")

    # Act / Assert
    assert memory.is_looping() is True


def test_is_looping_false_when_url_visited_once():
    # Arrange
    memory = TaskMemory()
    memory.record_action("click", "success", "http://example.com/once")

    # Act / Assert
    assert memory.is_looping() is False


def test_recent_history_text_returns_last_n_entries_only():
    # Arrange
    memory = TaskMemory()
    for i in range(8):
        memory.record_action(f"action_{i}", "success", f"http://x/{i}")

    # Act
    text = memory.recent_history_text(n=5)

    # Assert
    assert "action_7" in text
    assert "action_2" not in text
