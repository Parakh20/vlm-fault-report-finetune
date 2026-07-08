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


def test_record_action_failure_is_tracked_separately():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.record_action("click", "failed", "http://example.com/page1", error="element not found")

    # Assert
    assert len(memory.failures) == 1
    assert memory.failures[0]["action"] == "click"
    assert memory.failures[0]["error"] == "element not found"


def test_record_action_success_does_not_add_to_failures():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.record_action("click", "success", "http://example.com/page1")

    # Assert
    assert memory.failures == []


def test_is_repeating_failure_true_after_threshold_consecutive_failures():
    # Arrange
    memory = TaskMemory()
    memory.record_action("click", "failed", "http://x", error="not found")
    memory.record_action("click", "failed", "http://x", error="not found")

    # Act / Assert
    assert memory.is_repeating_failure("click") is True


def test_is_repeating_failure_false_when_different_actions_fail():
    # Arrange
    memory = TaskMemory()
    memory.record_action("click", "failed", "http://x", error="not found")
    memory.record_action("scroll", "failed", "http://x", error="timeout")

    # Act / Assert
    assert memory.is_repeating_failure("click") is False


def test_record_fact_and_facts_text():
    # Arrange
    memory = TaskMemory()

    # Act
    memory.record_fact("price", "$42")

    # Assert
    assert memory.extracted_data["price"] == "$42"
    assert "price: $42" in memory.facts_text()


def test_facts_text_empty_when_no_facts_recorded():
    # Arrange
    memory = TaskMemory()

    # Act / Assert
    assert memory.facts_text() == ""


def test_goal_defaults_to_empty_and_can_be_set():
    # Arrange / Act
    memory = TaskMemory(goal="find the price")

    # Assert
    assert memory.goal == "find the price"
