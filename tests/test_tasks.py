from evaluation.tasks import BENCHMARK_TASKS

ORIGINAL_CATEGORIES = {
    "information_extraction",
    "multi_step_navigation",
    "form_interaction",
    "data_aggregation",
    "reasoning_over_content",
}

NEW_CATEGORIES = {
    "shopping",
    "wikipedia",
    "tables",
    "downloads",
    "authentication",
    "pagination",
    "infinite_scroll",
    "captcha_detection",
    "dynamic_content",
    "file_uploads",
}

EXPECTED_CATEGORIES = ORIGINAL_CATEGORIES | NEW_CATEGORIES


def test_covers_at_least_14_categories():
    categories = {t["category"] for t in BENCHMARK_TASKS}
    assert categories == EXPECTED_CATEGORIES
    assert len(categories) >= 14


def test_original_categories_still_have_5_tasks_each():
    counts = {}
    for t in BENCHMARK_TASKS:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert all(counts[cat] == 5 for cat in ORIGINAL_CATEGORIES)


def test_new_categories_have_at_least_one_task_each():
    counts = {}
    for t in BENCHMARK_TASKS:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert all(counts.get(cat, 0) >= 1 for cat in NEW_CATEGORIES)


def test_captcha_category_instructs_detection_not_bypass():
    captcha_tasks = [t for t in BENCHMARK_TASKS if t["category"] == "captcha_detection"]
    assert captcha_tasks
    for t in captcha_tasks:
        assert "bypass" in t["prompt"].lower() or "cannot" in t["prompt"].lower()


def test_every_task_has_unique_id_and_nonempty_prompt():
    ids = [t["id"] for t in BENCHMARK_TASKS]
    assert len(ids) == len(set(ids))
    assert all(t["prompt"].strip() for t in BENCHMARK_TASKS)
