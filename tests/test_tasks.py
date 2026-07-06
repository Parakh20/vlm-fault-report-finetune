from evaluation.tasks import BENCHMARK_TASKS

EXPECTED_CATEGORIES = {
    "information_extraction",
    "multi_step_navigation",
    "form_interaction",
    "data_aggregation",
    "reasoning_over_content",
}


def test_exactly_25_tasks_defined():
    assert len(BENCHMARK_TASKS) == 25


def test_each_category_has_exactly_5_tasks():
    counts = {}
    for t in BENCHMARK_TASKS:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert set(counts.keys()) == EXPECTED_CATEGORIES
    assert all(v == 5 for v in counts.values())


def test_every_task_has_unique_id_and_nonempty_prompt():
    ids = [t["id"] for t in BENCHMARK_TASKS]
    assert len(ids) == len(set(ids))
    assert all(t["prompt"].strip() for t in BENCHMARK_TASKS)
