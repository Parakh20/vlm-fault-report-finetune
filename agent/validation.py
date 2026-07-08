from dataclasses import dataclass

from agent.actions import ACTION_SCHEMAS
from agent.types import PageState

_SCHEMA_BY_NAME = {schema["name"]: schema for schema in ACTION_SCHEMAS}
_ELEMENT_ID_ACTIONS = {"click", "type_text", "select_option", "hover"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None


def validate_action(action_name: str, action_args: dict, state: PageState) -> ValidationResult:
    """Checks a proposed action against the schema and the current page
    state before it reaches the browser, so a hallucinated element_id or a
    missing argument becomes a cheap repair note instead of a wasted round
    trip (or worse, a Playwright exception mid-loop)."""
    schema = _SCHEMA_BY_NAME.get(action_name)
    if schema is None:
        return ValidationResult(False, f"Unknown action '{action_name}'.")

    required = schema.get("parameters", {}).get("required", [])
    missing = [key for key in required if key not in action_args]
    if missing:
        return ValidationResult(
            False, f"Missing required argument(s) {missing} for '{action_name}'."
        )

    if action_name in _ELEMENT_ID_ACTIONS:
        element_id = action_args.get("element_id")
        known_ids = sorted(element.id for element in state.interactive_elements)
        if element_id not in known_ids:
            return ValidationResult(
                False,
                f"element_id '{element_id}' does not exist on the current page. "
                f"Known ids: {known_ids[:20]}",
            )

    return ValidationResult(True)
