from dataclasses import dataclass, field


@dataclass
class Element:
    id: str                 # stable short id e.g. "btn_3", "link_12", "input_0"
    tag: str                # "button", "a", "input", "select", "textarea", ...
    text: str               # visible text / label
    role: str                # accessibility role, "" if unknown
    placeholder: str | None
    href: str | None
    visible: bool
    bbox: dict               # {"x": float, "y": float, "width": float, "height": float}
    focused: bool = False    # True if this is document.activeElement


@dataclass
class PageState:
    url: str
    title: str
    screenshot_b64: str
    interactive_elements: list[Element]
    accessibility_tree: str
    scroll_y: int
    page_height: int
    dialog_visible: bool
    dialog_text: str | None
    ocr_text: str = ""       # supplementary OCR pass over the screenshot


@dataclass
class ActionResult:
    success: bool
    new_url: str
    error: str | None
    screenshot_b64: str


@dataclass
class Step:
    number: int
    url: str
    action_type: str
    action_input: dict
    action_result: ActionResult
    reasoning_text: str
    tokens_used: int


@dataclass
class AgentRun:
    task: str
    success: bool
    result: str
    steps: list[Step] = field(default_factory=list)
    total_actions: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    final_url: str = ""
