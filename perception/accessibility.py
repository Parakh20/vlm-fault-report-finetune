import re

from playwright.async_api import Page

# Matches a real accessibility node line emitted by Page.aria_snapshot(), e.g.:
#   heading "Test Fixture" [level=1]
#   link "About":
#   button "Search"
#   combobox:
#   paragraph: This is sample text used by perception tests.
# Group breakdown:
#   role   -> leading token, excludes '"' and ':' so it never swallows a trailing colon
#   name   -> optional quoted accessible name
#   attrs  -> optional bracketed metadata, e.g. "level=1" or "selected"
#   inline -> optional trailing text after ":" (used as a name substitute for text nodes)
_NODE_PATTERN = re.compile(
    r'^(?P<role>[^\s":]+)'
    r'(?:\s+"(?P<name>[^"]*)")?'
    r'(?:\s+\[(?P<attrs>[^\]]*)\])?'
    r'\s*:?\s*(?P<inline>.*)$'
)

# Matches a Playwright "property" line nested under a node, e.g. "/url: /about.html".
# These aren't real accessibility nodes; they're extra metadata about the parent node.
_PROPERTY_PATTERN = re.compile(r'^/(?P<key>[^:]+):\s*(?P<value>.*)$')

_DROPPED_ROLES = ("", "generic", "none")


def _parse_bracket_attrs(raw: str | None) -> dict[str, str]:
    """Parse bracketed attribute text (e.g. "level=1" or "selected") into a dict."""
    if not raw:
        return {}
    attrs: dict[str, str] = {}
    for token in raw.split():
        token = token.strip().rstrip(",")
        if not token:
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            attrs[key] = value
        else:
            attrs[token] = "true"
    return attrs


def _parse_aria_snapshot(snapshot: str) -> list[dict]:
    """
    Parse Playwright's YAML-like aria_snapshot() output into a tree of nodes.

    Each node is a dict: {"role": str, "name": str, "extra": dict, "children": list}.
    Property lines (e.g. "/url: ...") are folded into the owning node's "extra" dict
    rather than becoming nodes of their own, since they aren't accessibility nodes.
    """
    root: dict = {"role": "", "name": "", "extra": {}, "children": []}
    # stack of (level, node) — level is the indentation depth of `node`
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in snapshot.splitlines():
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        level = indent // 2
        content = raw_line.strip()
        if content.startswith("- "):
            content = content[2:]
        elif content == "-":
            content = ""

        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else root

        property_match = _PROPERTY_PATTERN.match(content)
        if property_match:
            key = property_match.group("key").strip()
            value = property_match.group("value").strip()
            parent["extra"][key] = value
            # Property lines still occupy a level so later siblings pop correctly,
            # but they don't push a new node onto the stack.
            stack.append((level, parent))
            continue

        node_match = _NODE_PATTERN.match(content)
        if not node_match:
            continue

        role = node_match.group("role") or ""
        name = node_match.group("name") or ""
        attrs = _parse_bracket_attrs(node_match.group("attrs"))
        inline = (node_match.group("inline") or "").strip()
        if not name and inline:
            name = inline

        node = {"role": role, "name": name, "extra": attrs, "children": []}
        parent["children"].append(node)
        stack.append((level, node))

    return root["children"]


def _format_node(node: dict, depth: int) -> list[str]:
    role = node.get("role") or ""
    name = (node.get("name") or "").strip()
    extra = node.get("extra") or {}
    children = node.get("children") or []

    lines: list[str] = []
    if role not in _DROPPED_ROLES and (name or children or extra):
        indent = "  " * depth
        label = f'{indent}[{role}]'
        if name:
            label += f' "{name}"'
        for key, value in extra.items():
            label += f' {key}="{value}"'
        lines.append(label)
        depth += 1

    for child in children:
        lines.extend(_format_node(child, depth))

    return lines


async def extract_accessibility_tree(page: Page) -> str:
    """
    Extract accessibility tree from the page.

    Playwright's async API no longer exposes the legacy `page.accessibility`
    snapshot API, so this uses `page.aria_snapshot()` (which is still
    available) and reformats its YAML-like output into the contracted
    `[role] "name" extra="value"` indented text-tree shape. Invisible/empty
    nodes are dropped: `aria_snapshot()` already excludes hidden elements,
    and nodes with no name, no children, and no extra attributes (e.g. bare
    "generic" wrappers) are filtered out during formatting.
    """
    snapshot = await page.aria_snapshot()
    if not snapshot:
        return ""
    nodes = _parse_aria_snapshot(snapshot)
    lines: list[str] = []
    for node in nodes:
        lines.extend(_format_node(node, 0))
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    from playwright.async_api import async_playwright

    async def _demo():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            print(await extract_accessibility_tree(page))
            await browser.close()

    asyncio.run(_demo())
