from playwright.async_api import Browser, BrowserContext, Dialog, Page, Playwright
from playwright.async_api import async_playwright

from agent.types import PageState
from perception.accessibility import extract_accessibility_tree
from perception.dom_parser import extract_interactive_elements
from perception.ocr import extract_ocr_text
from perception.screenshot import capture_annotated_screenshot

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

SUPPORTED_ENGINES = ("chromium", "firefox", "webkit")


class BrowserSession:
    def __init__(self, engine: str = "chromium") -> None:
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported browser engine '{engine}'. Choose from {SUPPORTED_ENGINES}.")
        self.engine = engine
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._dialog_text: str | None = None

    async def start(self, headless: bool = False, record_video_dir: str | None = None) -> None:
        self._pw = await async_playwright().start()
        browser_type = getattr(self._pw, self.engine)
        self._browser = await browser_type.launch(headless=headless)
        context_kwargs = dict(
            viewport={"width": 1280, "height": 800},
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="Asia/Kolkata",
        )
        if record_video_dir:
            context_kwargs["record_video_dir"] = record_video_dir
        self._context = await self._browser.new_context(**context_kwargs)
        self.page = await self._context.new_page()
        self.page.on("dialog", self._on_dialog)

    def _on_dialog(self, dialog: Dialog) -> None:
        self._dialog_text = dialog.message

    async def start_tracing(self) -> None:
        """Starts a Playwright trace (screenshots + DOM snapshots + network)
        that can be opened in `npx playwright show-trace`."""
        assert self._context is not None, "call start() first"
        await self._context.tracing.start(screenshots=True, snapshots=True, sources=True)

    async def stop_tracing(self, path: str) -> None:
        assert self._context is not None, "call start() first"
        await self._context.tracing.stop(path=path)

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        self._context = None
        self.page = None

    async def get_page_state(self) -> PageState:
        assert self.page is not None, "call start() first"

        elements = await extract_interactive_elements(self.page)
        accessibility_tree = await extract_accessibility_tree(self.page)
        screenshot_b64 = await capture_annotated_screenshot(self.page, elements)
        ocr_text = await extract_ocr_text(screenshot_b64)
        scroll_y = await self.page.evaluate("window.scrollY")
        page_height = await self.page.evaluate("document.body.scrollHeight")

        return PageState(
            url=self.page.url,
            title=await self.page.title(),
            screenshot_b64=screenshot_b64,
            interactive_elements=elements,
            accessibility_tree=accessibility_tree,
            scroll_y=int(scroll_y),
            page_height=int(page_height),
            dialog_visible=self._dialog_text is not None,
            dialog_text=self._dialog_text,
            ocr_text=ocr_text,
        )


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        session = BrowserSession()
        await session.start(headless=False)
        try:
            assert session.page is not None
            await session.page.goto("https://example.com")
            state = await session.get_page_state()
            print(state.url, state.title, len(state.interactive_elements))
        finally:
            await session.stop()

    asyncio.run(_demo())
