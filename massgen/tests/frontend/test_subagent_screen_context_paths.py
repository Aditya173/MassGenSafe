"""Tests for subagent header context path links."""

from textual.app import App, ComposeResult
from textual.widgets import Button

from massgen.frontend.displays.textual_widgets.subagent_screen import SubagentHeader
from massgen.subagent.models import SubagentDisplayData


def _make_subagent(subagent_id: str) -> SubagentDisplayData:
    return SubagentDisplayData(
        id=subagent_id,
        task="check output",
        status="running",
        progress_percent=0,
        elapsed_seconds=0.0,
        timeout_seconds=300.0,
        workspace_path="",
        workspace_file_count=0,
        last_log_line="",
        error=None,
        answer_preview=None,
        log_path=None,
    )


class _HeaderHarness(App[None]):
    def __init__(self, subagent: SubagentDisplayData) -> None:
        super().__init__()
        self._subagent = subagent
        self.clicked_paths: list[str] = []

    def compose(self) -> ComposeResult:
        yield SubagentHeader(self._subagent, id="subagent-header")

    def on_subagent_header_context_path_clicked(self, event: SubagentHeader.ContextPathClicked) -> None:
        self.clicked_paths.append(event.path)


async def test_subagent_header_context_paths_are_clickable() -> None:
    subagent = _make_subagent("sub_1")
    subagent.context_paths = ["docs/brief.md", "src/components"]
    app = _HeaderHarness(subagent)

    async with app.run_test(headless=True, size=(140, 24)) as pilot:
        context_buttons = [button for button in app.query(Button) if button.has_class("context-path-btn")]
        assert len(context_buttons) == 2

        # Generation counter starts at 1 after first mount
        await pilot.click(f"#{context_buttons[0].id}")
        assert app.clicked_paths == ["docs/brief.md"]
