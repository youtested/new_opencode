"""Question dialog: the model asks the user structured questions (opencode's).

Mirrors `packages/tui/src/routes/session/question.tsx`:

- A left-bordered accent panel with a tab strip (one tab per question + a
  Confirm tab when there are several / multi-select).
- Each question lists numbered options with a description under each, plus a
  "Type your own answer" row when `custom` is enabled.
- Keyboard: ↑/↓/j/k select, 1-9 pick, enter select/confirm/submit,
  Tab/h/l switch question, Esc dismiss (rejects the whole ask).
- Dismissing with the last result delivers ``answers`` (list of list[str], one
  per question); Esc/Close yields ``None`` (the ask is rejected).
"""

from __future__ import annotations

from typing import Any, Callable

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..question import QuestionInfo
from .theme import active_theme

DIALOG_CSS = """
QuestionDialog {
    background: rgba(0, 0, 0, 0.6);
    layout: vertical;
}
QuestionDialog #question-dialog {
    width: 88;
    max-width: 92%;
    max-height: 90%;
    height: auto;
    padding: 1 3 1 2;
    border: round $accent;
    background: $panel;
}
QuestionDialog #question-tabs {
    height: auto;
    padding: 0 1;
    margin-bottom: 1;
    text-style: bold;
}
QuestionDialog #question-body {
    height: auto;
    padding: 0 1;
}
QuestionDialog #question-footer {
    height: auto;
    padding: 1 1 0 1;
    color: $text-muted;
}
QuestionDialog #question-input {
    margin: 0 1;
}
"""


class QuestionDialog(ModalScreen[list[list[str]] | None]):
    """Modal asking the user to answer one or more questions.

    Yields ``answers`` (list of list[str]) on submit, ``None`` on dismiss.
    """

    CSS = DIALOG_CSS

    can_focus = True

    BINDINGS = [
        Binding("escape", "dismiss_q", "Dismiss"),
        Binding("tab", "next_tab", "Next"),
    ]

    def __init__(
        self,
        questions: list[QuestionInfo],
        on_done: Callable[[list[list[str]] | None], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.questions = questions
        self.on_done = on_done
        self._tab = 0  # index into questions; len(questions) == confirm
        self._selected = 0  # option row within the current question
        self._answers: list[list[str]] = [[] for _ in questions]
        self._custom: list[str] = ["" for _ in questions]
        self._editing = False

    @property
    def single(self) -> bool:
        return len(self.questions) == 1 and not self.questions[0].multiple

    @property
    def tabs(self) -> int:
        return 1 if self.single else len(self.questions) + 1

    @property
    def confirm_tab(self) -> bool:
        return not self.single and self._tab == len(self.questions)

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog"):
            yield Static("", id="question-tabs")
            yield Static("", id="question-body")
            yield Input(placeholder="Type your own answer", id="question-input")
            yield Static("", id="question-footer")

    def on_mount(self) -> None:
        self._redraw()
        if self._editing:
            self.query_one("#question-input", Input).focus()
        else:
            self.set_focus(self)

    # -- rendering ---------------------------------------------------------
    def _current(self) -> QuestionInfo | None:
        if self.confirm_tab:
            return None
        return self.questions[self._tab]

    def _redraw(self) -> None:
        from rich.console import Group, RenderableType

        theme = active_theme()
        accent = theme.c("accent")
        tabs = self.query_one("#question-tabs", Static)
        body = self.query_one("#question-body", Static)
        footer = self.query_one("#question-footer", Static)
        inp = self.query_one("#question-input", Input)

        # tabs strip (mirrors official header row + Confirm tab)
        tab_text = Text()
        for i, q in enumerate(self.questions):
            active = not self.confirm_tab and self._tab == i
            answered = bool(self._answers[i])
            style = (
                f"bold {accent}"
                if active
                else theme.c("text")
                if answered
                else theme.c("text_muted")
            )
            tab_text.append(f" {q.header} ", style=style)
        if not self.single:
            style = f"bold {accent}" if self.confirm_tab else theme.c("text_muted")
            tab_text.append(" Confirm ", style=style)
        tabs.update(tab_text)

        if self.confirm_tab:
            lines: list[RenderableType] = [Text("Review", style=theme.c("text"))]
            for i, q in enumerate(self.questions):
                value = ", ".join(self._answers[i]) if self._answers[i] else ""
                answered = bool(value)
                row = Text()
                row.append(f"{q.header}: ", style=theme.c("text_muted"))
                row.append(value if answered else "(not answered)", style=theme.c("text") if answered else theme.c("error"))
                lines.append(row)
            body.update(Group(*lines))
        else:
            q = self._current()
            lines = [Text(q.question + (" (select all that apply)" if q.multiple else ""), style=theme.c("text"))]
            row = 0
            for opt in q.options:
                picked = opt.label in self._answers[self._tab]
                active = row == self._selected
                marker = f"[{'✓' if picked else ' '}] " if q.multiple else ""
                suffix = " ✓" if (not q.multiple and picked) else ""
                label = marker + opt.label + suffix
                style = f"bold {accent}" if active else theme.c("text")
                lines.append(Text(f"{row + 1}. ", style=theme.c("text_muted")) + Text(label, style=style))
                if opt.description:
                    desc_style = f"dim {accent}" if active else "dim"
                    lines.append(Text(f"   {opt.description}", style=desc_style))
                row += 1
            if q.custom:
                picked = self._custom[self._tab] and self._custom[self._tab] in self._answers[self._tab]
                active = row == self._selected
                label = f"[{'✓' if picked else ' '}] " if q.multiple else ""
                label += "Type your own answer"
                style = f"bold {accent}" if active else theme.c("text")
                lines.append(Text(f"{row + 1}. ", style=theme.c("text_muted")) + Text(label, style=style))
                if self._custom[self._tab]:
                    lines.append(Text(f"   {self._custom[self._tab]}", style="dim"))
            body.update(Group(*lines))

        # footer hints
        hints = Text()
        if not self.single:
            hints.append("⇆", style=theme.c("text"))
            hints.append(" tab  ", style=theme.c("text_muted"))
        if not self.confirm_tab:
            hints.append("↑↓", style=theme.c("text"))
            hints.append(" select  ", style=theme.c("text_muted"))
        hints.append("enter", style=theme.c("text"))
        q = self._current()
        action = "submit" if (self.confirm_tab or self.single) else "toggle" if (q and q.multiple) else "confirm"
        hints.append(f" {action}  ", style=theme.c("text_muted"))
        hints.append("esc", style=theme.c("text"))
        hints.append(" dismiss", style=theme.c("text_muted"))
        footer.update(hints)

        # custom-answer text input
        editing = self._editing and not self.confirm_tab
        inp.display = editing
        if editing:
            inp.value = self._custom[self._tab]
        else:
            inp.value = ""

    # -- navigation --------------------------------------------------------
    def _rows(self) -> int:
        q = self._current()
        if q is None:
            return 0
        return len(q.options) + (1 if q.custom else 0)

    def _move(self, delta: int) -> None:
        total = max(1, self._rows())
        self._selected = (self._selected + delta) % total
        self._redraw()

    def _select(self) -> None:
        q = self._current()
        if q is None:
            return
        row = self._selected
        is_custom = q.custom and row >= len(q.options)
        if is_custom:
            if q.multiple:
                text = self._custom[self._tab]
                if text:
                    if text in self._answers[self._tab]:
                        self._answers[self._tab] = [a for a in self._answers[self._tab] if a != text]
                    else:
                        self._answers[self._tab].append(text)
                    self._redraw()
                else:
                    self._editing = True
                    self._redraw()
                    self.query_one("#question-input", Input).focus()
                return
            self._editing = True
            self._redraw()
            self.query_one("#question-input", Input).focus()
            return
        opt = q.options[row]
        if q.multiple:
            if opt.label in self._answers[self._tab]:
                self._answers[self._tab] = [a for a in self._answers[self._tab] if a != opt.label]
            else:
                self._answers[self._tab].append(opt.label)
            self._redraw()
            return
        self._answers[self._tab] = [opt.label]
        if self.single:
            self._done()
            return
        self._tab += 1
        self._selected = 0
        self._redraw()

    def _done(self) -> None:
        result = self._answers
        if self.on_done:
            self.on_done(result)
        self.dismiss(result)

    def _dismiss_q(self) -> None:
        if self._editing:
            self._editing = False
            self._redraw()
            self.focus()
            return
        if self.on_done:
            self.on_done(None)
        self.dismiss(None)

    def _next_tab(self, shift: bool = False) -> None:
        if self._editing:
            return
        delta = -1 if shift else 1
        self._tab = (self._tab + delta) % self.tabs
        self._selected = 0
        self._redraw()

    # -- events ------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        q = self._current()
        if q is None:
            return
        if text:
            self._custom[self._tab] = text
            if q.multiple:
                if text not in self._answers[self._tab]:
                    self._answers[self._tab].append(text)
            else:
                self._answers[self._tab] = [text]
        self._editing = False
        self._redraw()
        if self.single and self._answers[self._tab]:
            self._done()
        else:
            self.focus()

    def action_dismiss_q(self) -> None:
        self._dismiss_q()

    def action_next_tab(self) -> None:
        self._next_tab()

    def on_key(self, event: Key) -> None:
        if self._editing:
            if event.key == "escape":
                self._editing = False
                self._redraw()
                self.focus()
                event.stop()
            return
        key = event.key
        if key in ("up", "k"):
            self._move(-1)
            event.stop()
        elif key in ("down", "j"):
            self._move(1)
            event.stop()
        elif key in ("left", "h"):
            self._next_tab(shift=True)
            event.stop()
        elif key in ("right", "l"):
            self._next_tab()
            event.stop()
        elif key == "tab":
            self._next_tab()
            event.stop()
        elif key == "enter":
            if self.confirm_tab:
                self._done()
            else:
                self._select()
            event.stop()
        elif key in "123456789":
            idx = int(key) - 1
            if idx < self._rows():
                self._selected = idx
                self._select()
            event.stop()
        elif key == "escape":
            # Same double-ESC contract as the permission dialog: an armed
            # 1st ESC makes this the force-stop press (the app binding never
            # fires while a modal has focus).
            try:
                app = getattr(self, "app", None)
                if app is not None and getattr(app, "_esc_presses", 0) >= 1:
                    app._force_stop_all(pop_modal=False)
                    try:
                        app._disarm_interrupt_escape()
                    except Exception:
                        pass
            except Exception:
                pass
            self._dismiss_q()
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "question-submit":
            self._done()
        elif event.button.id == "question-close":
            self._dismiss_q()
