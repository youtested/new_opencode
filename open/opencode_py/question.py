"""Question service: the model asks the user structured questions.

Mirrors upstream opencode `packages/core/src/question.ts`: the `question` tool
registers a pending request, the TUI surfaces a modal, and the user's answers
(or a rejection) unblock the engine thread that called `ask`. When no UI is
attached (headless), `ask` rejects — the model sees "user dismissed this".

Structures mirror `packages/schema/src/question.ts`:

- ``QuestionOption``  -> `QuestionV2.Option`  ({label, description})
- ``QuestionInfo``    -> `QuestionV2.Info`    ({question, header, options,
  multiple, custom})
- ``QuestionRejectedError`` -> `QuestionV2.RejectedError`
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

_question_ids = itertools.count(1)


def _new_id() -> str:
    return f"que_{next(_question_ids)}"


class QuestionRejectedError(Exception):
    """Raised when the user dismisses a question without answering."""

    def __init__(self, message: str = "The user dismissed this question") -> None:
        super().__init__(message)


@dataclass
class QuestionOption:
    """A selectable choice: display text + short explanation."""

    label: str
    description: str = ""


@dataclass
class QuestionInfo:
    """One question for the user (mirrors `QuestionV2.Info`)."""

    question: str
    header: str
    options: list[QuestionOption] = field(default_factory=list)
    multiple: bool = False
    custom: bool = True


@dataclass
class QuestionRequest:
    """A pending ask: id + the questions to surface."""

    id: str
    questions: list[QuestionInfo]


def parse_questions(raw: Any) -> list[QuestionInfo]:
    """Normalize the tool's `questions` argument into QuestionInfo objects."""
    if not isinstance(raw, list):
        return []
    out: list[QuestionInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        header = str(item.get("header", "")).strip() or question[:30]
        options = []
        for opt in item.get("options") or []:
            if isinstance(opt, dict):
                options.append(
                    QuestionOption(
                        label=str(opt.get("label", "")).strip(),
                        description=str(opt.get("description", "")).strip(),
                    )
                )
            elif isinstance(opt, str) and opt.strip():
                options.append(QuestionOption(label=opt.strip()))
        options = [o for o in options if o.label]
        out.append(
            QuestionInfo(
                question=question,
                header=header[:30],
                options=options,
                multiple=bool(item.get("multiple")),
                custom=item.get("custom", True),
            )
        )
    return out


class QuestionService:
    """Owns pending question requests and bridges to the UI.

    The engine thread blocks in `ask` until the user answers or dismisses.
    The TUI sets `ask_callback` (same pattern as `PermissionEngine`); the
    callback runs on the UI thread and must return the answers (a list of
    list[str], one per question) or ``None`` to reject.
    """

    def __init__(
        self,
        ask_callback: Callable[[list[QuestionInfo]], list[list[str]] | None] | None = None,
    ) -> None:
        self.ask_callback = ask_callback
        self._pending: dict[str, QuestionRequest] = {}
        self._lock = threading.Lock()

    def ask(self, questions: list[QuestionInfo]) -> list[list[str]]:
        """Block until the user answers; raises QuestionRejectedError on dismiss.

        Mirrors `QuestionV2.ask`: register the request, publish to the UI via
        the callback, then return the answers (or raise on rejection).
        """
        if not questions:
            return []
        request = QuestionRequest(id=_new_id(), questions=questions)
        with self._lock:
            self._pending[request.id] = request
        try:
            if self.ask_callback is None:
                # headless: no UI to answer — treat as dismissed
                raise QuestionRejectedError()
            answers = self.ask_callback(questions)
            if answers is None:
                raise QuestionRejectedError()
            return answers
        finally:
            with self._lock:
                self._pending.pop(request.id, None)

    def list_pending(self) -> list[QuestionRequest]:
        with self._lock:
            return list(self._pending.values())
