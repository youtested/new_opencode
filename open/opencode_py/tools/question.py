"""Question tool: the model asks the user structured questions (opencode's).

RULES (full guidance for maintainers; the sent schema is short):
- Ask when ambiguous or weighing tradeoffs; then go autonomously.
- `custom` on (default) adds "Type your own answer" — never add an
  "Other" catch-all option yourself.
- Answers come back as arrays of labels; `multiple: true` allows several.
- Put your recommended option first with "(Recommended)" in the label.
- Mirrors upstream question.txt + question.v2 schema. run() resolves the
  asker lazily from the Registry (works headless: no asker -> dismissed).
"""

from __future__ import annotations

from .registry import Registry, Tool, schema_with


def tool(registry: Registry) -> Tool:
    def run(arguments: dict) -> dict:
        asker = getattr(registry, "question_asker", None)
        if asker is None:
            return {
                "output": "question tool is unavailable here (no UI to ask the user).",
                "error": True,
                "denied": True,
            }
        questions = arguments.get("questions") or []
        if not isinstance(questions, list) or not questions:
            return {"output": "question: no questions provided.", "error": True}
        from ..question import parse_questions

        parsed = parse_questions(questions)
        if not parsed:
            return {"output": "question: questions must have text.", "error": True}
        try:
            answers = asker(parsed)
        except Exception as e:
            return {
                "title": f"Asked {len(parsed)} question{'s' if len(parsed) != 1 else ''}",
                "output": f"The user dismissed this question: {e}",
                "error": True,
                "denied": True,
            }
        formatted = [
            f'"{q.question}"="{"Unanswered" if not a else ", ".join(a or [])}"'
            for q, a in zip(parsed, answers)
        ]
        return {
            "title": f"Asked {len(parsed)} question{'s' if len(parsed) != 1 else ''}",
            "output": (
                "User has answered your questions: "
                + ", ".join(formatted)
                + ". You can now continue with the user's answers in mind."
            ),
            "metadata": {"answers": [list(a) for a in answers]},
        }

    return Tool(
        name="question",
        description=(
            "Ask the user questions mid-task (preferences, ambiguity, "
            "decisions). Recommended option first with (Recommended); "
            "custom answers auto-included, never add Other."
        ),
        parameters=schema_with(
            {
                "questions": {
                    "type": "array",
                    "description": "Questions (question + options list)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question",
                            },
                            "header": {
                                "type": "string",
                                "description": "Short label, max 30 chars",
                            },
                            "options": {
                                "type": "array",
                                "description": "Choices (label + description each)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Choice text, 1-5 words",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Why this choice",
                                        },
                                    },
                                    "required": ["label"],
                                },
                            },
                            "multiple": {
                                "type": "boolean",
                                "description": "Allow several picks",
                            },
                            "custom": {
                                "type": "boolean",
                                "description": "Allow typed answer (default true)",
                            },
                        },
                        "required": ["question", "options"],
                    },
                }
            },
            required=["questions"],
        ),
        run=run,
    )
