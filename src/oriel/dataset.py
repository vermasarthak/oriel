from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    input_text: str
    required: dict[str, object]


def load_jsonl(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = Case(id=raw["id"], input_text=raw["input"], required=raw["required"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid case at line {line_number}") from error
        if not case.id or not case.input_text or not isinstance(case.required, dict):
            raise ValueError(f"invalid case at line {line_number}")
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("dataset has no cases")
    return cases
