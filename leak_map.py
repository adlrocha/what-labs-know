"""Leak map: which lab received which kinds of personal data, as counts only.

Usage: python3 -I leak_map.py CHUNKS_DIR [--config config.json] [--markdown]

Reads the chunks written by chunk.py and, per lab, counts sessions, dates, megabytes by who
produced them (what you typed, what your agents read, what the model wrote) and the sessions
that touch each data category. Credentials were masked by chunk.py; they are counted here from
their `[REDACTED:kind]` placeholders. Never prints matched text.

Personal terms (your employers, places, usernames) go in a JSON config kept out of git:
  {"employers": ["Acme"], "places": ["Springfield"], "identity": ["/home/me", "myhandle"]}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CATEGORIES: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)"),
    "iban_card": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){4,7}\b|\b(?:\d{4}[\s-]){3}\d{4}\b"),
    "money": re.compile(
        r"(€\s?\d[\d.,]*|\d[\d.,]*\s?(?:€|EUR\b|k€)|£\s?\d[\d.,]*|\$\s?\d[\d.,]*k?|\b(?:portfolio|salary|"
        r"salario|nómina|invoice|factura|mortgage|hipoteca|net worth|savings|ahorros|tax return|dividend|"
        r"ETF|ISIN)\b)",
        re.I,
    ),
    "family": re.compile(
        r"\b(?:my (?:wife|husband|partner|kids?|son|daughter|children|family|parents|mother|father|mum|mom|dad)"
        r"|mi (?:mujer|esposa|marido|hij[oa]s?|familia|madre|padre)|school run|guardería|colegio)\b",
        re.I,
    ),
    "health": re.compile(
        r"\b(?:doctor|médico|hospital|sick|enfermo|diagnos\w*|symptom\w*|therapy|terapia|medication|"
        r"medicación|injury|lesión|sleep(?:ing)? (?:badly|poorly)|anxiety|ansiedad|burn ?out)\b",
        re.I,
    ),
    "career_jobsearch": re.compile(
        r"\b(?:interview(?:ing)?|job offer|recruiter|CV|resume|résumé|cover letter|quit my job|"
        r"career|compensation|equity|vesting)\b",
        re.I,
    ),
}
SECRET = re.compile(r"\[REDACTED:([a-z-]+)\]")
HEADER = re.compile(r"^# session (\S+) · project (.*)$", re.M)
SPEAKER = re.compile(r"^\[(USER|ASSISTANT|TOOL|ATTACHMENT|PROJECT)\b([^\n]*)\]$", re.M)
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
KIND = {"USER": "you_typed", "ATTACHMENT": "you_typed", "PROJECT": "you_typed", "TOOL": "agent_read"}


def load_config(path: Path) -> dict[str, list[str]]:
    """Personal terms (employers, places, usernames) from a JSON file kept out of git."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse config {path}: {e}") from e
    return {k: [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}


def term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    """Case-insensitive whole-word match for any of the terms; None when there are none."""
    if not terms:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def personal_categories(cfg: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    """Categories built from the config: location, work_employer, identity."""
    extra = {
        "location": [*cfg.get("places", []), "my address", "postal code"],
        "work_employer": [*cfg.get("employers", []), "client", "contract", "NDA"],
        "identity": cfg.get("identity", []),
    }
    return {k: rx for k, terms in extra.items() if (rx := term_pattern(terms))}


def split_messages(chunk: str) -> list[tuple[str, str, str]]:
    """(kind, date, text) for each message in a chunk; kind is you_typed, agent_read or model_wrote."""
    marks = list(SPEAKER.finditer(chunk))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(chunk)
        day = DAY.search(m.group(2))
        out.append((KIND.get(m.group(1), "model_wrote"), day.group(0) if day else "", chunk[m.end() : end]))
    return out


class Tally:
    """Per-lab aggregate counters. Holds counts and session ids, never text."""

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.projects: Counter[str] = Counter()
        self.chars: Counter[str] = Counter()
        self.sessions_with: dict[str, set[str]] = defaultdict(set)
        self.secrets: Counter[str] = Counter()
        self.first = "9999"
        self.last = "0000"

    def add(self, session: str, kind: str, text: str, day: str, cats: dict[str, re.Pattern[str]]) -> None:
        self.sessions.add(session)
        self.chars[kind] += len(text)
        if day:
            self.first, self.last = min(self.first, day), max(self.last, day)
        for cat, rx in cats.items():
            if rx.search(text):
                self.sessions_with[cat].add(session)
        for kind_ in SECRET.findall(text):
            self.secrets[kind_] += 1
            self.sessions_with["secret"].add(session)

    def summary(self) -> dict[str, Any]:
        return {
            "sessions": len(self.sessions),
            "dates": f"{self.first}..{self.last}" if self.first != "9999" else "",
            "MB": {k: round(v / 1e6, 2) for k, v in sorted(self.chars.items())},
            "top_projects": [p for p, _ in self.projects.most_common(5)],
            "sessions_with": {c: len(s) for c, s in sorted(self.sessions_with.items())},
            "secrets_masked": dict(self.secrets),
        }


def leak_map(chunks: Path, cats: dict[str, re.Pattern[str]]) -> dict[str, dict[str, Any]]:
    """Aggregate every lab directory under the chunks directory."""
    result = {}
    for lab_dir in sorted(p for p in chunks.iterdir() if p.is_dir()):
        t = Tally()
        seen: set[str] = set()
        for f in sorted(lab_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="replace")
            h = HEADER.search(text)
            session, project = (h.group(1), h.group(2)) if h else (f.stem, "?")
            if session not in seen:
                seen.add(session)
                t.projects[project] += 1
            for kind, day, body in split_messages(text):
                t.add(session, kind, body, day, cats)
        result[lab_dir.name] = t.summary()
    return result


def markdown(result: dict[str, dict[str, Any]]) -> str:
    """Two tables for a write-up: where the sessions went, and sessions touching each category."""
    rows = ["| Lab | Sessions | Dates | MB you typed | MB agents read | MB model wrote |", "|---|---|---|---|---|---|"]
    for lab, s in result.items():
        mb = s["MB"]
        rows.append(
            f"| {lab} | {s['sessions']} | {s['dates']} | {mb.get('you_typed', 0)} | {mb.get('agent_read', 0)} "
            f"| {mb.get('model_wrote', 0)} |"
        )
    cats = sorted({c for s in result.values() for c in s["sessions_with"]})
    rows += ["", "| Category | " + " | ".join(result) + " |", "|---" * (len(result) + 1) + "|"]
    rows += [f"| {c} | " + " | ".join(str(s["sessions_with"].get(c, 0)) for s in result.values()) + " |" for c in cats]
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Per-lab counts of the personal data in your chunks.")
    p.add_argument("chunks", type=Path, help="directory written by chunk.py")
    p.add_argument("--config", type=Path, default=Path("config.json"), help="personal terms (JSON)")
    p.add_argument("--markdown", action="store_true", help="print Markdown tables instead of JSON")
    args = p.parse_args(argv)
    if not args.chunks.is_dir():
        raise SystemExit(f"Not a directory: {args.chunks}")
    cats = {**CATEGORIES, **personal_categories(load_config(args.config))}
    result = leak_map(args.chunks, cats)
    print(markdown(result) if args.markdown else json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
