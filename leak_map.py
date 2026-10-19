"""First-pass leak map: which lab received which kinds of personal data.

Reads nibble captures and the opencode DB locally, attributes each message to a
lab, and counts category hits with regexes. Prints aggregates only, never matches.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

HOME = Path.home()
CAPTURE = HOME / ".nibble/memory/capture"
OPENCODE_DB = HOME / ".local/share/opencode/opencode.db"

CATEGORIES: dict[str, re.Pattern[str]] = {
    "secret": re.compile(
        r"(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
        r"|xox[abp]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY|eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}"
        r"|(?:api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-/+]{16,})",
        re.I,
    ),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+34|\+44|\+1)[\s-]?\d{2,3}[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)"),
    "iban_card": re.compile(r"\bES\d{2}[\s]?\d{4}[\s]?\d{4}|\b(?:\d{4}[\s-]){3}\d{4}\b"),
    "money": re.compile(
        r"(€\s?\d[\d.,]*|\d[\d.,]*\s?(?:€|EUR\b|k€)|\$\s?\d[\d.,]*k?|\b(?:portfolio|salary|salario|nómina|"
        r"invoice|factura|mortgage|hipoteca|net worth|savings|ahorros|IRPF|tax return|dividend|ETF|ISIN)\b)",
        re.I,
    ),
    "family": re.compile(
        r"\b(?:my (?:wife|husband|partner|kids?|son|daughter|children|family|parents|mother|father|mum|dad)"
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


def lab_for_model(model: str) -> str:
    """Map a provider/model string to the organisations that receive the traffic."""
    m = model.lower()
    if m.startswith(("ollama/", "lmstudio/")):
        return "local (own hardware)"
    if m.startswith("github-copilot/claude"):
        return "GitHub/Microsoft -> Anthropic"
    if m.startswith("github-copilot/grok"):
        return "GitHub/Microsoft -> xAI"
    if m.startswith("github-copilot/"):
        return "GitHub/Microsoft -> other"
    if m.startswith(("zai", "zai-coding-plan/")) or ("glm" in m and m.startswith("zai")):
        return "Z.ai (Zhipu, China)"
    if m.startswith("opencode/"):
        return "OpenCode Zen -> " + m.split("/", 1)[1]
    if m.startswith("kimi"):
        return "Moonshot (Kimi, China)"
    if "claude" in m or m.startswith("anthropic/"):
        return "Anthropic"
    return f"unknown ({model})"


def harness_for_capture(name: str) -> str:
    """Infer the agent harness from a nibble capture filename."""
    if name.startswith("pi-"):
        return "pi"
    if name.startswith("omp-"):
        return "omp"
    return "claude-code"


CAPTURE_LAB = {
    "claude-code": "Anthropic (assumed: Claude Code)",
    "pi": "pi (model unknown)",
    "omp": "omp (model unknown)",
}


class Tally:
    """Per-lab aggregate counters."""

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.projects: Counter[str] = Counter()
        self.chars = Counter()
        self.hits: dict[str, Counter[str]] = defaultdict(Counter)
        self.sessions_with: dict[str, set[str]] = defaultdict(set)
        self.first = "9999"
        self.last = "0000"

    def add(self, session: str, project: str, kind: str, text: str, ts: str) -> None:
        self.sessions.add(session)
        self.projects[project] += 0
        self.chars[kind] += len(text)
        if ts:
            self.first, self.last = min(self.first, ts[:10]), max(self.last, ts[:10])
        for cat, rx in CATEGORIES.items():
            n = len(rx.findall(text))
            if n:
                self.hits[cat][kind] += n
                self.sessions_with[cat].add(session)


def scan_capture(tallies: dict[str, Tally]) -> None:
    for f in CAPTURE.rglob("*.jsonl"):
        project = f.parent.name
        lab = CAPTURE_LAB[harness_for_capture(f.name)]
        t = tallies[lab]
        t.projects[project] += 1
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            role, ts = rec.get("role", ""), rec.get("ts", "")
            if role == "user":
                t.add(f.stem, project, "you_typed", str(rec.get("content", "")), ts)
            elif role == "tool":
                t.add(f.stem, project, "agent_read", str(rec.get("input", "")) + str(rec.get("output", "")), ts)
            else:
                t.add(f.stem, project, "model_wrote", str(rec.get("content", "")), ts)


def scan_opencode(tallies: dict[str, Tally]) -> None:
    if not OPENCODE_DB.exists():
        return
    con = sqlite3.connect(f"file:{OPENCODE_DB}?mode=ro", uri=True)
    sess_model: dict[str, str] = {}
    for sid, data in con.execute("select session_id, data from message"):
        d = json.loads(data)
        if d.get("providerID") and d.get("modelID"):
            sess_model.setdefault(sid, f"{d['providerID']}/{d['modelID']}")
    msg_model: dict[str, str] = {}
    msg_role: dict[str, str] = {}
    for mid, sid, data in con.execute("select id, session_id, data from message"):
        d = json.loads(data)
        msg_role[mid] = d.get("role", "")
        model = f"{d['providerID']}/{d['modelID']}" if d.get("modelID") else sess_model.get(sid, "?")
        msg_model[mid] = model
    projects = {sid: Path(directory).name for sid, directory in con.execute("select id, directory from session")}
    seen_proj: set[tuple[str, str]] = set()
    for mid, sid, ts, data in con.execute("select message_id, session_id, time_created, data from part"):
        d = json.loads(data)
        lab = "opencode: " + lab_for_model(msg_model.get(mid, "?"))
        t = tallies[lab]
        proj = projects.get(sid, "?")
        if (lab, sid) not in seen_proj:
            seen_proj.add((lab, sid))
            t.projects[proj] += 1
        day = datetime.fromtimestamp(ts / 1000, tz=UTC).isoformat()
        if d.get("type") == "text":
            kind = "you_typed" if msg_role.get(mid) == "user" else "model_wrote"
            t.add(sid, proj, kind, d.get("text", ""), day)
        elif d.get("type") == "tool":
            st = d.get("state", {})
            t.add(sid, proj, "agent_read", json.dumps(st.get("input", "")) + str(st.get("output", "")), day)


def main() -> None:
    CATEGORIES.update(personal_categories(load_config(Path(os.environ.get("WLK_CONFIG", "config.json")))))
    tallies: dict[str, Tally] = defaultdict(Tally)
    scan_capture(tallies)
    scan_opencode(tallies)
    out = {}
    for lab, t in sorted(tallies.items(), key=lambda kv: -len(kv[1].sessions)):
        out[lab] = {
            "sessions": len(t.sessions),
            "dates": f"{t.first}..{t.last}",
            "MB": {k: round(v / 1e6, 2) for k, v in t.chars.items()},
            "top_projects": t.projects.most_common(8),
            "hits": {c: dict(v) for c, v in t.hits.items()},
            "sessions_with": {c: len(s) for c, s in t.sessions_with.items()},
        }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
