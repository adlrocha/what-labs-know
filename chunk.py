"""Build per-lab text chunks from nibble captures and the opencode DB.

Usage: python3 -I chunk.py OUT_DIR [--claude-export RAW_DIR]
Writes OUT_DIR/<lab-slug>/<session>-<part>.txt and OUT_DIR/manifest.json (counts only).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
CAPTURE = HOME / ".nibble/memory/capture"
OPENCODE_DB = HOME / ".local/share/opencode/opencode.db"
TOOL_CAP = 2000
CHUNK_CHARS = 150_000

Msg = tuple[str, str, str]  # (timestamp, speaker, text)


def lab_for_model(model: str) -> str:
    """Map an opencode provider/model string to a lab slug."""
    m = model.lower()
    if m.startswith(("ollama/", "lmstudio/")):
        return "local"
    if m.startswith("github-copilot/claude"):
        return "copilot-anthropic"
    if m.startswith("github-copilot/"):
        return "copilot-other"
    if m.startswith("zai"):
        return "zai"
    if m.startswith("kimi"):
        return "moonshot"
    if m.startswith("opencode/"):
        return "opencode-zen"
    return "unknown"


def capture_lab(name: str) -> str:
    """Lab slug for a nibble capture file, from its filename prefix."""
    if name.startswith("pi-"):
        return "pi-unknown"
    if name.startswith("omp-"):
        return "omp-unknown"
    return "anthropic-claude-code"


SECRETS: list[tuple[str, re.Pattern[str]]] = [
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("api-key", re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("slack-token", re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)((?:api[_-]?key|secret|token|password|passwd|bearer)['\"]?\s*[=:]\s*['\"]?)[A-Za-z0-9_\-/+.=]{12,}"
        ),
    ),
]


def redact(text: str) -> str:
    """Mask credentials so they never leave this machine; keep a typed placeholder."""
    for kind, rx in SECRETS:
        if kind == "assigned-secret":
            text = rx.sub(lambda m: m.group(1) + "[REDACTED:secret]", text)
        else:
            text = rx.sub(f"[REDACTED:{kind}]", text)
    return text


def cap(text: str) -> str:
    return text if len(text) <= TOOL_CAP else text[:TOOL_CAP] + f" …[{len(text) - TOOL_CAP} chars cut]"


def from_capture() -> dict[tuple[str, str], tuple[str, list[Msg]]]:
    sessions: dict[tuple[str, str], tuple[str, list[Msg]]] = {}
    for f in sorted(CAPTURE.rglob("*.jsonl")):
        if f.parent.name in {"test-project"}:
            continue
        msgs: list[Msg] = []
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts, role = str(r.get("ts", ""))[:19], r.get("role", "")
            if role == "tool":
                body = f"input: {cap(str(r.get('input', '')))}\noutput: {cap(str(r.get('output', '')))}"
                msgs.append((ts, f"TOOL {r.get('name', '')}", body))
            elif role in {"user", "assistant"}:
                msgs.append((ts, role.upper(), str(r.get("content", ""))))
        if msgs:
            sessions[(capture_lab(f.name), f.stem)] = (f.parent.name, msgs)
    return sessions


def from_opencode() -> dict[tuple[str, str], tuple[str, list[Msg]]]:
    if not OPENCODE_DB.exists():
        return {}
    con = sqlite3.connect(f"file:{OPENCODE_DB}?mode=ro", uri=True)
    sess_model: dict[str, str] = {}
    msg_info: dict[str, tuple[str, str]] = {}
    rows = list(con.execute("select id, session_id, data from message order by time_created"))
    for _, sid, data in rows:
        d = json.loads(data)
        if d.get("modelID"):
            sess_model.setdefault(sid, f"{d.get('providerID')}/{d['modelID']}")
    for mid, sid, data in rows:
        d = json.loads(data)
        model = f"{d.get('providerID')}/{d['modelID']}" if d.get("modelID") else sess_model.get(sid, "?")
        msg_info[mid] = (d.get("role", ""), model)
    project = {sid: Path(dr).name for sid, dr in con.execute("select id, directory from session")}
    grouped: dict[tuple[str, str], list[Msg]] = defaultdict(list)
    parts = con.execute("select message_id, session_id, time_created, data from part order by time_created")
    for mid, sid, ts, data in parts:
        d = json.loads(data)
        role, model = msg_info.get(mid, ("", "?"))
        key = (lab_for_model(model), sid)
        stamp = str(ts)
        if d.get("type") == "text":
            grouped[key].append((stamp, role.upper(), d.get("text", "")))
        elif d.get("type") == "tool":
            st = d.get("state", {})
            body = f"input: {cap(json.dumps(st.get('input', '')))}\noutput: {cap(str(st.get('output', '')))}"
            grouped[key].append((stamp, f"TOOL {d.get('tool', '')}", body))
    return {k: (project.get(k[1], "?"), v) for k, v in grouped.items()}


def write_chunks(out: Path, sessions: dict[tuple[str, str], tuple[str, list[Msg]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for (lab, sid), (proj, msgs) in sessions.items():
        d = out / lab
        d.mkdir(parents=True, exist_ok=True)
        header = f"# session {sid} · project {proj}\n"
        parts: list[str] = []
        buf = header
        for ts, who, text in msgs:
            block = f"\n[{who} {ts}]\n{text}\n"
            if len(buf) + len(block) > CHUNK_CHARS and buf != header:
                parts.append(buf)
                buf = header
            buf += block[: CHUNK_CHARS - len(header)]
        parts.append(buf)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)
        for i, p in enumerate(parts):
            (d / f"{safe}-{i:03d}.txt").write_text(redact(p), encoding="utf-8")
        counts[lab] += len(parts)
    return counts


def from_claude_export(raw: Path) -> dict[tuple[str, str], tuple[str, list[Msg]]]:
    """claude.ai export: conversations (text, attachments, tool calls) and project docs."""
    lab = "anthropic-claude-ai"
    sessions: dict[tuple[str, str], tuple[str, list[Msg]]] = {}
    for conv in json.loads((raw / "conversations.json").read_text(encoding="utf-8")):
        msgs: list[Msg] = []
        for m in conv.get("chat_messages", []):
            ts, who = str(m.get("created_at", ""))[:19], "USER" if m.get("sender") == "human" else "ASSISTANT"
            texts = [c.get("text", "") for c in m.get("content", []) if c.get("type") == "text"]
            msgs.append((ts, who, "\n".join(t for t in texts if t) or str(m.get("text", ""))))
            for a in m.get("attachments", []):
                msgs.append((ts, f"ATTACHMENT {a.get('file_name', '')}", cap(str(a.get("extracted_content", "")))))
            for c in m.get("content", []):
                if c.get("type") == "tool_use":
                    msgs.append((ts, f"TOOL {c.get('name', '')}", cap(json.dumps(c.get("input", "")))))
        if msgs:
            sessions[(lab, conv["uuid"])] = (f"claude.ai: {conv.get('name', '')}", msgs)
    for p in sorted((raw / "projects").glob("*.json")):
        proj = json.loads(p.read_text(encoding="utf-8"))
        msgs = [("", "PROJECT INSTRUCTIONS", str(proj.get("prompt_template", "")))]
        msgs += [
            ("", f"PROJECT DOC {d.get('filename', '')}", cap(str(d.get("content", "")))) for d in proj.get("docs", [])
        ]
        sessions[(lab, f"project-{proj['uuid']}")] = (f"claude.ai project: {proj.get('name', '')}", msgs)
    return sessions


def main() -> None:
    out = Path(sys.argv[1])
    if len(sys.argv) > 3 and sys.argv[2] == "--claude-export":
        sessions = from_claude_export(Path(sys.argv[3]))
    else:
        sessions = {**from_capture(), **from_opencode()}
    counts = write_chunks(out, sessions)
    chars = {lab: sum(p.stat().st_size for p in (out / lab).glob("*.txt")) for lab in counts}
    manifest = {"chunks": dict(counts), "MB": {k: round(v / 1e6, 1) for k, v in chars.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
