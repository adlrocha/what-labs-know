"""Build per-lab text chunks from chat exports and agent transcripts, with credentials masked.

Usage:
  python3 -I chunk.py OUT_DIR [--claude-export DIR] [--chatgpt-export DIR] [--claude-code DIR]
                              [--opencode DB] [--nibble DIR]

Each source is optional; pass at least one. Writes OUT_DIR/<lab>/<session>-<part>.txt and
OUT_DIR/manifest.json (counts only). Every chunk starts with `# session <id> · project <name>`,
and every message with a `[SPEAKER timestamp]` line, where SPEAKER is USER, ASSISTANT, TOOL,
ATTACHMENT or PROJECT.

Env: WLK_CHUNK_CHARS (default 150000, about 40k tokens; use roughly 3x your model's context
length in tokens) and WLK_TOOL_CAP (characters kept per tool output, default 2000).

Example:
  python3 -I chunk.py data/chunks --claude-export ~/Downloads/claude-export/raw \\
      --claude-code ~/.claude/projects
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOOL_CAP = int(os.environ.get("WLK_TOOL_CAP", "2000"))
# ~4 characters per token: 150k characters needs a model context of about 48k tokens.
CHUNK_CHARS = int(os.environ.get("WLK_CHUNK_CHARS", "150000"))

Msg = tuple[str, str, str]  # (timestamp, speaker, text)
Sessions = dict[tuple[str, str], tuple[str, list[Msg]]]  # (lab, session id) -> (project, messages)


def lab_for_model(model: str) -> str:
    """Map a provider/model string (opencode style, or a bare model id) to a lab slug."""
    m = model.lower()
    if m.startswith(("ollama/", "lmstudio/")):
        return "local"
    if m.startswith("github-copilot/claude"):
        return "copilot-anthropic"
    if m.startswith("github-copilot/"):
        return "copilot-other"
    if m.startswith(("zai", "glm")):
        return "zai"
    if m.startswith("kimi"):
        return "moonshot"
    if m.startswith("opencode/"):
        return "opencode-zen"
    if m.startswith(("anthropic/", "claude")):
        return "anthropic"
    if m.startswith(("openai/", "gpt", "o1", "o3", "o4")):
        return "openai"
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
    """Cut long tool output and attachments to TOOL_CAP characters, saying how much was cut."""
    return text if len(text) <= TOOL_CAP else text[:TOOL_CAP] + f" …[{len(text) - TOOL_CAP} chars cut]"


def iso(ts: object) -> str:
    """Normalise epoch seconds or milliseconds, or an ISO string, to `YYYY-MM-DDTHH:MM:SS`."""
    if isinstance(ts, int | float):
        secs = ts / 1000 if ts > 1e11 else ts
        return datetime.fromtimestamp(secs, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return str(ts or "")[:19]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping lines that don't parse."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"Failed to read {path}: {e}") from e


def from_nibble(capture: Path) -> Sessions:
    """nibble captures: <capture>/<project>/<session>.jsonl with role/content/name/input/output."""
    sessions: Sessions = {}
    for f in sorted(capture.rglob("*.jsonl")):
        if f.parent.name == "test-project":
            continue
        msgs: list[Msg] = []
        for r in read_jsonl(f):
            ts, role = iso(r.get("ts")), r.get("role", "")
            if role == "tool":
                body = f"input: {cap(str(r.get('input', '')))}\noutput: {cap(str(r.get('output', '')))}"
                msgs.append((ts, f"TOOL {r.get('name', '')}", body))
            elif role in {"user", "assistant"}:
                msgs.append((ts, role.upper(), str(r.get("content", ""))))
        if msgs:
            sessions[(capture_lab(f.name), f.stem)] = (f.parent.name, msgs)
    return sessions


def from_opencode(db: Path) -> Sessions:
    """opencode SQLite DB: model per message, text and tool parts."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = list(con.execute("select id, session_id, data from message order by time_created"))
    sess_model: dict[str, str] = {}
    for _, sid, data in rows:
        d = json.loads(data)
        if d.get("modelID"):
            sess_model.setdefault(sid, f"{d.get('providerID')}/{d['modelID']}")
    msg_info: dict[str, tuple[str, str]] = {}
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
        if d.get("type") == "text":
            grouped[key].append((iso(ts), role.upper(), d.get("text", "")))
        elif d.get("type") == "tool":
            st = d.get("state", {})
            body = f"input: {cap(json.dumps(st.get('input', '')))}\noutput: {cap(str(st.get('output', '')))}"
            grouped[key].append((iso(ts), f"TOOL {d.get('tool', '')}", body))
    con.close()
    return {k: (project.get(k[1], "?"), v) for k, v in grouped.items()}


def block_text(content: object) -> str:
    """Text of a message content that is a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content if isinstance(b, str | dict)
        )
    return ""


def from_claude_code(projects: Path) -> Sessions:
    """Claude Code transcripts (~/.claude/projects/**/*.jsonl). The lab comes from each session's model."""
    sessions: Sessions = {}
    for f in sorted(projects.rglob("*.jsonl")):
        msgs: list[Msg] = []
        model, project = "", f.parent.name
        for r in read_jsonl(f):
            msg = r.get("message")
            if r.get("type") not in {"user", "assistant"} or not isinstance(msg, dict):
                continue
            ts = iso(r.get("timestamp"))
            project = Path(r["cwd"]).name if r.get("cwd") else project
            model = model or (msg.get("model", "") if r["type"] == "assistant" else "")
            content = msg.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": block_text(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text"):
                    msgs.append((ts, r["type"].upper(), str(b["text"])))
                elif b.get("type") == "tool_use":
                    msgs.append((ts, f"TOOL {b.get('name', '')}", f"input: {cap(json.dumps(b.get('input', '')))}"))
                elif b.get("type") == "tool_result":
                    msgs.append((ts, "TOOL result", f"output: {cap(block_text(b.get('content')))}"))
        if msgs:
            lab = "anthropic-claude-code" if lab_for_model(model) in {"anthropic", "unknown"} else lab_for_model(model)
            sessions[(lab, f.stem)] = (project, msgs)
    return sessions


def from_claude_export(raw: Path) -> Sessions:
    """Unzipped claude.ai export: conversations (text, attachments, tool calls) and project docs."""
    lab = "anthropic-claude-ai"
    sessions: Sessions = {}
    for conv in load_json(raw / "conversations.json"):
        msgs: list[Msg] = []
        for m in conv.get("chat_messages", []):
            ts, who = iso(m.get("created_at")), "USER" if m.get("sender") == "human" else "ASSISTANT"
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
        proj = load_json(p)
        msgs = [("", "PROJECT instructions", str(proj.get("prompt_template", "")))]
        msgs += [
            ("", f"PROJECT doc {d.get('filename', '')}", cap(str(d.get("content", "")))) for d in proj.get("docs", [])
        ]
        sessions[(lab, f"project-{proj['uuid']}")] = (f"claude.ai project: {proj.get('name', '')}", msgs)
    return sessions


def chatgpt_messages(conv: dict[str, Any]) -> list[Msg]:
    """Visible user, assistant and tool messages of one ChatGPT conversation, in time order."""
    nodes = [n.get("message") for n in conv.get("mapping", {}).values()]
    msgs = [m for m in nodes if isinstance(m, dict)]
    out: list[Msg] = []
    for m in sorted(msgs, key=lambda m: m.get("create_time") or 0):
        role = m.get("author", {}).get("role", "")
        if role not in {"user", "assistant", "tool"} or m.get("metadata", {}).get(
            "is_visually_hidden_from_conversation"
        ):
            continue
        parts = m.get("content", {}).get("parts") or [m.get("content", {}).get("text", "")]
        text = "\n".join(p for p in parts if isinstance(p, str) and p)
        if text:
            who = f"TOOL {m.get('author', {}).get('name', '')}" if role == "tool" else role.upper()
            out.append((iso(m.get("create_time")), who, cap(text) if role == "tool" else text))
    return out


def from_chatgpt_export(raw: Path) -> Sessions:
    """Unzipped ChatGPT export (Settings → Data controls → Export): conversations.json."""
    sessions: Sessions = {}
    for conv in load_json(raw / "conversations.json"):
        msgs = chatgpt_messages(conv)
        if msgs:
            sid = str(conv.get("conversation_id") or conv.get("id") or len(sessions))
            sessions[("openai-chatgpt", sid)] = (f"chatgpt: {conv.get('title', '')}", msgs)
    return sessions


def write_chunks(out: Path, sessions: Sessions) -> Counter[str]:
    """Split each session into chunks of at most CHUNK_CHARS and write them redacted."""
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


def collect(args: argparse.Namespace) -> Sessions:
    """Read every source given on the command line."""
    readers = [
        (args.nibble, from_nibble),
        (args.opencode, from_opencode),
        (args.claude_code, from_claude_code),
        (args.claude_export, from_claude_export),
        (args.chatgpt_export, from_chatgpt_export),
    ]
    sessions: Sessions = {}
    for path, reader in readers:
        if path is None:
            continue
        if not path.exists():
            raise SystemExit(f"Source not found: {path}")
        sessions.update(reader(path))
    return sessions


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk chat and agent history per lab, credentials masked.")
    p.add_argument("out", type=Path, help="output directory for chunks")
    p.add_argument("--claude-export", type=Path, help="unzipped claude.ai export directory")
    p.add_argument("--chatgpt-export", type=Path, help="unzipped ChatGPT export directory")
    p.add_argument("--claude-code", type=Path, help="Claude Code transcripts, usually ~/.claude/projects")
    p.add_argument("--opencode", type=Path, help="opencode database, usually ~/.local/share/opencode/opencode.db")
    p.add_argument("--nibble", type=Path, help="nibble capture directory, usually ~/.nibble/memory/capture")
    args = p.parse_args(argv)
    if not any((args.claude_export, args.chatgpt_export, args.claude_code, args.opencode, args.nibble)):
        p.error("pass at least one source")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    counts = write_chunks(args.out, collect(args))
    chars = {lab: sum(p.stat().st_size for p in (args.out / lab).glob("*.txt")) for lab in counts}
    manifest = {"chunks": dict(counts), "MB": {k: round(v / 1e6, 1) for k, v in chars.items()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
