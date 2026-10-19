"""Tests for chunk.py, run.py and leak_map.py, on synthetic data only."""

from __future__ import annotations

import chunk
import json
import sqlite3
import sys
from pathlib import Path

import pytest

import leak_map
import run

FAKE_LLM = f"{sys.executable} -c \"import sys; print('SYS=' + sys.argv[-1] + ' IN=' + sys.stdin.read())\""
GH = "ghp_" + "z" * 36


def jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n[1]\n")
    return path


# chunk.py: helpers


def test_redact_masks_each_secret_kind() -> None:
    text = (
        "ghp_"
        + "a" * 36
        + " sk-ant-"
        + "b" * 30
        + " AKIA"
        + "C" * 16
        + " AIza"
        + "d" * 35
        + " password=hunter2hunter2 -----BEGIN RSA PRIVATE KEY-----\nxyz\n-----END RSA PRIVATE KEY-----"
    )
    out = chunk.redact(text)
    for kind in ("github-token", "api-key", "aws-key", "google-key", "private-key"):
        assert f"[REDACTED:{kind}]" in out
    assert "password=[REDACTED:secret]" in out
    assert "hunter2" not in out and "xyz" not in out


def test_redact_leaves_plain_text() -> None:
    assert chunk.redact("token budget is small") == "token budget is small"


@pytest.mark.parametrize(
    ("model", "lab"),
    [
        ("ollama/llama3", "local"),
        ("github-copilot/claude-sonnet", "copilot-anthropic"),
        ("github-copilot/gpt-5", "copilot-other"),
        ("zai-coding-plan/glm-5", "zai"),
        ("glm-5.3-flash", "zai"),
        ("kimi-for-coding/k2", "moonshot"),
        ("opencode/grok", "opencode-zen"),
        ("claude-opus-4-6", "anthropic"),
        ("gpt-5", "openai"),
        ("mystery/x", "unknown"),
    ],
)
def test_lab_for_model(model: str, lab: str) -> None:
    assert chunk.lab_for_model(model) == lab


def test_capture_lab_cap_and_iso() -> None:
    assert chunk.capture_lab("pi-123.jsonl") == "pi-unknown"
    assert chunk.capture_lab("omp-1.jsonl") == "omp-unknown"
    assert chunk.capture_lab("0b1c.jsonl") == "anthropic-claude-code"
    assert chunk.cap("x" * 10) == "x" * 10
    assert chunk.cap("x" * (chunk.TOOL_CAP + 5)).endswith("[5 chars cut]")
    assert chunk.iso(0) == "1970-01-01T00:00:00"
    assert chunk.iso(1_700_000_000_000) == chunk.iso(1_700_000_000)
    assert chunk.iso("2026-01-02T03:04:05.678Z") == "2026-01-02T03:04:05"
    assert chunk.iso(None) == ""


def test_block_text() -> None:
    assert chunk.block_text("hi") == "hi"
    assert chunk.block_text([{"type": "text", "text": "a"}, "b", 3]) == "a\nb"
    assert chunk.block_text(None) == ""


def test_load_json_reports_bad_file(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(SystemExit, match="Failed to read"):
        chunk.load_json(tmp_path / "bad.json")


def test_write_chunks_splits_and_redacts(tmp_path: Path) -> None:
    msgs = [("t", "USER", "y" * (chunk.CHUNK_CHARS // 2)) for _ in range(3)] + [("t", "USER", GH)]
    counts = chunk.write_chunks(tmp_path, {("lab", "s/1"): ("proj", msgs)})
    files = sorted((tmp_path / "lab").glob("*.txt"))
    assert counts["lab"] == len(files) >= 2
    assert files[0].name.startswith("s_1-")
    assert all(f.stat().st_size <= chunk.CHUNK_CHARS + 100 for f in files)
    assert "[REDACTED:github-token]" in files[-1].read_text()


# chunk.py: readers


def test_from_nibble(tmp_path: Path) -> None:
    rows = [
        {"ts": "2026-01-01T00:00:00Z", "role": "user", "content": "hello"},
        {"ts": "2026-01-01T00:00:01Z", "role": "assistant", "content": "hi"},
        {"ts": "2026-01-01T00:00:02Z", "role": "tool", "name": "Bash", "input": "ls", "output": "a.txt"},
        {"role": "system", "content": "ignored"},
    ]
    jsonl(tmp_path / "proj" / "abc.jsonl", rows)
    jsonl(tmp_path / "proj" / "pi-1.jsonl", rows[:1])
    jsonl(tmp_path / "proj" / "empty.jsonl", [{"role": "system"}])
    jsonl(tmp_path / "test-project" / "x.jsonl", rows)
    s = chunk.from_nibble(tmp_path)
    assert set(s) == {("anthropic-claude-code", "abc"), ("pi-unknown", "pi-1")}
    proj, msgs = s[("anthropic-claude-code", "abc")]
    assert proj == "proj"
    assert [m[1] for m in msgs] == ["USER", "ASSISTANT", "TOOL Bash"]
    assert "output: a.txt" in msgs[2][2]


def test_from_opencode(tmp_path: Path) -> None:
    db = tmp_path / "oc.db"
    con = sqlite3.connect(db)
    con.executescript(
        "create table message (id, session_id, time_created, data);"
        "create table part (message_id, session_id, time_created, data);"
        "create table session (id, directory);"
    )
    con.executemany(
        "insert into message values (?,?,?,?)",
        [
            ("m1", "s1", 1, json.dumps({"role": "user"})),
            ("m2", "s1", 2, json.dumps({"role": "assistant", "providerID": "zai", "modelID": "glm-5"})),
        ],
    )
    con.executemany(
        "insert into part values (?,?,?,?)",
        [
            ("m1", "s1", 1_700_000_000_000, json.dumps({"type": "text", "text": "hi"})),
            ("m2", "s1", 1_700_000_001_000, json.dumps({"type": "tool", "tool": "read", "state": {"output": "x"}})),
            ("m2", "s1", 1_700_000_002_000, json.dumps({"type": "reasoning"})),
        ],
    )
    con.execute("insert into session values ('s1', '/home/u/proj')")
    con.commit()
    con.close()
    s = chunk.from_opencode(db)
    proj, msgs = s[("zai", "s1")]
    assert proj == "proj"
    assert [m[1] for m in msgs] == ["USER", "TOOL read"]
    assert msgs[0][0].startswith("2023-11-14")


def test_from_claude_code(tmp_path: Path) -> None:
    rows = [
        {"type": "user", "timestamp": "2026-02-01T10:00:00Z", "cwd": "/home/u/app", "message": {"content": "fix it"}},
        {
            "type": "assistant",
            "timestamp": "2026-02-01T10:00:01Z",
            "message": {
                "model": "claude-opus-4-6",
                "content": [
                    {"type": "thinking", "thinking": "skip"},
                    "stray",
                    {"type": "text", "text": "on it"},
                    {"type": "tool_use", "name": "Read", "input": {"path": "a.py"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": [{"type": "text", "text": "src"}]}]},
        },
        {"type": "summary", "summary": "ignored"},
        {"type": "user", "message": "not a dict"},
    ]
    jsonl(tmp_path / "-home-u-app" / "s1.jsonl", rows)
    kimi = [{"type": "assistant", "message": {"model": "kimi-k2", "content": "hello"}}]
    jsonl(tmp_path / "-home-u-app" / "s2.jsonl", kimi)
    jsonl(tmp_path / "-home-u-app" / "s3.jsonl", [{"type": "summary"}])
    s = chunk.from_claude_code(tmp_path)
    assert set(s) == {("anthropic-claude-code", "s1"), ("moonshot", "s2")}
    proj, msgs = s[("anthropic-claude-code", "s1")]
    assert proj == "app"
    assert [m[1] for m in msgs] == ["USER", "ASSISTANT", "TOOL Read", "TOOL result"]
    assert msgs[3][2] == "output: src"


def test_from_claude_export(tmp_path: Path) -> None:
    convs = [
        {
            "uuid": "c1",
            "name": "Trip",
            "chat_messages": [
                {
                    "sender": "human",
                    "created_at": "2026-03-01T00:00:00Z",
                    "content": [{"type": "text", "text": "plan"}],
                    "attachments": [{"file_name": "cv.pdf", "extracted_content": "Jane Doe"}],
                },
                {
                    "sender": "assistant",
                    "text": "fallback",
                    "content": [{"type": "tool_use", "name": "web_search", "input": {"q": "x"}}],
                },
            ],
        },
        {"uuid": "c2", "chat_messages": []},
    ]
    (tmp_path / "conversations.json").write_text(json.dumps(convs))
    (tmp_path / "projects").mkdir()
    proj = {"uuid": "p1", "name": "Work", "prompt_template": "be brief", "docs": [{"filename": "a.md", "content": "x"}]}
    (tmp_path / "projects" / "p1.json").write_text(json.dumps(proj))
    s = chunk.from_claude_export(tmp_path)
    assert set(s) == {("anthropic-claude-ai", "c1"), ("anthropic-claude-ai", "project-p1")}
    _, msgs = s[("anthropic-claude-ai", "c1")]
    assert [m[1] for m in msgs] == ["USER", "ATTACHMENT cv.pdf", "ASSISTANT", "TOOL web_search"]
    assert msgs[2][2] == "fallback"
    assert s[("anthropic-claude-ai", "project-p1")][1][1][1] == "PROJECT doc a.md"


def test_from_chatgpt_export(tmp_path: Path) -> None:
    def node(role: str, t: float | None, parts: list, **meta: object) -> dict:
        content = {"parts": parts}
        return {
            "message": {
                "author": {"role": role, "name": "browser"},
                "create_time": t,
                "content": content,
                "metadata": meta,
            }
        }

    convs = [
        {
            "conversation_id": "g1",
            "title": "Taxes",
            "mapping": {
                "root": {"message": None},
                "a": node("system", 1, ["sys"]),
                "b": node("user", 3, ["my salary", {"image": 1}]),
                "c": node("assistant", 4, ["ok"]),
                "d": node("tool", 5, ["result"]),
                "e": node("user", 2, ["hidden"], is_visually_hidden_from_conversation=True),
                "f": node("assistant", None, [""]),
            },
        },
        {"id": "g2", "mapping": {}},
    ]
    (tmp_path / "conversations.json").write_text(json.dumps(convs))
    s = chunk.from_chatgpt_export(tmp_path)
    assert set(s) == {("openai-chatgpt", "g1")}
    proj, msgs = s[("openai-chatgpt", "g1")]
    assert proj == "chatgpt: Taxes"
    assert [(m[1], m[2]) for m in msgs] == [("USER", "my salary"), ("ASSISTANT", "ok"), ("TOOL browser", "result")]


def test_chunk_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jsonl(tmp_path / "cap" / "p" / "s.jsonl", [{"role": "user", "content": GH}])
    out = tmp_path / "out"
    assert chunk.main([str(out), "--nibble", str(tmp_path / "cap")]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == {"anthropic-claude-code": 1}
    assert (out / "manifest.json").exists()
    with pytest.raises(SystemExit, match="Source not found"):
        chunk.main([str(out), "--claude-code", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        chunk.main([str(out)])


# run.py


def test_parse_claims_keeps_valid_lines() -> None:
    text = '{"claim": "a"}\n{"claim": ""}\nnot json\n{broken\n  {"claim": "b"},\n'
    claims, bad = run.parse_claims(text)
    assert [c["claim"] for c in claims] == ["a", "b"]
    assert bad == 2


def test_batches_respect_budget() -> None:
    assert run.batches(["a" * 4, "b" * 4, "c" * 4], 10) == ["aaaa\nbbbb\n", "cccc\n"]
    assert run.batches([], 10) == []


def test_chat_cmd_passes_system_as_arg_and_user_on_stdin() -> None:
    assert run.chat_cmd(FAKE_LLM, "S", "U").strip() == "SYS=S IN=U"


def test_chat_cmd_fails_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="failed after retries"):
        run.chat_cmd(f'{sys.executable} -c "raise SystemExit(1)"', "S", "U", retries=2)


def test_chat_uses_cmd_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "CHAT_CMD", FAKE_LLM)
    assert "IN=hello" in run.chat("sys", "hello", 10)


def test_stage1_then_stage2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "chunks" / "lab").mkdir(parents=True)
    for i in range(3):
        (tmp_path / "chunks" / "lab" / f"c{i}.txt").write_text(f"chunk {i}")
    monkeypatch.setattr(run, "STAGE2_BUDGET_CHARS", 60)

    def fake_chat(system: str, user: str, max_tokens: int) -> str:
        if system == run.STAGE1:
            if "chunk 2" in user:
                raise RuntimeError("boom")
            return "\n".join(json.dumps({"claim": f"{user} fact {j}", "category": "x"}) for j in range(3))
        if system == run.MERGE:
            return "x | merged | stated | 1 | 3"
        return "**P1** merged"

    monkeypatch.setattr(run, "chat", fake_chat)
    monkeypatch.setattr(sys, "argv", ["run.py", str(tmp_path), "lab"])
    run.main()
    assert len(list((tmp_path / "stage1" / "lab").glob("*.jsonl"))) == 2
    assert "boom" in (tmp_path / "errors.log").read_text()
    assert (tmp_path / "stage2" / "lab.md").read_text() == "**P1** merged"
    assert json.loads((tmp_path / "progress.json").read_text())["lab"]["stage"] == "done"


def test_main_logs_stage2_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "stage1" / "lab").mkdir(parents=True)

    def fail(*_a: object) -> str:
        raise RuntimeError("down")

    monkeypatch.setattr(run, "chat", fail)
    monkeypatch.setattr(sys, "argv", ["run.py", str(tmp_path), "lab"])
    run.main()
    assert "stage2 failed: down" in (tmp_path / "errors.log").read_text()


# leak_map.py


def test_load_config(tmp_path: Path) -> None:
    assert leak_map.load_config(tmp_path / "missing.json") == {}
    good = tmp_path / "c.json"
    good.write_text('{"employers": ["Acme"], "note": "ignored"}')
    assert leak_map.load_config(good) == {"employers": ["Acme"]}
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(SystemExit, match="Failed to parse config"):
        leak_map.load_config(bad)


def test_personal_categories_match_whole_words() -> None:
    cats = leak_map.personal_categories({"employers": ["Acme"], "places": ["Springfield"]})
    assert cats["work_employer"].search("I work at ACME")
    assert not cats["work_employer"].search("acmeville")
    assert cats["location"].search("moved to Springfield")
    assert "identity" not in cats
    assert leak_map.term_pattern([]) is None


def test_split_messages_kinds_and_dates() -> None:
    text = (
        "# session s · project p\n\n[USER 2026-01-02T03:04:05]\nhi\n\n"
        "[TOOL Bash 2026-01-03T00:00:00]\nout\n\n[ASSISTANT ]\nok\n"
    )
    kinds = [(k, d) for k, d, _ in leak_map.split_messages(text)]
    assert kinds == [("you_typed", "2026-01-02"), ("agent_read", "2026-01-03"), ("model_wrote", "")]


def test_leak_map_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    msgs = [
        ("2026-01-02T00:00:00", "USER", "mail a@b.co about my salary at Acme " + GH),
        ("2026-01-05T00:00:00", "TOOL Read", "x" * 30_000),
    ]
    chunk.write_chunks(tmp_path / "chunks", {("lab", "s1"): ("proj", msgs), ("lab", "s2"): ("proj", msgs[1:])})
    (tmp_path / "chunks" / "lab" / "stray.txt").write_text("no header")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"employers": ["Acme"]}')
    assert leak_map.main([str(tmp_path / "chunks"), "--config", str(cfg)]) == 0
    r = json.loads(capsys.readouterr().out)["lab"]
    assert r["sessions"] == 2 and r["dates"] == "2026-01-02..2026-01-05"
    assert r["sessions_with"]["email"] == 1 and r["sessions_with"]["work_employer"] == 1
    assert r["secrets_masked"] == {"github-token": 1}
    assert set(r["MB"]) == {"you_typed", "agent_read"} and r["MB"]["agent_read"] == 0.06
    assert leak_map.main([str(tmp_path / "chunks"), "--markdown", "--config", str(tmp_path / "none.json")]) == 0
    md = capsys.readouterr().out
    assert "| lab | 2 |" in md and "| secret | 1 |" in md
    with pytest.raises(SystemExit, match="Not a directory"):
        leak_map.main([str(tmp_path / "nope")])
