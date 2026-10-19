"""Tests for chunk.py, run.py and leak_map.py, on synthetic data only."""

from __future__ import annotations

import chunk
import json
import sys
from pathlib import Path

import pytest

import leak_map
import run

FAKE_LLM = f"{sys.executable} -c \"import sys; print('SYS=' + sys.argv[-1] + ' IN=' + sys.stdin.read())\""


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
        ("kimi-for-coding/k2", "moonshot"),
        ("opencode/grok", "opencode-zen"),
        ("mystery/x", "unknown"),
    ],
)
def test_chunk_lab_for_model(model: str, lab: str) -> None:
    assert chunk.lab_for_model(model) == lab


def test_capture_lab_and_cap() -> None:
    assert chunk.capture_lab("pi-123.jsonl") == "pi-unknown"
    assert chunk.capture_lab("omp-1.jsonl") == "omp-unknown"
    assert chunk.capture_lab("0b1c.jsonl") == "anthropic-claude-code"
    assert chunk.cap("x" * 10) == "x" * 10
    assert chunk.cap("x" * (chunk.TOOL_CAP + 5)).endswith("[5 chars cut]")


def test_write_chunks_splits_and_redacts(tmp_path: Path) -> None:
    msgs = [("t", "USER", "y" * (chunk.CHUNK_CHARS // 2)) for _ in range(3)] + [("t", "USER", "ghp_" + "z" * 36)]
    counts = chunk.write_chunks(tmp_path, {("lab", "s1"): ("proj", msgs)})
    files = sorted((tmp_path / "lab").glob("*.txt"))
    assert counts["lab"] == len(files) >= 2
    assert all(f.stat().st_size <= chunk.CHUNK_CHARS + 100 for f in files)
    assert "[REDACTED:github-token]" in files[-1].read_text()


def test_parse_claims_keeps_valid_lines() -> None:
    text = '{"claim": "a"}\n{"claim": ""}\nnot json\n{broken\n  {"claim": "b"},\n'
    claims, bad = run.parse_claims(text)
    assert [c["claim"] for c in claims] == ["a", "b"]
    assert bad == 2


def test_batches_respect_budget() -> None:
    parts = run.batches(["a" * 4, "b" * 4, "c" * 4], 10)
    assert parts == ["aaaa\nbbbb\n", "cccc\n"]
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


def test_stage2_merges_then_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "stage1" / "lab"
    out.mkdir(parents=True)
    (out / "c.jsonl").write_text("".join(json.dumps({"claim": f"c{i}", "category": "x"}) + "\n" for i in range(50)))
    monkeypatch.setattr(run, "STAGE2_BUDGET_CHARS", 500)
    calls: list[str] = []

    def fake_chat(system: str, user: str, max_tokens: int) -> str:
        calls.append(system[:20])
        return "x | merged | stated | 1 | 50" if system == run.MERGE else "**P1** merged"

    monkeypatch.setattr(run, "chat", fake_chat)
    run.stage2(tmp_path, "lab", run.Progress(tmp_path / "progress.json"))
    assert (tmp_path / "stage2" / "lab.md").read_text() == "**P1** merged"
    assert calls.count(run.MERGE[:20]) >= 2 and calls[-1] == run.STAGE2[:20]


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


def test_leak_map_lab_and_harness() -> None:
    assert leak_map.lab_for_model("github-copilot/claude-x") == "GitHub/Microsoft -> Anthropic"
    assert leak_map.lab_for_model("zai/glm") == "Z.ai (Zhipu, China)"
    assert leak_map.lab_for_model("anthropic/claude") == "Anthropic"
    assert leak_map.harness_for_capture("pi-1") == "pi"


def test_tally_counts_categories_without_storing_text() -> None:
    t = leak_map.Tally()
    t.add("s", "p", "you_typed", "mail me at a@b.co about the invoice €300", "2026-01-02T00:00")
    assert t.sessions_with["email"] == {"s"} and t.hits["money"]["you_typed"] >= 2
    assert (t.first, t.last) == ("2026-01-02", "2026-01-02")
