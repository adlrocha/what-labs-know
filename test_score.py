"""Tests for score.py, on synthetic profiles only."""

from __future__ import annotations

import csv
from pathlib import Path

import score

PROFILE = """# Profile
## Work
- **P1.** Works at Acme as an engineer.
- **P2**: Lives in Springfield.
| P3 | Has two cats | 0.9 |
P4) Plain line claim.
## Second-order inferences
- P5 Likely to change jobs (supports P1, P2).
## Most sensitive
1. **P2** Lives in Springfield, repeated reference.
"""


def test_parse_profile_numbers_sections_and_dedup() -> None:
    rows = score.parse_profile(PROFILE)
    assert [r["id"] for r in rows] == ["P1", "P2", "P3", "P4", "P5"]
    assert rows[0]["claim"] == "Works at Acme as an engineer."
    assert rows[2]["claim"].startswith("Has two cats")
    assert rows[4]["section"] == "Second-order inferences"


def test_parse_profile_without_claims() -> None:
    assert score.parse_profile("# Nothing\nno numbered claims here\n") == []


def test_tally_rows_counts_and_rates() -> None:
    rows = [
        {"score": "right", "publish": "no"},
        {"score": "HDIK", "publish": "no"},
        {"score": "wrong", "publish": "no"},
        {"score": "right", "publish": "ok"},
        {"score": "right", "publish": ""},
        {"score": "", "publish": ""},
        {"score": "maybe", "publish": "ok"},
    ]
    t = score.tally_rows(rows)
    assert (t["claims"], t["right"], t["hdik"], t["wrong"]) == (7, 3, 1, 1)
    assert (t["exposure"], t["unscored"], t["invalid"], t["publish_missing"]) == (2, 1, 1, 1)
    assert t["accuracy"] == 0.8


def test_tally_rows_empty_sheet_has_no_accuracy() -> None:
    assert score.tally_rows([])["accuracy"] is None


def test_sheet_then_tally_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    (tmp_path / "stage2").mkdir()
    (tmp_path / "stage2" / "lab.md").write_text(PROFILE)
    assert score.write_sheets(tmp_path) == {"lab": 5}
    sheet = tmp_path / "scoring" / "lab.tsv"
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    rows[0].update(score="right", publish="no")
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, score.COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    assert score.write_sheets(tmp_path) == {}
    t = score.tally(tmp_path)["lab"]
    assert (t["right"], t["exposure"], t["unscored"], t["accuracy"]) == (1, 1, 4, 1.0)


def test_main_rejects_bad_usage() -> None:
    assert score.main(["score.py"]) == 2
    assert score.main(["score.py", "nope", "."]) == 2
