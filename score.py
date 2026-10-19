"""Turn profiles into scoring sheets, and tally scored sheets. Stdlib only.

Usage:
  python3 score.py sheet BASE_DIR   stage2/<lab>.md -> scoring/<lab>.tsv (never overwrites)
  python3 score.py tally BASE_DIR   prints counts per lab, never claim text

Sheet columns: id, section, claim, score, publish.
  score:   right | wrong | hdik  (hdik = "how did it know?", counts as right)
  publish: ok | no
Accuracy = (right + hdik) / scored. Exposure = right-or-hdik claims marked `no`.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

CLAIM = re.compile(r"^\s*(?:[-*|]\s*)?(?:\*\*)?\[?P(\d+)\]?(?:\*\*)?[\s.:)\]|*-]*(.*)$")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")
COLUMNS = ["id", "section", "claim", "score", "publish"]
SCORES = {"right", "wrong", "hdik"}
PUBLISH = {"ok", "no"}


def parse_profile(md: str) -> list[dict[str, str]]:
    """Numbered claims in order of first appearance, with the heading they sit under."""
    section, seen, rows = "", set(), []
    for line in md.splitlines():
        if h := HEADING.match(line):
            section = h.group(1).strip()
            continue
        m = CLAIM.match(line)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        text = m.group(2).replace("**", "").strip(" |")
        rows.append({"id": f"P{m.group(1)}", "section": section, "claim": text, "score": "", "publish": ""})
    return rows


def write_sheets(base: Path) -> dict[str, int]:
    """One TSV per profile; existing sheets are left alone so scores are never lost."""
    out = base / "scoring"
    out.mkdir(exist_ok=True)
    made = {}
    for md in sorted((base / "stage2").glob("*.md")):
        dest = out / f"{md.stem}.tsv"
        if dest.exists():
            continue
        rows = parse_profile(md.read_text(encoding="utf-8"))
        with dest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, COLUMNS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        made[md.stem] = len(rows)
    return made


def tally_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """Counts and rates for one sheet. Unknown values count as unscored and are reported."""
    c: Counter[str] = Counter()
    for r in rows:
        s, p = r.get("score", "").strip().lower(), r.get("publish", "").strip().lower()
        c["claims"] += 1
        if s not in SCORES:
            c["unscored" if not s else "invalid"] += 1
            continue
        c[s] += 1
        if p not in PUBLISH:
            c["publish_missing"] += 1
        elif s != "wrong" and p == "no":
            c["exposure"] += 1
    scored = c["right"] + c["wrong"] + c["hdik"]
    acc = round((c["right"] + c["hdik"]) / scored, 2) if scored else None
    return {
        **{k: c[k] for k in ("claims", "right", "hdik", "wrong", "exposure", "unscored", "invalid", "publish_missing")},
        "accuracy": acc,
    }


def tally(base: Path) -> dict[str, dict[str, object]]:
    result = {}
    for tsv in sorted((base / "scoring").glob("*.tsv")):
        with tsv.open(encoding="utf-8", newline="") as f:
            result[tsv.stem] = tally_rows(list(csv.DictReader(f, delimiter="\t")))
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"sheet", "tally"}:
        print(__doc__, file=sys.stderr)
        return 2
    base = Path(argv[2])
    if argv[1] == "sheet":
        for lab, n in write_sheets(base).items():
            print(f"{lab}: {n} claims")
    else:
        for lab, t in tally(base).items():
            print(lab, " ".join(f"{k}={v}" for k, v in t.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
