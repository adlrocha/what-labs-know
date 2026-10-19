# what-labs-know

Works out what each AI provider received from your chat and agent history, and what a model can
infer about you from it. Profiling runs on hardware you control, so your history doesn't go to
yet another lab.

## Pipeline

1. `leak_map.py`: regex counts per lab (sessions, MB you typed vs MB your agents read, data
   categories). Prints aggregates only. Put your employers, places and usernames in `config.json`
   (see `config.example.json`, gitignored) or set `WLK_CONFIG`.
2. `chunk.py OUT_DIR [--claude-export RAW_DIR]`: splits nibble captures, the opencode database or
   an unzipped claude.ai export into per-lab chunks. Credentials are masked before anything is
   written.
3. `run.py BASE_DIR LAB...`: Stage 1 extracts claims per chunk, Stage 2 merges them and writes a
   numbered profile. Talks to any OpenAI-compatible server (`API`, `MODEL`, `WORKERS`), or to a
   command (`CHAT_CMD`) that takes the system prompt as its last argument and the text on stdin.
   Resumable.
4. `score.py sheet BASE_DIR` writes one TSV per profile to fill in: `right`, `wrong` or `hdik`
   ("how did it know?"), and `ok` or `no` for whether you'd publish it. `score.py tally BASE_DIR`
   prints accuracy and exposure per lab as counts.
5. `status.sh BASE_DIR`: progress from file counts, safe to watch remotely.

## Develop

```sh
uvx pytest
uvx ruff check . && uvx ruff format --check .
```

## Known limits

- nibble captures without a `pi-`/`omp-` prefix are assumed to be Claude Code.
- The readers for each source (captures, opencode, claude.ai export) have no tests yet.
