# what-labs-know

Find out what each AI provider has received from you, and what a model can work out about you
from it.

It reads your chat exports and coding-agent transcripts, sorts every session by the provider
that served it, counts the personal data in each provider's share, and has a model **you run
yourself** write the profile that provider could build. You then score that profile claim by
claim.

It was built for an experiment on my own history (write-up on
[adlrocha.substack.com](https://adlrocha.substack.com/)). Across six providers, about 90% of
what each one got was what my agents read on my machine (files, command output), not what I
typed.

## What it reads

| Source | Flag | Where to get it |
|---|---|---|
| claude.ai export | `--claude-export DIR` | Settings → Privacy → Export data; unzip into `DIR` |
| ChatGPT export | `--chatgpt-export DIR` | Settings → Data controls → Export; unzip into `DIR` |
| Claude Code transcripts | `--claude-code DIR` | `~/.claude/projects` |
| opencode | `--opencode DB` | `~/.local/share/opencode/opencode.db` |
| nibble captures | `--nibble DIR` | `~/.nibble/memory/capture` |

Sessions are assigned to a provider from the model they used where the source records it
(opencode, Claude Code), and from the product otherwise.

## Quick start

Python 3.11+, no dependencies. Run from a clone of this repo.

```sh
# 1. Chunk your history. API keys, tokens and private keys are masked before anything is written.
python3 -I chunk.py data/chunks --claude-export ~/Downloads/claude-export --claude-code ~/.claude/projects

# 2. Leak map: counts per provider, no text. Add your employers, places and usernames first.
cp config.example.json config.json   # then edit; it's gitignored
python3 -I leak_map.py data/chunks --markdown

# 3. Profile, on a model you control (any OpenAI-compatible server).
API=http://localhost:11434/v1 MODEL=qwen3:32b python3 -I run.py data anthropic-claude-ai

# 4. Score it: fill in data/scoring/<provider>.tsv, then tally.
python3 -I score.py sheet data
python3 -I score.py tally data
```

`run.py` resumes where it stopped. `status.sh data` prints progress from file counts, so you
can watch a long run over ssh without printing any of your data.

## Privacy rules this tool follows, and you should too

- **Don't profile on a hosted model you haven't already sent this data to.** Sending your
  history to a new provider so it can tell you what the old one knows is the leak this tool
  measures. Use a local server (Ollama, llama.cpp, vLLM, LM Studio).
- **Don't paste the profile into a chatbot to discuss it.** Same reason.
- Secrets are masked by pattern (GitHub, OpenAI/Anthropic, AWS, Google, Slack, JWT, private
  keys, `password=`-style assignments). Patterns miss things: rotate anything you find.
- Outputs (`data/`) and `config.json` are gitignored. Keep them that way.

## Sizing the run

Chunks default to 150,000 characters, about 40k tokens. **Your server's context window must
fit a chunk plus the prompt plus ~4k tokens of output**, or the model silently sees only part
of each chunk. Ollama defaults to a small context: raise `num_ctx`, or shrink chunks with
`WLK_CHUNK_CHARS` (roughly 3× the context length in tokens).

For scale: my run (1,290 chunks, about 80 MB) took 19 hours on a DGX Spark with
deepseek-v4-flash at about 75 output tokens/s.

## How it works

1. **Chunk** (`chunk.py`): one session per chunk, split if long. Tool output and attachments
   are capped at 2,000 characters each (`WLK_TOOL_CAP`).
2. **Leak map** (`leak_map.py`): per provider, sessions, dates, megabytes you typed versus
   megabytes your agents read, and how many sessions touch email addresses, money, family,
   health, location, career, your employers and identity. Counts only.
3. **Stage 1** (`run.py`): for each chunk, up to 40 claims about you as JSON lines, each with
   category, evidence, whether it was stated or inferred, and confidence.
4. **Stage 2** (`run.py`): merges claims in rounds until they fit one context, then writes a
   profile with numbered claims, "second-order inferences" no single session states, and the
   ten claims you'd least want held.
5. **Score** (`score.py`): mark each claim `right`, `wrong` or `hdik` ("how did it know?"),
   and `ok`/`no` for whether you'd publish it. Accuracy = (right + hdik) / scored; exposure =
   correct claims you marked `no`.

The prompts are at the top of `run.py`. `CHAT_CMD` swaps the HTTP API for a command that takes
the system prompt as its last argument and the text on stdin.

## Limits

- Leak-map counts are keyword matches: read them as upper bounds.
- "What you typed" includes text your client injects into user messages (instruction files,
  reminders).
- A profile shows what a model can infer from what you sent, not what any provider has done
  with it.

## Develop

```sh
uvx pytest
uvx ruff check . && uvx ruff format --check .
```

MIT licence.
