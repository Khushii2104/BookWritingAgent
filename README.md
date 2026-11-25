# Book Writing Agent OS

A multi-agent system that plans and writes a book with saved memory, quality checks and human approval.
Part 1 (build) is this repo. Part 2 (refining an existing book) is in [docs/refining_plan.md](docs/refining_plan.md).

```
Planner -> [author approves plan] -> per chapter:
  Outliner -> Writer (section by section) -> Critic <-> Editor -> [author approves] -> Archivist -> memory
```
Diagram: [docs/architecture.png](docs/architecture.png) (also [docs/architecture.md](docs/architecture.md)). Design choices and trade-offs: [docs/approach_note.md](docs/approach_note.md).

## Agents

| Agent | Job |
|---|---|
| **Planner** | Turns the setup into a title, voice rules, canonical names and a chapter plan. Revises it from author feedback. |
| **Outliner** | Breaks one chapter into 4-8 sections (beats) with word targets, so chapters reach real book length. |
| **Writer** | Drafts one section at a time using memory and the chapter so far; expands short sections; rewrites sections on request. |
| **Critic** | Rule checks (name drift, repetition against earlier chapters, length) plus an LLM review for consistency, invented facts, style, coverage and flow, with severities. |
| **Editor** | Turns critic or author feedback into the smallest fix: exact find/replace edits, or a targeted rewrite of only the affected sections. |
| **Archivist** | After approval, saves summary, names (with aliases), terms, facts, stories used and open threads to memory. |

Fiction, memoir and non-fiction each get their own planning structure and writing rules (e.g. memoir and non-fiction never invent facts and mark gaps as `[AUTHOR TO CONFIRM: ...]`).

## Setup
Requires Python 3.10+.
```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   then add OPENAI_API_KEY (never commit .env)
```

## Run
```bash
# interactive: asks for the setup, then for approval of the plan and of each chapter
python main.py

# from a setup file, write chapters 1 and 2 (length is read from the setup, e.g. "40,000 words, 12 chapters")
python main.py --setup examples/setup_nonfiction.json --chapters 2

# override the length
python main.py --setup examples/setup_nonfiction.json --words-per-chapter 4000 --num-chapters 12

# non-interactive demo (skips the human gates)
python main.py --setup examples/setup_nonfiction.json --auto-approve

# no API key: deterministic mock LLM (placeholder text, shows the flow and files)
python main.py --setup examples/setup_nonfiction.json --mock --auto-approve
```

At each gate: `a` approve, `r` revise (then type feedback), `v` view the full text, `q` quit.
Before approving a chapter you can also **edit `drafts/chapter_NN.md` directly**; your edits are kept.
Runs are resumable: the approved plan, approved chapters and even half-written chapters are reused, so re-run the same command after a quit, crash or network error.

Setup file fields: `book_type` (`fiction` | `memoir` | `non-fiction`), `idea`, `target_reader`, `length`, `style` are required. Optional: `author_notes` (facts, people, material the book must use — important for memoir), `num_chapters`, `words_per_chapter`.

## Simple web UI
Prefer clicking to typing commands? Start the local UI (no extra packages needed; it reads your key from .env by itself):
```bash
python ui.py #run
```
It opens http://127.0.0.1:8765 in your browser. Fill in the book setup (or click "Fill with example"), press Start, and the page shows live progress. When the plan or a chapter is ready, you get the text with **Approve / Revise with feedback / Stop** buttons. When it finishes you can read and download `manuscript.md`. Tick "Demo mode" to try it without an API key. The UI only listens on your own computer (127.0.0.1).

## Configuration (`.env`)
| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | – | required for the OpenAI provider |
| `BOOK_PROVIDER` | `openai` | or `anthropic` (needs `ANTHROPIC_API_KEY`) |
| `BOOK_MODEL` | `gpt-4o` / `claude-sonnet-4-5` | any chat model; reasoning models (o-series, gpt-5) are supported |
| `BOOK_MAX_TOKENS` | `8000` | max output tokens per call |
| `BOOK_TIMEOUT`, `BOOK_MAX_RETRIES` | `300`, `4` | request timeout and automatic retries on rate limits / server errors |
| `BOOK_PRICE_INPUT_PER_1M`, `BOOK_PRICE_OUTPUT_PER_1M` | – | optional; adds `estimated_cost_usd` to `usage.json` |
| `BOOK_MOCK` | `0` | `1` = offline mock LLM |

## Output (`book_data/<book>/`)
`setup.json`, `plan_approved.json`, `chapters/chapter_NN.md`, `manuscript.md` (all approved chapters),
`memory.json` (bible, names + aliases, terms, facts, stories used, open threads, summaries, chapter endings),
`review_chNN.json` (critic findings, word count, auto-fixes), `usage.json` (tokens per agent, optional cost),
`drafts/` (in-progress chapter, removed once approved).

## Tests
```bash
pytest -q
```
16 tests on the mock LLM: JSON parsing/retry, setup and length parsing, name-drift detection and auto-fix, repetition, memory aliases and UTF-8, end-to-end two chapters at target length, chapter 2 receiving chapter 1's memory, plan gate, author revision, resume after quitting mid-chapter, critic-to-editor fix loop, and author edits to the draft file.

## Layout
`bookos/agents.py` six agents and their prompts · `bookos/orchestrator.py` pipeline, loops and gates · `bookos/memory.py` file memory and drafts · `bookos/checks.py` rule-based checks · `bookos/llm.py` OpenAI, Anthropic and mock clients with retries and usage tracking · `main.py` CLI · `ui.py` local web UI.

## Sample output
No sample is bundled. Generate a real one (about 3,300 words per chapter for the example setup) and keep it in `examples/sample_output/`:
```bash
python main.py --setup examples/setup_nonfiction.json --chapters 2 --auto-approve --out examples/sample_output
```
(`book_data/` is gitignored; `examples/sample_output/` is not, so a real sample can be committed.)
