# Approach note: Part 1

## What I built
A CLI pipeline in plain Python with six agents, each with one job, its own prompt and a validated output shape. It runs on the OpenAI API by default; Anthropic works through the same small interface, and a deterministic mock is used for tests and offline demos. I didn't use an agent framework. The flow is a linear pipeline with two loops (critic/editor and author revision), and plain functions keep every prompt, decision and file write easy to see and test.

```
setup -> Planner -> [gate 1: author approves plan] -> per chapter:
  Outliner -> Writer (section by section, expand if short) -> Critic <-> Editor (max N rounds)
  -> [gate 2: author approves / revises / edits the draft file] -> Archivist -> memory -> manuscript
```

## Agents
- **Planner**: setup to title, premise, voice rules (POV, tense), canonical key names and chapter plan. The structure depends on book type: an arc for fiction, episodes for memoir, a progressive argument for non-fiction.
- **Outliner**: splits one chapter into 4-8 sections with briefs and word targets that add up to the chapter target.
- **Writer**: writes one section per call, with the outline, memory and the last ~1,500 words of the chapter so far. A section that comes in under 75% of its target is expanded once.
- **Critic**: runs rule-based checks first (name drift, near-duplicate sentences against earlier chapters through a trigram index, length), then an LLM review for consistency, invented facts, style, coverage and flow. Every issue has a severity, an exact quote and a fix.
- **Editor**: turns issues or author feedback into the smallest change. That is either exact find/replace edits, or a rewrite of only the affected sections, which the Writer carries out.
- **Archivist**: after approval, extracts summary, entities, terms, facts, stories used and open threads.

## Why chapters are now book-length
A single "write the chapter" call usually gives back 800-1,500 words, whatever you ask for. The Outliner plus section-by-section writing makes length a matter of structure. Each call has a small, concrete brief (~700 words), so the prose stays specific instead of padded. The target comes from the setup ("40,000 words, 12 chapters" gives ~3,333 words per chapter) or from `--words-per-chapter`.

## Memory
Memory is plain files under `book_data/<book>/`. `memory.json` holds:
- the bible (title, premise, voice)
- entities with aliases ("Maya" is folded into "Maya Okafor")
- terms and facts
- stories already used
- open threads
- chapter summaries
- the closing passage of each chapter, so the next one picks up where it ended

Memory is written only after a human approval, so rejected drafts never pollute it. Key names are seeded from the approved plan, so Chapter 1 already uses canonical spellings. Chapter 2 sees Chapter 1 through this memory, and the Critic compares Chapter 2 against Chapter 1's full text for repetition.

## Human approval and robustness
- **Gate 1 (plan) and gate 2 (each chapter):** approve, revise with feedback, or quit. At gate 2 the author can also edit `drafts/chapter_NN.md` directly, and those edits are kept.
- **Resumable at section level:** state is saved after every section, so a quit, crash or network error loses at most one call.
- **Network handling:** SDK retries with backoff, timeouts, one retry when the model returns invalid JSON, and warnings when output hits the token limit.
- **Safe writes:** atomic UTF-8 file writes.
- **Cost tracking:** token use per agent in `usage.json`, with an optional cost estimate.

## Key trade-offs
- **Files over a database:** easy to inspect and diff, and the author can edit them by hand. There is no concurrent writing, which is fine for one author and one book.
- **Section-by-section writing:** reliable length and focus, at the cost of more calls (~12-17 per chapter) and a small risk of seams. The Writer always sees the text just before its section, and the Critic checks flow.
- **Summary-based memory:** cheap and scales well, but a detail the Archivist misses can be forgotten. Facts, names and used stories are kept as separate lists to reduce that, and the repetition check runs on full earlier text.
- **Edits over rewrites:** the Editor prefers minimal find/replace, which keeps approved text stable and costs little. Edits whose exact text can't be found are skipped and logged.
- **Name-drift auto-fix:** fixed mechanically (same first letter, high similarity), so it is fast and exact. Rare false positives show up in `review_chNN.json`.
- **Capped revision loop:** issues still open after N rounds are shown to the author instead of looping forever.

## Interfaces
The same pipeline is driven by a CLI (`main.py`) and a small local web page (`ui.py`, standard library only). The pipeline asks the author through one callback, so the CLI and the UI only differ in how they show the plan or chapter and collect approve / revise / stop.

## With more time
- Embedding retrieval over earlier chapters for paraphrased repetition and fact lookup
- A timeline memory for fiction
- Inline diffs for the approval gates in the web UI
- Parallel drafting of independent sections
- Prompt caching for the memory block
- Regression tests with recorded real-model fixtures

## AI tools used
Claude and GitHub Copilot helped with design, code and documentation. Runtime LLM: OpenAI (Anthropic optional).
