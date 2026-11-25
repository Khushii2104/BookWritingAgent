# Architecture: Book Writing Agent OS (Part 1)

```mermaid
flowchart TD
    S[/"Book setup<br/>type, idea, reader, length, style"/] --> P

    subgraph PLAN["Stage 1: Plan"]
        P["Planner<br/>title, voice, canonical names, chapter plan"] --> G1{{"Author gate 1<br/>approve / revise / quit"}}
        G1 -- "revise + feedback" --> P
    end

    G1 -- approve --> SEED[("Seed memory<br/>bible + key names")]
    SEED --> O

    subgraph CH["Stage 2: Per chapter (repeat)"]
        O["Outliner<br/>chapter to 4-8 sections with word targets"] --> W
        W["Writer<br/>one section at a time<br/>memory + chapter so far"] -- "short? expand" --> W
        W --> NF["Rule checks<br/>name drift auto-fix"]
        NF --> C["Critic<br/>rules: repetition, length<br/>LLM: consistency, facts, style, coverage, flow"]
        C -- "high/medium issues<br/>(max N rounds)" --> E["Editor<br/>exact find/replace edits<br/>or targeted section rewrites"]
        E --> NF
        C -- "passed / rounds used" --> G2{{"Author gate 2<br/>approve / revise / edit draft file / quit"}}
        G2 -- "revise + feedback" --> E
        G2 -- approve --> A["Archivist<br/>summary, names, terms, facts,<br/>stories used, open threads"]
    end

    A --> M
    M[("memory.json<br/>bible, entities + aliases, terms, facts,<br/>stories, open threads, summaries, chapter endings")]
    M -. "context for next chapter" .-> O
    M -. "context" .-> W
    M -. "context" .-> C
    A --> OUT[/"chapters/chapter_NN.md<br/>manuscript.md, usage.json"/]
```

## Files on disk (`book_data/<book>/`)

| File | Written when | Purpose |
|---|---|---|
| `setup.json` | start | the book setup |
| `plan_draft.json` / `plan_approved.json` | gate 1 | plan; approved plan is reused on resume |
| `drafts/chapter_NN.json` | after every section | resumable chapter state (outline + sections) |
| `drafts/chapter_NN.md` | before gate 2 | draft the author can read and edit directly |
| `review_chNN.json` | before gate 2 | critic findings, stats, auto-fixes |
| `chapters/chapter_NN.md` | after gate 2 | final approved chapter |
| `memory.json` | after gate 1 and each gate 2 | the book's long-term memory |
| `manuscript.md`, `usage.json` | end of run | compiled book; tokens per agent and optional cost |
