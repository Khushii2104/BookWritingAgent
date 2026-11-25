"""The six agents. Each has one job, its own system prompt and a validated output shape.

Planner   - book setup -> title, voice, canonical names and a chapter plan (revises on author feedback).
Outliner  - one chapter -> ordered sections (beats) with word targets, so chapters reach real length.
Writer    - drafts one section at a time with memory and the chapter-so-far; rewrites sections on request.
Critic    - rule-based checks + LLM review for consistency, repetition, invented facts, style, coverage, flow.
Editor    - turns critic/author feedback into the smallest fix: exact find/replace edits or targeted section rewrites.
Archivist - after approval, extracts summary, names, terms, facts, used stories and open threads into memory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .checks import count_placeholders, length_check, name_drift, repetition, word_count
from .llm import LLM, LLMError

GENRE_GUIDE = {
    "fiction": """FICTION craft:
- Tell the story in scenes: action, dialogue, sensory detail and interiority. Show, don't summarise.
- Keep POV and tense exactly as in the book bible. Characters have distinct voices, wants and obstacles.
- Every scene changes something. The chapter ends with forward pull.
- Invented people and places are fine, but once named they are canon: reuse exact spellings and established facts.""",
    "memoir": """MEMOIR craft:
- First person, the author's real life. Blend scene (what happened) with reflection (what it meant then and now).
- NEVER invent events, people, dates, places or quotes. Use only what the setup, plan or memory states.
- When a concrete detail is needed but unknown, write [AUTHOR TO CONFIRM: what is needed] instead of guessing.
- Keep reconstructed dialogue minimal and plausible.""",
    "non-fiction": """NON-FICTION craft:
- Build a clear, progressive argument. Each section has one job; the chapter ends with practical takeaways.
- Explain with concrete examples. Hypothetical or composite examples are fine but must be framed as such ("Imagine a team that...").
- NEVER invent statistics, studies, named experts, quotes or company stories. If a real source is needed write [AUTHOR TO CONFIRM: claim/source needed].""",
}

PLAN_STRUCTURE = {
    "fiction": "Give the book a clear arc (set-up, rising complications, midpoint turn, crisis, resolution) spread across the chapters, with subplots woven in.",
    "memoir": "Organise as chronological or thematic episodes anchored in the life events the setup describes. Only list people and places the setup mentions.",
    "non-fiction": "Build a progressive argument: the problem, the core ideas, the methods, then application and pitfalls. No two chapters cover the same ground.",
}

PLANNER_SYS = """You are the Planner on a professional book-writing team. Turn the book setup into a complete plan.
{structure}
Each chapter builds on the earlier ones and must not overlap them. Key points are concrete and specific to this book.
Fix canonical names now: list the main people/characters, places, organisations and coined terms with exact spellings.
JSON shape:
{{"title": "...", "premise": "1-2 sentences: the core story or thesis",
 "voice": {{"pov": "e.g. first person / close third / second person", "tense": "past|present", "rules": ["3-5 concrete voice rules derived from the style"]}},
 "key_entities": [{{"name": "canonical full name", "kind": "character|person|place|org|concept", "note": "one line"}}],
 "chapters": [{{"number": 1, "title": "chapter title without the word Chapter", "summary": "2-3 sentences", "key_points": ["3-5 concrete things this chapter must cover"]}}]}}"""

OUTLINER_SYS = """You are the Outliner. Break ONE chapter into sections (beats) that a writer will draft one at a time.
- Produce exactly NUM_SECTIONS sections in reading order that together cover every key point of the chapter.
- Each section has one clear job and does not overlap other sections or anything already used in earlier chapters (see MEMORY).
- The first section connects to how the previous chapter ended; the last lands this chapter and leans into the next one.
- Each brief names concrete content: the scene/event/example/argument, who is involved, and what changes or is learned.
- target_words per section should add up to TARGET_WORDS.
JSON shape: {"sections": [{"number": 1, "heading": "short heading", "brief": "3-5 sentences of concrete content", "target_words": 600}]}"""

WRITER_SYS = """You are the Writer on a professional book-writing team. You draft ONE section of a chapter at a time as polished, final prose in Markdown.
Rules:
- Write AT LEAST TARGET_WORDS words for this section. Reach the length with substance (scene, detail, example, reasoning), never filler or repetition.
- Follow the author's style, the voice rules and the target reader exactly.
- Do not write the chapter title. {headings}
- Use established names, terms and spellings EXACTLY as given in memory.
- Do not repeat stories, examples or explanations already used in earlier chapters or earlier sections; refer back briefly if needed.
- Do not contradict established facts. Continue seamlessly from the text so far; never recap it.
- Only the final section of a chapter may close with a wrap-up.
- Output only the section text: no notes, no preamble, no code fences.
{genre}"""

HEADINGS = {
    "non-fiction": "Start the section with '## <section heading>'.",
    "fiction": "No headings. Use a line containing only '* * *' for a genuine scene break.",
    "memoir": "No headings. Use a line containing only '* * *' for a genuine scene break.",
}

CRITIC_SYS = """You are the Critic. Review the draft chapter against the setup, the chapter plan and the book memory.
Check for:
- consistency: names, terms, spellings, timeline or facts that contradict MEMORY
- repetition: stories, examples or explanations already used in earlier chapters, or repeated inside this chapter
- facts: numbers, studies, quotes, people or events that look invented and are not marked [AUTHOR TO CONFIRM] (non-fiction/memoir)
- style: drift from the requested style, voice, POV or tense; wrong level for the target reader
- coverage: key points from the chapter plan that are missing or thin
- flow: abrupt transitions, a weak opening, or no connection to the previous chapter
severity: high = must fix (contradiction, invented fact, wrong name), medium = should fix, low = optional polish.
"quote" must be copied exactly from the draft (max ~25 words) so an editor can locate it.
Report only real problems; do not nitpick. passed = true when nothing is high or medium.
JSON shape: {{"passed": true, "issues": [{{"type": "consistency|repetition|facts|style|coverage|flow", "severity": "high|medium|low", "quote": "...", "problem": "...", "fix": "..."}}]}}
{genre}"""

EDITOR_SYS = """You are the Editor. You receive a chapter split into numbered sections plus problems from the critic or requests from the author.
Plan the SMALLEST set of changes that resolves them while keeping the author's voice and everything that already works.
- Prefer precise edits: "find" is an exact span copied verbatim from that section (long enough to be unique); "replace" is the new text.
- Use a section rewrite only when a change is broad (structure, missing coverage, tone across a section, more depth or length).
- Never invent facts, statistics, quotes or events. If a fix needs a fact you do not have, insert [AUTHOR TO CONFIRM: ...].
- Use established names and terms exactly.
JSON shape: {{"edits": [{{"section": 1, "find": "exact text", "replace": "new text", "reason": "..."}}],
 "rewrite_sections": [{{"section": 2, "instruction": "specific instruction for the writer"}}]}}
{genre}"""

ARCHIVIST_SYS = """You are the Archivist. Read an approved chapter and record what later chapters must stay consistent with.
- Use the canonical spelling from KNOWN_NAMES when a known entity appears. Add every new named person, place, organisation or coined term.
- facts: concrete statements now true in the book (who/what/when/where, numbers, decisions, relationships). One short fact per item.
- stories_told: a short label for every anecdote, scene, example, metaphor or explanation used, so it is not repeated.
- open_threads: the full current list of unresolved questions, promises or set-ups still to pay off (keep older ones that remain open).
JSON shape: {"summary": "4-6 sentences", "entities": [{"name": "...", "kind": "character|person|place|org|concept", "note": "one line"}],
 "terms": [{"term": "...", "definition": "..."}], "facts": ["..."], "stories_told": ["..."], "open_threads": ["..."]}"""


def setup_block(setup: dict) -> str:
    return "\n".join(f"{k.upper()}: {v}" for k, v in setup.items())


def clean_prose(text: str) -> str:
    text = re.sub(r"\A\s*```[a-zA-Z]*\n|\n```\s*\Z", "", text.strip())
    text = re.sub(r"\A# .*\n+", "", text.strip())  # the chapter title is added by the pipeline
    return text.strip()


def tail_words(text: str, n: int) -> str:
    words = text.split(" ")
    return text if len(words) <= n else "..." + " ".join(words[-n:])


def is_blocking(issue: dict) -> bool:
    return issue.get("severity", "medium") in ("high", "medium")


@dataclass
class ChapterContext:
    """Everything an agent needs to know about the chapter being written."""

    setup: dict
    plan: dict
    chapter: dict
    memory: str
    names: list[str]
    target_words: int

    @property
    def book_type(self) -> str:
        return self.setup["book_type"]

    def header(self) -> str:
        outline = "\n".join(f"{c['number']}. {c['title']}: {c['summary']}" for c in self.plan["chapters"])
        ch = self.chapter
        return (
            f"{setup_block(self.setup)}\nBOOK_TITLE: {self.plan.get('title', '')}\n\nFULL BOOK OUTLINE:\n{outline}\n\n"
            f"CHAPTER_NUMBER: {ch['number']}\nCHAPTER_TITLE: {ch['title']}\nCHAPTER_SUMMARY: {ch['summary']}\n"
            f"KEY_POINTS: {'; '.join(ch['key_points'])}\nKNOWN_NAMES: {', '.join(self.names)}\n"
        )


def normalize_plan(raw: dict) -> dict:
    chapters = [c for c in raw.get("chapters") or [] if isinstance(c, dict) and c.get("title")]
    if not chapters:
        raise LLMError(f"Planner returned no chapters: {json.dumps(raw)[:300]}")
    voice = raw.get("voice") if isinstance(raw.get("voice"), dict) else {}
    return {
        "title": str(raw.get("title", "")).strip(),
        "premise": str(raw.get("premise", "")).strip(),
        "voice": {"pov": voice.get("pov", ""), "tense": voice.get("tense", ""), "rules": list(voice.get("rules") or [])},
        "key_entities": [e for e in raw.get("key_entities") or [] if isinstance(e, dict) and e.get("name")],
        "chapters": [
            {
                "number": i,
                "title": re.sub(r"^chapter\s+\d+\s*[:.\-]\s*", "", str(c["title"]).strip(), flags=re.I),
                "summary": str(c.get("summary", "")).strip(),
                "key_points": [str(k) for k in c.get("key_points") or []],
            }
            for i, c in enumerate(chapters, 1)
        ],
    }


def render_plan(plan: dict) -> str:
    v = plan.get("voice", {})
    lines = [f"TITLE: {plan.get('title', '')}", f"PREMISE: {plan.get('premise', '')}",
             f"VOICE: {v.get('pov', '')}, {v.get('tense', '')} tense; " + "; ".join(v.get("rules", [])),
             "KEY NAMES: " + ", ".join(e["name"] for e in plan.get("key_entities", [])), ""]
    for c in plan["chapters"]:
        lines.append(f"{c['number']}. {c['title']}\n   {c['summary']}")
        lines += [f"   - {k}" for k in c["key_points"]]
    return "\n".join(lines)


class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def plan(self, setup: dict, num_chapters: int, feedback: str = "", previous: dict | None = None) -> dict:
        system = PLANNER_SYS.format(structure=PLAN_STRUCTURE[setup["book_type"]])
        user = setup_block(setup) + f"\nNUM_CHAPTERS: {num_chapters}\n"
        if previous and feedback:
            user += f"\nCURRENT PLAN:\n{json.dumps(previous, ensure_ascii=False)}\n\nAUTHOR FEEDBACK (apply it, keep everything else):\n{feedback}\n"
        return normalize_plan(self.llm.complete_json("planner", system, user))


class Outliner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def outline(self, ctx: ChapterContext, num_sections: int) -> dict:
        user = f"{ctx.header()}\nNUM_SECTIONS: {num_sections}\nTARGET_WORDS: {ctx.target_words}\n\nMEMORY:\n{ctx.memory}\n"
        raw = self.llm.complete_json("outliner", OUTLINER_SYS + "\n" + GENRE_GUIDE[ctx.book_type], user)
        sections = [s for s in raw.get("sections") or [] if isinstance(s, dict) and (s.get("brief") or s.get("heading"))]
        if not sections:
            sections = [{"heading": ctx.chapter["title"], "brief": ctx.chapter["summary"]}]
        weights = [max(int(s.get("target_words") or 0), 1) if str(s.get("target_words", "")).isdigit() else 1 for s in sections]
        total = sum(weights)
        return {"sections": [
            {
                "number": i,
                "heading": str(s.get("heading") or f"Part {i}").strip(),
                "brief": str(s.get("brief", "")).strip(),
                "target_words": max(150, round(ctx.target_words * w / total)),
            }
            for i, (s, w) in enumerate(zip(sections, weights), 1)
        ]}


class Writer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def _system(self, book_type: str) -> str:
        return WRITER_SYS.format(headings=HEADINGS[book_type], genre=GENRE_GUIDE[book_type])

    def _user(self, ctx: ChapterContext, outline: dict, idx: int, text_before: str) -> str:
        sec = outline["sections"][idx]
        plan = "\n".join(
            f"{s['number']}. {s['heading']}: {s['brief']}" + ("   <-- YOU ARE WRITING THIS ONE" if i == idx else "")
            for i, s in enumerate(outline["sections"])
        )
        so_far = tail_words(text_before, 1500) if text_before.strip() else (
            "(none - this is the opening section; connect to how the previous chapter ended, if any)"
        )
        return (
            f"{ctx.header()}\nSECTION_NUMBER: {idx + 1} of {len(outline['sections'])}\n"
            f"SECTION_HEADING: {sec['heading']}\nSECTION_BRIEF: {sec['brief']}\nTARGET_WORDS: {sec['target_words']}\n\n"
            f"CHAPTER SECTION PLAN:\n{plan}\n\nMEMORY:\n{ctx.memory}\n\nTEXT SO FAR IN THIS CHAPTER:\n{so_far}\n"
        )

    def write_section(self, ctx: ChapterContext, outline: dict, idx: int, text_before: str) -> str:
        user = self._user(ctx, outline, idx, text_before) + "\nWrite this section now.\n"
        return clean_prose(self.llm.complete("writer", self._system(ctx.book_type), user))

    def expand_section(self, ctx: ChapterContext, outline: dict, idx: int, text_before: str, draft: str) -> str:
        target = outline["sections"][idx]["target_words"]
        instruction = (
            f"This draft is {word_count(draft)} words but the target is {target}. Rewrite it as a complete, fuller version "
            f"of at least {target} words. Keep what works; deepen it with concrete detail, examples or reasoning. No padding."
        )
        return self.revise_section(ctx, outline, idx, text_before, draft, instruction)

    def revise_section(self, ctx: ChapterContext, outline: dict, idx: int, text_before: str, current: str, instruction: str) -> str:
        user = (
            self._user(ctx, outline, idx, text_before)
            + f"\nCURRENT VERSION OF THIS SECTION:\n{current}\n\nREVISION_INSTRUCTION: {instruction}\n"
            "Return the whole revised section. Keep everything the instruction does not touch, keep the voice, "
            "and keep it at least as long unless told to shorten it.\n"
        )
        return clean_prose(self.llm.complete("writer", self._system(ctx.book_type), user))


class Critic:
    def __init__(self, llm: LLM):
        self.llm = llm

    def review(self, ctx: ChapterContext, chapter_text: str, earlier_chapters: list[str]) -> dict:
        issues = (
            name_drift(chapter_text, ctx.names)
            + repetition(chapter_text, earlier_chapters)
            + length_check(chapter_text, ctx.target_words)
        )
        user = f"{ctx.header()}\nTARGET_WORDS: {ctx.target_words}\n\nMEMORY:\n{ctx.memory}\n\nDRAFT CHAPTER:\n{chapter_text}\n"
        result = self.llm.complete_json("critic", CRITIC_SYS.format(genre=GENRE_GUIDE[ctx.book_type]), user)
        for i in result.get("issues") or []:
            if isinstance(i, dict) and i.get("problem"):
                issues.append({
                    "type": str(i.get("type", "other")),
                    "severity": i.get("severity") if i.get("severity") in ("high", "medium", "low") else "medium",
                    "quote": str(i.get("quote", "")),
                    "problem": str(i["problem"]),
                    "fix": str(i.get("fix", "")),
                })
        stats = {
            "words": word_count(chapter_text),
            "target_words": ctx.target_words,
            "author_to_confirm_markers": count_placeholders(chapter_text),
        }
        return {"passed": not any(is_blocking(i) for i in issues), "issues": issues, "stats": stats}


class Editor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def plan(self, ctx: ChapterContext, sections: list[str], problems: str, source: str) -> dict:
        numbered = "\n\n".join(f"[SECTION {i}]\n{s}" for i, s in enumerate(sections, 1))
        label = "AUTHOR FEEDBACK (the author's request has priority)" if source == "author" else "CRITIC ISSUES"
        user = f"{ctx.header()}\nMEMORY:\n{ctx.memory}\n\nCHAPTER BY SECTION:\n{numbered}\n\n{label}:\n{problems}\n"
        raw = self.llm.complete_json("editor", EDITOR_SYS.format(genre=GENRE_GUIDE[ctx.book_type]), user)

        def section_no(x) -> int | None:
            try:
                n = int(x)
            except (TypeError, ValueError):
                return None
            return n if 1 <= n <= len(sections) else None

        edits = [
            {"section": section_no(e.get("section")), "find": str(e["find"]), "replace": str(e.get("replace", "")), "reason": str(e.get("reason", ""))}
            for e in raw.get("edits") or [] if isinstance(e, dict) and e.get("find")
        ]
        rewrites = [
            {"section": section_no(r.get("section")), "instruction": str(r["instruction"])}
            for r in raw.get("rewrite_sections") or [] if isinstance(r, dict) and r.get("instruction") and section_no(r.get("section"))
        ]
        return {"edits": edits, "rewrite_sections": rewrites}


class Archivist:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, ctx: ChapterContext, chapter_text: str, open_threads: list[str]) -> dict:
        threads = "\n".join(f"- {t}" for t in open_threads) or "(none)"
        user = (
            f"BOOK_TYPE: {ctx.book_type}\nCHAPTER_NUMBER: {ctx.chapter['number']}\nKNOWN_NAMES: {', '.join(ctx.names)}\n\n"
            f"OPEN_THREADS_BEFORE_THIS_CHAPTER:\n{threads}\n\nCHAPTER:\n{chapter_text}\n"
        )
        return self.llm.complete_json("archivist", ARCHIVIST_SYS, user)
