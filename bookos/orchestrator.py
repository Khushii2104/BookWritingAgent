"""Pipeline:
  setup -> Planner -> [author approves plan] -> for each chapter:
    Outliner -> Writer (section by section, expand if short) -> Critic <-> Editor (capped loop)
    -> [author approves / revises / edits the draft file] -> Archivist -> memory -> manuscript.md
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from .agents import (Archivist, ChapterContext, Critic, Editor, Outliner, Planner, Writer, is_blocking, render_plan)
from .checks import apply_name_fixes, name_drift, word_count
from .llm import LLM
from .memory import SECTION_MARK, Memory, slugify, tail_paragraphs

logger = logging.getLogger("bookos")

REQUIRED_SETUP = ["book_type", "idea", "target_reader", "length", "style"]
BOOK_TYPES = ("fiction", "memoir", "non-fiction")

# Human-in-the-loop callback: (kind, content) -> ("approve", "") | ("revise", feedback) | ("quit", "").
Approver = Callable[[str, str], tuple[str, str]]


def auto_approver(kind: str, content: str) -> tuple[str, str]:
    return ("approve", "")


def validate_setup(setup: dict) -> dict:
    missing = [k for k in REQUIRED_SETUP if not str(setup.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Setup is missing: {', '.join(missing)}")
    book_type = re.sub(r"[\s_]+", "-", str(setup["book_type"]).strip().lower()).replace("nonfiction", "non-fiction")
    if book_type not in BOOK_TYPES:
        raise ValueError(f"book_type must be one of {', '.join(BOOK_TYPES)}")
    return {**setup, "book_type": book_type}


def parse_length(length: str) -> tuple[int | None, int | None]:
    """'40,000 words, 12 chapters' -> (40000, 12). Also understands '60k words'."""
    words = chapters = None
    m = re.search(r"(\d[\d,.]*)\s*(k)?\s*words", length, re.I)
    if m:
        words = int(float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1))
    m = re.search(r"(\d+)\s*chapters", length, re.I)
    if m:
        chapters = int(m.group(1))
    return words, chapters


def resolve_targets(setup: dict, num_chapters: int | None = None, words_per_chapter: int | None = None) -> tuple[int, int]:
    total, parsed_chapters = parse_length(str(setup.get("length", "")))
    n = int(num_chapters or setup.get("num_chapters") or parsed_chapters or 10)
    wpc = int(words_per_chapter or setup.get("words_per_chapter") or (total // n if total else 3000))
    return n, max(500, min(wpc, 12000))


def render_chapter(n: int, title: str, sections: list[str]) -> str:
    return f"# Chapter {n}: {title}\n\n" + "\n\n".join(s.strip() for s in sections) + "\n"


class Pipeline:
    def __init__(
        self,
        llm: LLM,
        out_dir: Path,
        approver: Approver = auto_approver,
        max_revisions: int = 2,
        log: Callable[[str], None] | None = None,
        words_per_chapter: int | None = None,
        section_words: int = 700,
    ):
        self.llm, self.out_dir, self.approver, self.max_revisions = llm, Path(out_dir), approver, max_revisions
        self.log = log or logger.info
        self.words_per_chapter, self.section_words = words_per_chapter, section_words
        self.planner, self.outliner, self.writer = Planner(llm), Outliner(llm), Writer(llm)
        self.critic, self.editor, self.archivist = Critic(llm), Editor(llm), Archivist(llm)

    def run(self, setup: dict, chapters_to_write: int = 2, num_chapters: int | None = None) -> Memory:
        setup = validate_setup(setup)
        n_total, wpc = resolve_targets(setup, num_chapters, self.words_per_chapter)
        n_total = max(n_total, chapters_to_write)
        mem = Memory(self.out_dir / slugify(setup["idea"]))
        mem.data["style"] = setup["style"]
        mem.save()
        mem.save_json("setup.json", setup)
        self.log(f"Book folder: {mem.root} | {n_total} chapters planned, ~{wpc} words each")

        plan = self._approved_plan(setup, mem, n_total)
        for ch in plan["chapters"][:chapters_to_write]:
            if ch["number"] in mem.approved_chapters():
                self.log(f"Chapter {ch['number']} already approved, skipping (resume).")
                continue
            if not self._write_chapter(setup, plan, ch, mem, wpc):
                self.log("Stopped by author. Progress is saved; re-run the same command to resume.")
                break
        path = mem.export_manuscript(plan.get("title", ""))
        mem.save_json("usage.json", self.llm.usage_report())
        self.log(f"Manuscript: {path}")
        return mem

    # ---- stage 1: plan; the author must approve before anything is written
    def _approved_plan(self, setup: dict, mem: Memory, n: int) -> dict:
        plan = mem.load_json("plan_approved.json")
        if plan:
            self.log("Using previously approved plan.")
            return plan
        self.log("Planner: drafting the chapter plan...")
        plan = self.planner.plan(setup, n)
        while True:
            mem.save_json("plan_draft.json", plan)
            decision, feedback = self.approver("plan", render_plan(plan))
            if decision == "approve":
                mem.save_json("plan_approved.json", plan)
                mem.seed_from_plan(plan)
                return plan
            if decision == "quit":
                raise SystemExit("Plan not approved. Nothing written.")
            self.log("Planner: revising the plan with your feedback...")
            plan = self.planner.plan(setup, n, feedback, plan)

    # ---- stage 2: one chapter, section by section, with critic/editor loop and author gate
    def _write_chapter(self, setup: dict, plan: dict, ch: dict, mem: Memory, wpc: int) -> bool:
        n = ch["number"]
        ctx = ChapterContext(setup, plan, ch, mem.context_for_writer(n), mem.known_names(), wpc)
        state = mem.load_draft_state(n) or {}
        if not state.get("outline"):
            n_sections = max(2, min(8, round(wpc / self.section_words)))
            self.log(f"Chapter {n}: Outliner planning {n_sections} sections...")
            state["outline"] = self.outliner.outline(ctx, n_sections)
            mem.save_draft_state(n, state)
        outline = state["outline"]
        sections: list[str] = state.get("sections", [])
        if sections and len(sections) < len(outline["sections"]):
            self.log(f"Chapter {n}: resuming at section {len(sections) + 1}.")

        for idx in range(len(sections), len(outline["sections"])):
            sec = outline["sections"][idx]
            self.log(f"Chapter {n}: Writer drafting section {idx + 1}/{len(outline['sections'])} '{sec['heading']}' (~{sec['target_words']} words)...")
            before = "\n\n".join(sections)
            text = self.writer.write_section(ctx, outline, idx, before)
            if word_count(text) < 0.75 * sec["target_words"]:
                self.log(f"Chapter {n}: section {idx + 1} is short ({word_count(text)} words); expanding...")
                text = self.writer.expand_section(ctx, outline, idx, before, text)
            sections.append(text)
            state["sections"] = sections
            mem.save_draft_state(n, state)

        earlier = [mem.load_chapter(i) for i in mem.approved_chapters() if i != n]
        sections, review = self._review_and_fix(ctx, outline, sections, earlier)
        while True:
            state["sections"] = sections
            mem.save_draft_state(n, state)
            draft_file = mem.write_draft(n, ch["title"], sections)
            mem.save_json(f"review_ch{n:02d}.json", review)
            decision, feedback = self.approver(f"chapter {n}", self._review_text(n, ch, sections, review, draft_file))
            # Honour any edits the author made directly in the draft file.
            sections = mem.read_draft(n) or sections
            if decision == "approve":
                break
            if decision == "quit":
                state["sections"] = sections
                mem.save_draft_state(n, state)
                return False
            self.log(f"Chapter {n}: Editor applying your feedback...")
            sections = self._revise(ctx, outline, sections, feedback, "author")
            sections, review = self._review_and_fix(ctx, outline, sections, earlier)

        final = render_chapter(n, ch["title"], sections)
        mem.save_chapter(n, final)
        self.log(f"Chapter {n}: approved ({word_count(final)} words). Archivist updating memory...")
        extracted = self.archivist.extract(ctx, final, mem.data["open_threads"])
        for note in mem.merge(n, extracted, ending=tail_paragraphs(final)):
            self.log(f"Memory: {note}")
        mem.clear_draft(n)
        return True

    def _review_and_fix(self, ctx, outline, sections, earlier) -> tuple[list[str], dict]:
        attempt = 0
        while True:
            # Misspelled known names are fixed mechanically before the LLM critic sees the text.
            drift = name_drift("\n\n".join(sections), ctx.names)
            if drift:
                sections = [apply_name_fixes(s, drift) for s in sections]
                self.log(f"Chapter {ctx.chapter['number']}: auto-fixed {len(drift)} name spelling(s).")
            self.log(f"Chapter {ctx.chapter['number']}: Critic reviewing...")
            review = self.critic.review(ctx, "\n\n".join(sections), earlier)
            if drift:
                review["auto_fixed"] = drift
            blocking = [i for i in review["issues"] if is_blocking(i)]
            review["passed"] = not blocking
            if not blocking or attempt >= self.max_revisions:
                return sections, review
            attempt += 1
            self.log(f"Chapter {ctx.chapter['number']}: {len(blocking)} issue(s) to fix, revision {attempt}/{self.max_revisions}.")
            problems = "\n".join(
                f"- [{i['severity']}/{i['type']}] \"{i.get('quote', '')}\": {i['problem']} Fix: {i.get('fix', '')}" for i in blocking
            )
            sections = self._revise(ctx, outline, sections, problems, "critic")

    def _revise(self, ctx, outline, sections: list[str], problems: str, source: str) -> list[str]:
        sections = list(sections)
        plan = self.editor.plan(ctx, sections, problems, source)
        applied = skipped = 0
        for e in plan["edits"]:
            order = ([e["section"] - 1] if e["section"] else []) + list(range(len(sections)))
            target = next((i for i in order if e["find"] in sections[i]), None)
            if target is None:
                skipped += 1
                continue
            sections[target] = sections[target].replace(e["find"], e["replace"], 1)
            applied += 1
        rewrites = {r["section"] - 1: r["instruction"] for r in plan["rewrite_sections"]}
        if source == "author" and not applied and not rewrites:
            # The editor found nothing to change; still honour the author's request everywhere.
            rewrites = {i: problems for i in range(len(sections))}
        # Rewritten sections must line up with the outline; extra sections from manual edits get the last brief.
        for i, instruction in sorted(rewrites.items()):
            idx = min(i, len(outline["sections"]) - 1)
            sections[i] = self.writer.revise_section(ctx, outline, idx, "\n\n".join(sections[:i]), sections[i], instruction)
        self.log(f"Editor: {applied} edit(s) applied, {skipped} not found, {len(rewrites)} section(s) rewritten.")
        return sections

    @staticmethod
    def _review_text(n: int, ch: dict, sections: list[str], review: dict, draft_file: Path) -> str:
        s = review["stats"]
        lines = [
            f"Draft file: {draft_file}",
            f"  (You may edit it before approving. Keep the '{SECTION_MARK}' lines.)",
            f"Words: {s['words']} / target ~{s['target_words']} | Sections: {len(sections)} | "
            f"[AUTHOR TO CONFIRM] markers: {s['author_to_confirm_markers']}",
            "Critic: passed" if review["passed"] else f"Critic: {len(review['issues'])} open issue(s) (see review_ch{n:02d}.json):",
        ]
        lines += [f"  - [{i['severity']}/{i['type']}] {i['problem']}" for i in review["issues"][:10]]
        return "\n".join(lines) + "\n\n" + render_chapter(n, ch["title"], sections)
