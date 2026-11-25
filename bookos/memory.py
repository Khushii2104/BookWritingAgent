"""File-based memory. Everything lives under <out>/<slug>/ as plain JSON/Markdown,
so the author can inspect, diff and fix it by hand.

memory.json holds:
  bible:           title, premise and voice rules from the approved plan
  entities:        canonical names -> kind, note, aliases, first_seen, recent updates
  terms:           glossary / spelling decisions
  facts:           things the book has established as true
  stories:         anecdotes/examples/explanations already used (so they are not repeated)
  open_threads:    unresolved questions or set-ups still to pay off
  summaries:       chapter number -> summary
  chapter_endings: chapter number -> closing passage (for a smooth hand-off)
  style:           voice from the setup

Memory is only written after a human approval, so rejected drafts never pollute it.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

SECTION_MARK = "<!-- section -->"

EMPTY = {
    "bible": {}, "entities": {}, "terms": {}, "facts": [], "stories": [], "open_threads": [],
    "summaries": {}, "chapter_endings": {}, "style": "",
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:60].strip("-") or "book"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def tail_paragraphs(text: str, max_words: int = 250) -> str:
    out, words = [], 0
    for p in reversed([p for p in text.split("\n\n") if p.strip() and not p.startswith("#")]):
        out.insert(0, p.strip())
        words += len(p.split())
        if words >= max_words:
            break
    return "\n\n".join(out)


class Memory:
    def __init__(self, root: Path):
        self.root = Path(root)
        for d in ("chapters", "drafts"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.path = self.root / "memory.json"
        self.data = self.load_json("memory.json", {})
        for k, v in EMPTY.items():
            self.data.setdefault(k, copy.deepcopy(v))
        for e in self.data["entities"].values():
            e.setdefault("aliases", [])
            e.setdefault("updates", [])

    # ---- generic files
    def save_json(self, name: str, obj) -> None:
        write_text_atomic(self.root / name, json.dumps(obj, indent=2, ensure_ascii=False))

    def load_json(self, name: str, default=None):
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def save(self) -> None:
        self.save_json("memory.json", self.data)

    # ---- approved chapters
    def chapter_path(self, n: int) -> Path:
        return self.root / "chapters" / f"chapter_{n:02d}.md"

    def save_chapter(self, n: int, text: str) -> None:
        write_text_atomic(self.chapter_path(n), text.strip() + "\n")

    def load_chapter(self, n: int) -> str:
        return self.chapter_path(n).read_text(encoding="utf-8")

    def approved_chapters(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in (self.root / "chapters").glob("chapter_*.md"))

    # ---- in-progress drafts (resumable; the author may edit the .md before approving)
    def draft_path(self, n: int) -> Path:
        return self.root / "drafts" / f"chapter_{n:02d}.md"

    def load_draft_state(self, n: int) -> dict | None:
        return self.load_json(f"drafts/chapter_{n:02d}.json")

    def save_draft_state(self, n: int, state: dict) -> None:
        self.save_json(f"drafts/chapter_{n:02d}.json", state)

    def write_draft(self, n: int, title: str, sections: list[str]) -> Path:
        body = f"\n\n{SECTION_MARK}\n\n".join(sections)
        write_text_atomic(self.draft_path(n), f"# Chapter {n}: {title}\n\n{body}\n")
        return self.draft_path(n)

    def read_draft(self, n: int) -> list[str]:
        text = self.draft_path(n).read_text(encoding="utf-8")
        text = re.sub(r"\A\s*# .*\n", "", text)
        return [s.strip() for s in text.split(SECTION_MARK) if s.strip()]

    def clear_draft(self, n: int) -> None:
        self.draft_path(n).unlink(missing_ok=True)
        (self.root / "drafts" / f"chapter_{n:02d}.json").unlink(missing_ok=True)

    # ---- entities
    def known_names(self) -> list[str]:
        return [x for name, e in self.data["entities"].items() for x in [name, *e["aliases"]]]

    def find_entity(self, name: str) -> str | None:
        ents = self.data["entities"]
        if name in ents:
            return name
        low = name.lower()
        for canon, e in ents.items():
            if canon.lower() == low or low in (a.lower() for a in e["aliases"]):
                return canon
        # "Maya" -> "Maya Okafor"; ambiguous partial matches stay separate.
        tokens = set(low.split())
        matches = [c for c in ents if tokens and (tokens < set(c.lower().split()) or set(c.lower().split()) < tokens)]
        return matches[0] if len(matches) == 1 else None

    def merge_entity(self, name: str, kind: str, note: str, chapter: int) -> str | None:
        """Add or update an entity. Returns a message when a name was folded into an existing one."""
        ents = self.data["entities"]
        canon = self.find_entity(name)
        if canon is None:
            ents[name] = {"kind": kind or "other", "note": note, "aliases": [], "first_seen": chapter, "updates": []}
            return None
        e = ents[canon]
        if note and note != e["note"] and not any(u.endswith(note) for u in e["updates"]):
            e["updates"] = (e["updates"] + [f"ch{chapter}: {note}"])[-3:]
        if name != canon and name not in e["aliases"]:
            e["aliases"].append(name)
            return f"'{name}' recorded as an alias of '{canon}'"
        return None

    def seed_from_plan(self, plan: dict) -> None:
        self.data["bible"] = {k: plan.get(k) for k in ("title", "premise", "voice") if plan.get(k)}
        for e in plan.get("key_entities", []):
            self.merge_entity(e["name"], e.get("kind", "other"), e.get("note", ""), 0)
        self.save()

    # ---- merge what the archivist extracted from an approved chapter
    def merge(self, chapter: int, extracted: dict, ending: str = "") -> list[str]:
        notes = []
        for e in extracted.get("entities", []):
            if isinstance(e, dict) and e.get("name", "").strip():
                msg = self.merge_entity(e["name"].strip(), e.get("kind", "other"), e.get("note", ""), chapter)
                if msg:
                    notes.append(msg)
        for t in extracted.get("terms", []):
            if isinstance(t, dict) and t.get("term"):
                self.data["terms"].setdefault(t["term"], t.get("definition", ""))
        for key, src in (("facts", "facts"), ("stories", "stories_told")):
            seen = {x.lower() for x in self.data[key]}
            for item in extracted.get(src, []):
                if isinstance(item, str) and item.strip() and item.lower() not in seen:
                    self.data[key].append(item.strip())
                    seen.add(item.lower())
        if isinstance(extracted.get("open_threads"), list):
            self.data["open_threads"] = [t for t in extracted["open_threads"] if isinstance(t, str)]
        self.data["summaries"][str(chapter)] = extracted.get("summary", "")
        if ending:
            self.data["chapter_endings"][str(chapter)] = ending
        self.save()
        return notes

    def context_for_writer(self, up_to: int) -> str:
        """Memory block injected into prompts for chapter `up_to`."""
        d = self.data
        b = d["bible"]
        voice = b.get("voice") or {}
        lines = ["## Book bible", f"- Title: {b.get('title', '')}", f"- Premise: {b.get('premise', '')}"]
        if voice:
            lines.append(f"- POV: {voice.get('pov', '')}; tense: {voice.get('tense', '')}")
            lines += [f"- Voice rule: {r}" for r in voice.get("rules", [])]
        lines.append("## Established names (use EXACTLY these spellings)")
        for name, e in d["entities"].items():
            alias = f"; also called {', '.join(e['aliases'])}" if e["aliases"] else ""
            latest = f" | latest: {e['updates'][-1]}" if e["updates"] else ""
            lines.append(f"- {name} ({e['kind']}{alias}): {e['note']}{latest}")
        lines += ["## Terms", *[f"- {k}: {v}" for k, v in d["terms"].items()]]
        lines += ["## Facts already established (do not contradict)", *[f"- {f}" for f in d["facts"]]]
        lines += ["## Stories, examples and explanations already used (do NOT repeat)", *[f"- {s}" for s in d["stories"]]]
        lines += ["## Open threads", *[f"- {t}" for t in d["open_threads"]]]
        lines.append("## Previous chapter summaries")
        lines += [f"- Ch {k}: {v}" for k, v in sorted(d["summaries"].items(), key=lambda kv: int(kv[0])) if 0 < int(k) < up_to]
        ending = d["chapter_endings"].get(str(up_to - 1))
        if ending:
            lines += [f"## How chapter {up_to - 1} ended (continue from here, do not repeat it)", ending]
        return "\n".join(lines)

    def export_manuscript(self, title: str = "") -> Path:
        parts = [f"# {title}"] if title else []
        parts += [self.load_chapter(n).strip() for n in self.approved_chapters()]
        path = self.root / "manuscript.md"
        write_text_atomic(path, "\n\n---\n\n".join(parts) + "\n")
        return path
