"""Deterministic quality checks. Cheap, no LLM, run before the LLM critic.

1. Name drift: a capitalised word that is almost, but not exactly, a known name.
2. Repetition: sentences that closely repeat sentences from earlier chapters (trigram index).
3. Length: chapter far below its word target.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

PLACEHOLDER = "[AUTHOR TO CONFIRM"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_placeholders(text: str) -> int:
    return text.count(PLACEHOLDER)


def name_drift(text: str, known_names: list[str]) -> list[dict]:
    parts = {p for n in known_names for p in re.findall(r"[A-Za-z][A-Za-z'-]+", n) if len(p) >= 4}
    lowercase_words = set(re.findall(r"\b[a-z]+\b", text))
    issues = []
    for w in sorted(set(re.findall(r"\b[A-Z][a-z]{3,}\b", text))):
        # Words that also appear lowercase are ordinary vocabulary, not names.
        if w in parts or w.lower() in lowercase_words:
            continue
        for p in sorted(parts):
            same_start = w[0].lower() == p[0].lower() and abs(len(w) - len(p)) <= 2
            if same_start and 0.8 <= SequenceMatcher(None, w.lower(), p.lower()).ratio() < 1.0:
                issues.append({
                    "type": "name_drift", "severity": "high", "quote": w,
                    "problem": f"'{w}' looks like a misspelling of '{p}'", "fix": f"Use '{p}'.", "replace": p,
                })
                break
    return issues


def apply_name_fixes(text: str, issues: list[dict]) -> str:
    for i in issues:
        if i.get("type") == "name_drift" and i.get("replace"):
            text = re.sub(rf"\b{re.escape(i['quote'])}\b", i["replace"], text)
    return text


def _sentences(text: str) -> list[str]:
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6]


def _trigrams(sentence: str) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", sentence.lower())
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def repetition(text: str, earlier: list[str], threshold: float = 0.6, limit: int = 10) -> list[dict]:
    prev: list[tuple[str, set]] = []
    index: dict[tuple, set[int]] = defaultdict(set)
    for chapter in earlier:
        for s in _sentences(chapter):
            grams = _trigrams(s)
            if grams:
                for g in grams:
                    index[g].add(len(prev))
                prev.append((s, grams))
    issues = []
    for s in _sentences(text):
        grams = _trigrams(s)
        candidates = Counter(i for g in grams for i in index.get(g, ()))
        for i, shared in candidates.most_common(3):
            if shared / len(grams | prev[i][1]) >= threshold:
                issues.append({
                    "type": "repetition", "severity": "medium", "quote": s,
                    "problem": f"Near-duplicate of an earlier chapter: \"{prev[i][0][:120]}\"", "fix": "Rewrite or cut.",
                })
                break
        if len(issues) >= limit:
            break
    return issues


def length_check(text: str, target_words: int, min_ratio: float = 0.7) -> list[dict]:
    words = word_count(text)
    if target_words and words < target_words * min_ratio:
        return [{
            "type": "length", "severity": "medium", "quote": "",
            "problem": f"Chapter is {words} words; target is about {target_words}.",
            "fix": "Deepen the thinnest sections with real substance (scene, example, reasoning), not padding.",
        }]
    return []
