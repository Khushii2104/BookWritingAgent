#!/usr/bin/env python3
"""CLI: python main.py --setup examples/setup_nonfiction.json [--chapters 2] [--mock] [--auto-approve]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bookos.env import load_env

ENV_FILE = load_env()

from bookos.llm import LLMError, get_llm
from bookos.orchestrator import BOOK_TYPES, Pipeline, auto_approver, validate_setup

PREVIEW_CHARS = 4000


def ask_setup() -> dict:
    print("Book setup (press Enter to accept [default]).")

    def q(label: str, default: str = "", required: bool = True) -> str:
        while True:
            v = input(f"{label}{f' [{default}]' if default else ''}: ").strip() or default
            if v or not required:
                return v
            print("  This field is required.")

    while (book_type := q(f"Book type {list(BOOK_TYPES)}", "non-fiction")) not in BOOK_TYPES:
        print(f"  Choose one of: {', '.join(BOOK_TYPES)}")
    return {
        "book_type": book_type,
        "idea": q("Idea (one or two sentences)"),
        "target_reader": q("Target reader"),
        "length": q("Length", "40,000 words, 12 chapters"),
        "style": q("Style / voice"),
        "author_notes": q("Facts, people or must-include material (optional)", required=False),
    }


def cli_approver(kind: str, content: str) -> tuple[str, str]:
    bar = "=" * 78
    preview = content if len(content) <= PREVIEW_CHARS else content[:PREVIEW_CHARS] + "\n\n... [truncated: press v to view all, or open the draft file]"
    print(f"\n{bar}\nREVIEW {kind.upper()}\n{bar}\n{preview}\n{bar}")
    while True:
        a = input("[a]pprove / [r]evise with feedback / [v]iew full / [q]uit: ").strip().lower()
        if a.startswith("a"):
            return "approve", ""
        if a.startswith("r"):
            fb = input("Feedback: ").strip()
            if fb:
                return "revise", fb
            print("  Please describe what to change.")
        elif a.startswith("v"):
            print(content)
        elif a.startswith("q"):
            return "quit", ""


def main() -> int:
    p = argparse.ArgumentParser(description="Book Writing Agent OS: plan, write and check a book with a team of AI agents.")
    p.add_argument("--setup", help="JSON file with the book setup (otherwise asked interactively)")
    p.add_argument("--chapters", type=int, default=2, help="how many chapters to write in this run (default 2)")
    p.add_argument("--num-chapters", type=int, help="chapters in the plan (default: from setup length, else 10)")
    p.add_argument("--words-per-chapter", type=int, help="target words per chapter (default: from setup length, else 3000)")
    p.add_argument("--out", default="book_data", help="output/memory directory (default book_data)")
    p.add_argument("--provider", choices=["openai", "anthropic"], help="LLM provider (default: BOOK_PROVIDER or openai)")
    p.add_argument("--mock", action="store_true", help="offline deterministic LLM, no API key, placeholder text")
    p.add_argument("--auto-approve", action="store_true", help="skip human gates (demos/tests only)")
    p.add_argument("--max-revisions", type=int, default=2, help="max critic->editor rounds per chapter (default 2)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = p.parse_args()

    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        setup = json.loads(Path(a.setup).read_text(encoding="utf-8")) if a.setup else ask_setup()
        validate_setup(setup)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Invalid setup: {e}", file=sys.stderr)
        return 2

    llm = get_llm(a.mock, a.provider)
    pipe = Pipeline(
        llm, Path(a.out), auto_approver if a.auto_approve else cli_approver, a.max_revisions,
        words_per_chapter=a.words_per_chapter,
    )
    try:
        mem = pipe.run(setup, chapters_to_write=a.chapters, num_chapters=a.num_chapters)
    except LLMError as e:
        print(f"Model error: {e}\nProgress is saved; re-run the same command to resume.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run the same command to resume.", file=sys.stderr)
        return 130
    except Exception as e:  # provider/network errors after the SDK's own retries
        logging.getLogger("bookos").debug("Run failed", exc_info=True)
        print(f"Run failed: {type(e).__name__}: {e}\nProgress is saved; re-run the same command to resume "
              "(use --log-level DEBUG for a traceback).", file=sys.stderr)
        return 1
    print(f"\nDone. Output in {mem.root}\nLLM usage: {json.dumps(llm.usage_report(), indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
