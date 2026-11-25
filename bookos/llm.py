"""LLM access behind one small interface, so agents don't care about the provider.

OpenAILLM (default) and AnthropicLLM call real APIs with timeouts, automatic retries and
per-agent token tracking. MockLLM is a deterministic offline stand-in for tests and demos.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re

log = logging.getLogger("bookos")

# Creative agents sample warmer; reviewing/extracting agents stay strict.
ROLE_TEMPERATURE = {"planner": 0.7, "outliner": 0.6, "writer": 0.85, "critic": 0.2, "editor": 0.3, "archivist": 0.1}


class LLMError(RuntimeError):
    """The model returned something the pipeline cannot use."""


def parse_json(raw: str) -> dict:
    """Parse a JSON object even if the model wrapped it in fences or added chatter."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


class LLM:
    model = "unknown"

    def __init__(self) -> None:
        self.usage: dict = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "by_role": {}}

    def _call(self, role: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""
        raise NotImplementedError

    def complete(self, role: str, system: str, user: str, json_mode: bool = False) -> str:
        text, tin, tout = self._call(role, system, user, json_mode)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += tin
        self.usage["output_tokens"] += tout
        r = self.usage["by_role"].setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        r["calls"] += 1
        r["input_tokens"] += tin
        r["output_tokens"] += tout
        return text

    def complete_json(self, role: str, system: str, user: str) -> dict:
        system += "\nReturn ONLY one valid JSON object. No prose, no code fences."
        raw = self.complete(role, system, user, json_mode=True)
        try:
            return parse_json(raw)
        except ValueError:
            log.warning("%s returned invalid JSON; retrying once", role)
        raw = self.complete(role, system, user + "\n\nYour previous reply was not valid JSON. Reply with the JSON object only.", json_mode=True)
        try:
            return parse_json(raw)
        except ValueError as e:
            raise LLMError(f"{role} did not return valid JSON: {raw[:300]!r}") from e

    def usage_report(self) -> dict:
        report = {"model": self.model, **self.usage}
        price_in = float(os.getenv("BOOK_PRICE_INPUT_PER_1M") or 0)
        price_out = float(os.getenv("BOOK_PRICE_OUTPUT_PER_1M") or 0)
        if price_in or price_out:
            cost = (self.usage["input_tokens"] * price_in + self.usage["output_tokens"] * price_out) / 1_000_000
            report["estimated_cost_usd"] = round(cost, 4)
        return report


def _client_opts() -> dict:
    return {"timeout": float(os.getenv("BOOK_TIMEOUT", "300")), "max_retries": int(os.getenv("BOOK_MAX_RETRIES", "4"))}


class OpenAILLM(LLM):
    """OpenAI Chat Completions. Default provider."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Install dependencies first: pip install -r requirements.txt") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env, or run with --mock.")
        self.client = OpenAI(**_client_opts())
        self.model = model or os.getenv("BOOK_MODEL") or "gpt-4o"
        self.max_tokens = max_tokens or int(os.getenv("BOOK_MAX_TOKENS") or 8000)
        # Reasoning models reject a custom temperature.
        self.fixed_temperature = self.model.startswith(("o1", "o3", "o4", "gpt-5"))

    def _call(self, role: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if not self.fixed_temperature:
            kwargs["temperature"] = ROLE_TEMPERATURE.get(role, 0.7)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            log.warning("%s output hit the token limit (BOOK_MAX_TOKENS=%s) and may be cut off", role, self.max_tokens)
        u = resp.usage
        return choice.message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        super().__init__()
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Install the Anthropic SDK first: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env, or run with --mock.")
        self.client = anthropic.Anthropic(**_client_opts())
        self.model = model or os.getenv("BOOK_MODEL") or "claude-sonnet-4-5"
        self.max_tokens = max_tokens or int(os.getenv("BOOK_MAX_TOKENS") or 8000)

    def _call(self, role: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=ROLE_TEMPERATURE.get(role, 0.7),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "max_tokens":
            log.warning("%s output hit the token limit (BOOK_MAX_TOKENS=%s) and may be cut off", role, self.max_tokens)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


_VOCAB = (
    "the a quiet morning plan team work desk window light rain coffee notebook idea question answer slow "
    "steady careful bright small large early late room door table meeting hour week habit focus attention "
    "signal noise silence decision promise reason example story lesson change pattern routine evening "
    "message calendar project deadline problem solution simple honest clear patient brave tired curious "
    "walked waited listened noticed wrote counted opened closed carried shared asked answered returned "
    "began finished remembered forgot decided learned turned watched held kept moved and but with into "
    "under over between after before because while during without across toward again still almost"
).split()


class MockLLM(LLM):
    """Deterministic fake that reads the structured prompts the agents build. Text is placeholder."""

    model = "mock"

    def _call(self, role: str, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        out = getattr(self, f"_{role}")(user)
        return out, (len(system) + len(user)) // 4, len(out) // 4

    @staticmethod
    def _grab(user: str, key: str, default: str = "") -> str:
        m = re.search(rf"^{key}: (.*)$", user, flags=re.M)
        return m.group(1).strip() if m else default

    def _planner(self, user: str) -> str:
        n = int(self._grab(user, "NUM_CHAPTERS", "6"))
        idea = self._grab(user, "IDEA", "an idea")
        return json.dumps({
            "title": f"A Book About {idea}",
            "premise": f"An exploration of {idea}.",
            "voice": {"pov": "close third", "tense": "past", "rules": ["short paragraphs"]},
            "key_entities": [{"name": "Maya Okafor", "kind": "character", "note": "protagonist"}],
            "chapters": [
                {"number": i, "title": f"Stage {i}", "summary": f"Develops {idea}, stage {i} of {n}.",
                 "key_points": [f"point {i}.a", f"point {i}.b"]}
                for i in range(1, n + 1)
            ],
        })

    def _outliner(self, user: str) -> str:
        n = int(self._grab(user, "NUM_SECTIONS", "3"))
        target = int(self._grab(user, "TARGET_WORDS", "900"))
        return json.dumps({"sections": [
            {"number": i, "heading": f"Part {i}", "brief": f"Beat {i} of the chapter.", "target_words": target // n}
            for i in range(1, n + 1)
        ]})

    def _writer(self, user: str) -> str:
        ch = self._grab(user, "CHAPTER_NUMBER", "1")
        sec = self._grab(user, "SECTION_NUMBER", "1").split()[0]
        target = int(self._grab(user, "TARGET_WORDS", "300"))
        names = [x for x in self._grab(user, "KNOWN_NAMES").split(", ") if x]
        who = names[0] if names else "Maya Okafor"
        rng = random.Random(f"{ch}-{sec}-{'rev' if 'REVISION_INSTRUCTION' in user else 'new'}")
        heading = f"## {self._grab(user, 'SECTION_HEADING', 'Section')}\n\n" if "BOOK_TYPE: non-fiction" in user else ""
        paragraphs, count = [f"{who} returned to the work in chapter {ch}, section {sec}."], 10
        while count < target:
            sentences = [" ".join(rng.choices(_VOCAB, k=rng.randint(9, 16))).capitalize() + "." for _ in range(5)]
            paragraphs.append(" ".join(sentences))
            count += sum(len(s.split()) for s in sentences)
        return heading + "\n\n".join(paragraphs)

    def _critic(self, user: str) -> str:
        return json.dumps({"passed": True, "issues": []})

    def _editor(self, user: str) -> str:
        if "AUTHOR FEEDBACK" in user:
            return json.dumps({"edits": [], "rewrite_sections": [{"section": 1, "instruction": "Apply the author feedback."}]})
        return json.dumps({"edits": [], "rewrite_sections": []})

    def _archivist(self, user: str) -> str:
        num = self._grab(user, "CHAPTER_NUMBER", "1")
        return json.dumps({
            "summary": f"Chapter {num}: the protagonist advances the plan.",
            "entities": [{"name": "Maya Okafor", "kind": "character", "note": "protagonist"}],
            "terms": [],
            "facts": [f"Chapter {num} takes place in a single day."],
            "stories_told": [f"opening scene of chapter {num}"],
            "open_threads": [f"what happens after chapter {num}"],
        })


def get_llm(mock: bool = False, provider: str | None = None) -> LLM:
    if mock or os.getenv("BOOK_MOCK") == "1":
        return MockLLM()
    provider = (provider or os.getenv("BOOK_PROVIDER") or "openai").lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise SystemExit(f"Unknown provider '{provider}'. Use openai or anthropic.")
