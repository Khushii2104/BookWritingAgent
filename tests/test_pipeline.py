import json

import pytest

from bookos.checks import apply_name_fixes, name_drift, repetition, word_count
from bookos.llm import LLMError, MockLLM, get_llm, parse_json
from bookos.memory import Memory
from bookos.orchestrator import Pipeline, parse_length, resolve_targets, validate_setup

SETUP = {"book_type": "fiction", "idea": "A lighthouse keeper", "target_reader": "adults", "length": "short", "style": "quiet"}
QUIET = {"log": lambda *_: None, "words_per_chapter": 900}


def run(tmp_path, llm=None, approver=None, chapters=2, setup=SETUP):
    kwargs = {"approver": approver} if approver else {}
    return Pipeline(llm or MockLLM(), tmp_path, **kwargs, **QUIET).run(setup, chapters_to_write=chapters)


# ---- units
def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} done') == {"a": 1}


def test_complete_json_retries_once_then_fails():
    class Bad(MockLLM):
        def _critic(self, user):
            return "not json"
    llm = Bad()
    with pytest.raises(LLMError):
        llm.complete_json("critic", "sys", "user")
    assert llm.usage["by_role"]["critic"]["calls"] == 2


def test_setup_validation_and_normalisation():
    with pytest.raises(ValueError):
        validate_setup({**SETUP, "book_type": "poetry"})
    with pytest.raises(ValueError):
        validate_setup({"book_type": "fiction"})
    assert validate_setup({**SETUP, "book_type": "Non Fiction"})["book_type"] == "non-fiction"


def test_length_targets():
    assert parse_length("40,000 words, 12 chapters") == (40000, 12)
    assert parse_length("60k words") == (60000, None)
    assert resolve_targets({"length": "40,000 words, 12 chapters"}) == (12, 3333)
    assert resolve_targets({"length": "short"}, words_per_chapter=1500) == (10, 1500)


def test_name_drift_flags_and_fixes_misspelling():
    issues = name_drift("Maya Okafar walked home.", ["Maya Okafor"])
    assert issues and issues[0]["type"] == "name_drift"
    assert apply_name_fixes("Maya Okafar walked home.", issues) == "Maya Okafor walked home."
    assert not name_drift("Maya Okafor walked home.", ["Maya Okafor"])
    # ordinary words used elsewhere in lowercase are not names
    assert not name_drift("Many people came. There were many.", ["Maya Okafor"])


def test_repetition_flags_duplicate_sentence():
    s = "She poured the tea slowly while the rain kept falling outside."
    assert repetition(s, ["Earlier. " + s])
    assert not repetition("A totally different sentence about spaceships and tides.", [s])


def test_memory_aliases_and_utf8(tmp_path):
    mem = Memory(tmp_path)
    mem.merge(1, {"entities": [{"name": "Maya Okafor", "kind": "character", "note": "lead"}], "summary": "Café — début"})
    notes = mem.merge(2, {"entities": [{"name": "Maya", "kind": "character", "note": "now a captain"}], "summary": "x"})
    assert list(mem.data["entities"]) == ["Maya Okafor"] and "alias" in notes[0]
    assert "Maya" in mem.known_names()
    assert Memory(tmp_path).data["summaries"]["1"] == "Café — début"


def test_provider_selection(monkeypatch):
    assert isinstance(get_llm(mock=True), MockLLM)
    monkeypatch.setenv("BOOK_MOCK", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BOOK_PROVIDER", "openai")
    with pytest.raises(SystemExit) as e:
        get_llm()
    assert "OPENAI_API_KEY" in str(e.value) or "Install" in str(e.value)
    monkeypatch.setenv("BOOK_PROVIDER", "nope")
    with pytest.raises(SystemExit):
        get_llm()


# ---- pipeline
def test_end_to_end_two_chapters_with_length_and_memory(tmp_path):
    llm = MockLLM()
    mem = run(tmp_path, llm)
    assert mem.approved_chapters() == [1, 2]
    for n in (1, 2):
        assert word_count(mem.load_chapter(n)) >= 900 * 0.9
    assert "Maya Okafor" in mem.data["entities"]  # seeded from the plan
    assert set(mem.data["summaries"]) >= {"1", "2"}
    assert mem.data["chapter_endings"]["1"]
    assert "Maya Okafor" in mem.load_chapter(2)
    assert (mem.root / "manuscript.md").exists() and (mem.root / "usage.json").exists()
    assert {"planner", "outliner", "writer", "critic", "archivist"} <= set(llm.usage["by_role"])
    assert not list((mem.root / "drafts").iterdir())


def test_chapter_two_prompt_contains_chapter_one_memory(tmp_path):
    prompts = []

    class Spy(MockLLM):
        def _writer(self, user):
            prompts.append(user)
            return super()._writer(user)
    run(tmp_path, Spy())
    ch2 = [p for p in prompts if "CHAPTER_NUMBER: 2" in p]
    assert ch2 and "Ch 1: Chapter 1: the protagonist advances the plan." in ch2[0]
    assert "How chapter 1 ended" in ch2[0] and "opening scene of chapter 1" in ch2[0]


def test_plan_must_be_approved_before_writing(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, approver=lambda kind, content: ("quit", ""))
    assert not list(tmp_path.rglob("chapters/chapter_*.md"))


def test_author_revision_loop_and_resume(tmp_path):
    calls = {"n": 0}

    def approver(kind, content):
        if kind == "chapter 1" and calls["n"] == 0:
            calls["n"] += 1
            return ("revise", "make it more tense")
        return ("approve", "")
    mem = run(tmp_path, approver=approver, chapters=1)
    assert calls["n"] == 1 and mem.approved_chapters() == [1]
    mem2 = run(tmp_path, approver=lambda k, c: pytest.fail("should not ask"), chapters=1)
    assert mem2.approved_chapters() == [1]


def test_quit_mid_chapter_then_resume_keeps_draft(tmp_path):
    mem = run(tmp_path, approver=lambda k, c: ("quit", "") if k == "chapter 1" else ("approve", ""), chapters=1)
    assert mem.approved_chapters() == [] and mem.load_draft_state(1)["sections"]

    class NoWriting(MockLLM):
        def _writer(self, user):
            pytest.fail("draft should be reused, not rewritten")
    assert run(tmp_path, NoWriting(), chapters=1).approved_chapters() == [1]


def test_name_drift_is_auto_fixed(tmp_path):
    class DriftLLM(MockLLM):
        def _writer(self, user):
            return super()._writer(user).replace("Maya Okafor", "Maya Okafar")
    out = run(tmp_path, DriftLLM(), chapters=1)
    assert "Okafar" not in out.load_chapter(1) and "Maya Okafor" in out.load_chapter(1)


def test_critic_issue_is_fixed_by_editor(tmp_path):
    class Flagging(MockLLM):
        def _critic(self, user):
            if "the lamp went dark" in user:
                return json.dumps({"passed": False, "issues": [{"type": "facts", "severity": "high", "quote": "the lamp went dark",
                                                               "problem": "contradicts memory", "fix": "keep the lamp lit"}]})
            return super()._critic(user)

        def _writer(self, user):
            extra = "\n\nThen the lamp went dark." if "SECTION_NUMBER: 1 of" in user else ""
            return super()._writer(user) + extra

        def _editor(self, user):
            return json.dumps({"edits": [{"section": 1, "find": "the lamp went dark", "replace": "the lamp burned on", "reason": "x"}],
                               "rewrite_sections": []})
    mem = run(tmp_path, Flagging(), chapters=1)
    text = mem.load_chapter(1)
    assert "lamp went dark" not in text and "lamp burned on" in text
    assert mem.load_json("review_ch01.json")["passed"] is True


def test_author_edits_to_draft_file_are_kept(tmp_path):
    def approver(kind, content):
        if kind == "chapter 1":
            path = tmp_path / "a-lighthouse-keeper" / "drafts" / "chapter_01.md"
            path.write_text(path.read_text(encoding="utf-8") + "\nAUTHOR WROTE THIS LINE.\n", encoding="utf-8")
        return ("approve", "")
    mem = run(tmp_path, approver=approver, chapters=1)
    assert "AUTHOR WROTE THIS LINE." in mem.load_chapter(1)
    assert "<!-- section -->" not in mem.load_chapter(1)
