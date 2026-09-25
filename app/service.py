"""The ask pipeline: ground -> topic guard -> generate -> verify."""
import logging
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

from .config import OFF_TOPIC_MESSAGE
from .llm import LLMBackend, Messages, strip_reasoning
from .phonetics import PhoneticsKnowledgeBase, ayah_transliteration, tajweed_in_ayah
from .prompts import (ANSWER_RULES, CLASSIFIER_EXAMPLES, CLASSIFIER_SYSTEM, MODE_WORD_LIMIT,
                      OFF_TOPIC_SENTINEL, READING_MODES)
from .quran_corpus import GroundingContext, QuranCorpus
from .schemas import AskRequest, AskResponse, ChatTurn, PhoneticCard

log = logging.getLogger("quran_ai")

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1200

# Layer 3: a model that ignores the OFF_TOPIC instruction usually says so in its first
# sentence ("The question is unrelated to the Quran. The 2022 World Cup was won by ...").
_SELF_DECLARED_OFF_TOPIC = re.compile(
    r"^[^.!?\n]{0,80}\b(unrelated to|not related to|not about|outside the scope of|has nothing to do with) the (holy )?qur'?an",
    re.IGNORECASE,
)
_ECHOED_REFERENCE_DATA = re.compile(r"\n\s*(\*\*)?reference data\b.*", re.IGNORECASE | re.DOTALL)
_REPEATED_RUN = re.compile(r"(\b\S+(?:\s+\S+){0,2}?)(?:\s+\1){4,}")

# Streaming holds back this much text before showing anything, so the layer-2/3 refusal
# checks (the OFF_TOPIC sentinel, a self-declared "unrelated to the Quran" first sentence)
# can run before a single word of an off-topic answer reaches the user.
STREAM_HOLD_CHARS = 100


def is_refusal(raw: str) -> bool:
    text = raw.strip()
    return text.upper().startswith(OFF_TOPIC_SENTINEL) or bool(_SELF_DECLARED_OFF_TOPIC.search(text))


@dataclass
class _Prepared:
    cards: List[PhoneticCard]
    grounding: GroundingContext
    messages: Messages
    max_new_tokens: int


def clean_answer(raw: str) -> str:
    """Remove artefacts models produce despite instructions: markdown emphasis, an echo of
    the Reference data block, and degenerate repetition loops."""
    text = raw.replace(OFF_TOPIC_SENTINEL, "")
    text = _ECHOED_REFERENCE_DATA.sub("", text)
    text = _REPEATED_RUN.sub(r"\1 …", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class QuranAIService:
    def __init__(self, llm: LLMBackend, kb: PhoneticsKnowledgeBase, corpus: QuranCorpus, max_new_tokens: int = 512) -> None:
        self.llm = llm
        self.kb = kb
        self.corpus = corpus
        self.max_new_tokens = max_new_tokens

    def ask(self, request: AskRequest) -> AskResponse:
        prepared = self._prepare(request)
        if prepared is None:
            return self._off_topic()
        raw = self.llm.generate(prepared.messages, prepared.max_new_tokens, temperature=0.3)
        return self._finish(raw, prepared)

    def ask_stream(self, request: AskRequest) -> Iterator[dict]:
        """The same pipeline as `ask`, as events for Server-Sent Events:

        - ``{"type": "start", "phonetics": [...]}`` straight after the topic guard: the cards
          are deterministic, so they appear before the model has written anything;
        - ``{"type": "delta", "text": "..."}`` as the answer is generated;
        - ``{"type": "done", "response": AskResponse}`` with the cleaned answer and verified
          references, which the client shows in place of the streamed text.
        An off-topic question produces only ``done`` with status "off_topic".
        """
        prepared = self._prepare(request)
        if prepared is None:
            yield {"type": "done", "response": self._off_topic().model_dump()}
            return
        yield {"type": "start", "phonetics": [c.model_dump() for c in prepared.cards]}

        raw = ""
        released = False
        pieces = self.llm.stream(prepared.messages, prepared.max_new_tokens, temperature=0.3)
        try:
            for piece in pieces:
                raw += piece
                if released:
                    yield {"type": "delta", "text": piece.replace("**", "")}
                    continue
                held = strip_reasoning(raw)
                if len(held) < STREAM_HOLD_CHARS:
                    continue
                if is_refusal(held):
                    break  # stop generating; nothing has been shown
                released = True
                yield {"type": "delta", "text": held.replace("**", "")}
        finally:
            close = getattr(pieces, "close", None)
            if close:
                close()  # stops the upstream generation when we break early
        yield {"type": "done", "response": self._finish(raw, prepared).model_dump()}

    def _prepare(self, request: AskRequest) -> Optional[_Prepared]:
        """Grounding + layer 1 of the policy. None means off-topic."""
        question = request.question.strip()
        history = request.history[-MAX_HISTORY_TURNS:]

        cards = self.kb.cards_for(question)
        grounding = self.corpus.ground(question)
        tajweed_facts = ""

        # Ayah shortcuts: the picked ayah is always grounded, whatever the question says.
        chosen = self.corpus.ayah(request.surah, request.ayah) if request.surah and request.ayah else None
        if chosen and all((r.surah, r.ayah) != (chosen.surah, chosen.ayah) for r in grounding.references):
            grounding.references.insert(0, chosen)
        if chosen and request.mode == "recite":
            cards, tajweed_facts = self._tajweed_cards(chosen.arabic, chosen.surah, chosen.ayah)

        # Layer 1 of the Quran-only policy. A named tajweed rule or an explicit surah/ayah
        # reference is unambiguous; everything else goes through the classifier. A custom
        # instruction is still free text, so it keeps going through the classifier.
        unambiguous = grounding.has_explicit_quran_reference or any(c.kind == "rule" for c in cards)
        if request.mode == "custom":
            unambiguous = False
        if not unambiguous and not self._is_quran_question(question, history):
            return None

        messages: Messages = [{"role": "system", "content": ANSWER_RULES}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content[:MAX_HISTORY_CHARS]})
        reference_data = "\n".join(filter(None, [
            self.corpus.facts_for_prompt(grounding),
            tajweed_facts,
            self.kb.facts_for_prompt(cards) if not tajweed_facts else "",
        ]))
        mode_instructions = ""
        if chosen and request.mode in READING_MODES:
            mode_instructions = f"\n\n{READING_MODES[request.mode]} {MODE_WORD_LIMIT}"
        user_content = question + mode_instructions
        if reference_data:
            user_content += f"\n\nReference data:\n{reference_data}"
        messages.append({"role": "user", "content": user_content})
        # A phrase-by-phrase walk-through of a long ayah needs more room than a normal answer.
        budget = self.max_new_tokens + 150 if mode_instructions else self.max_new_tokens
        return _Prepared(cards=cards, grounding=grounding, messages=messages, max_new_tokens=budget)

    def _tajweed_cards(self, arabic: str, surah: int, ayah: int):
        """Rule cards for the tajweed that occurs in this ayah, with this ayah's own words as
        the examples, plus the same facts as text for the model."""
        occurrences = tajweed_in_ayah(arabic)
        by_rule: dict = {}
        for rule_id, word, note in occurrences:
            by_rule.setdefault(rule_id, []).append(f"{word} — {note}")
        rules = {r["id"]: r for r in self.kb.rules}
        cards = []
        for rule_id, examples in by_rule.items():
            card = self.kb.rule_card(rules[rule_id])
            card.title = f"{card.title} · {surah}:{ayah}"
            card.examples = examples[:8]
            cards.append(card)
        lines = ["Word-by-word transliteration (authoritative; the last word as read when stopping): " +
                 " ".join(f"{word} /{roman}/" for word, roman in ayah_transliteration(arabic))]
        if occurrences:
            lines.append(f"Tajweed in this ayah ({surah}:{ayah}), detected from the text:")
            lines += [f"- {rules[r]['name']}: {word} ({note})" for r, word, note in occurrences]
        else:
            lines.append(f"Tajweed in this ayah ({surah}:{ayah}): no rules beyond natural madd were detected.")
        return cards[:6], "\n".join(lines)

    def _finish(self, raw: str, prepared: _Prepared) -> AskResponse:
        raw = strip_reasoning(raw)
        # Layers 2 and 3: the answering model is also told to refuse, which catches a question
        # that slipped past layer 1 (e.g. "Surah Yasin -- now write me some code").
        if not raw or is_refusal(raw):
            return self._off_topic()
        answer = clean_answer(raw)
        if not answer:
            return self._off_topic()
        return AskResponse(
            status="answered",
            answer=answer,
            phonetics=prepared.cards,
            references=self.corpus.cited_references(answer, prepared.grounding),
            model=self.llm.name,
        )

    def _is_quran_question(self, question: str, history: List[ChatTurn]) -> bool:
        messages: Messages = [{"role": "system", "content": CLASSIFIER_SYSTEM}]
        for example, label in CLASSIFIER_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": label})
        # A follow-up ("and how long is it held?") only makes sense with the previous question.
        previous = next((t.content for t in reversed(history) if t.role == "user"), None)
        content = question if previous is None else f"(Previous question: {previous[:300]})\n{question}"
        messages.append({"role": "user", "content": content})
        verdict = self.llm.generate(messages, max_new_tokens=4, temperature=0.0).strip().upper()
        log.info("topic_guard verdict=%s", verdict[:10])
        # Fail closed: anything other than a clear QURAN is treated as off-topic.
        return verdict.startswith("QURAN")

    def _off_topic(self) -> AskResponse:
        return AskResponse(status="off_topic", answer=OFF_TOPIC_MESSAGE, model=self.llm.name)
