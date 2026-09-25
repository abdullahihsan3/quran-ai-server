"""Grounding data from the app's own Quran text: surah facts, explicit references, retrieval.

A model should never be the source of truth for "how many ayahs does Al-Baqarah have" --
justdeen/QuranPlus answered "8" (it is 286). Facts like that are looked up here and given to
the LLM as context it must use.
"""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import DATA_DIR
from .schemas import AyahReference

# Common alternative spellings -> surah number. The metadata's own transliterations
# ("Al-Baqara", "Aal-i-Imraan") are added automatically in __init__.
SURAH_ALIASES = {
    "fatiha": 1, "fatihah": 1, "opening": 1, "baqarah": 2, "baqara": 2, "cow": 2, "imran": 3,
    "al imran": 3, "nisa": 4, "maidah": 5, "maida": 5, "anam": 6, "araf": 7, "anfal": 8,
    "tawbah": 9, "taubah": 9, "bara'ah": 9, "yunus": 10, "hud": 11, "yusuf": 12, "joseph": 12, "rad": 13,
    "ibrahim": 14, "hijr": 15, "nahl": 16, "isra": 17, "bani israil": 17, "kahf": 18, "cave": 18,
    "maryam": 19, "mary": 19, "taha": 20, "ta ha": 20, "anbiya": 21, "hajj": 22, "muminun": 23,
    "muminoon": 23, "nur": 24, "noor": 24, "furqan": 25, "shuara": 26, "naml": 27, "qasas": 28,
    "ankabut": 29, "rum": 30, "luqman": 31, "sajdah": 32, "ahzab": 33, "azab": 33, "ahzaab": 33, "saba": 34, "fatir": 35,
    "yasin": 36, "yaseen": 36, "ya sin": 36, "ya seen": 36, "saffat": 37, "sad": 38, "zumar": 39,
    "ghafir": 40, "fussilat": 41, "shura": 42, "zukhruf": 43, "dukhan": 44, "jathiyah": 45,
    "ahqaf": 46, "muhammad": 47, "fath": 48, "hujurat": 49, "qaf": 50, "dhariyat": 51, "tur": 52,
    "najm": 53, "qamar": 54, "rahman": 55, "waqiah": 56, "waqia": 56, "hadid": 57, "mujadilah": 58,
    "hashr": 59, "mumtahanah": 60, "saff": 61, "jumuah": 62, "jumah": 62, "munafiqun": 63,
    "taghabun": 64, "talaq": 65, "tahrim": 66, "mulk": 67, "qalam": 68, "haqqah": 69, "maarij": 70,
    "nuh": 71, "noah": 71, "jinn": 72, "muzzammil": 73, "muddaththir": 74, "muddathir": 74,
    "qiyamah": 75, "insan": 76, "dahr": 76, "mursalat": 77, "naba": 78, "naziat": 79, "abasa": 80,
    "takwir": 81, "infitar": 82, "mutaffifin": 83, "inshiqaq": 84, "buruj": 85, "tariq": 86,
    "ala": 87, "a'la": 87, "ghashiyah": 88, "fajr": 89, "balad": 90, "shams": 91, "layl": 92,
    "duha": 93, "sharh": 94, "inshirah": 94, "tin": 95, "alaq": 96, "qadr": 97, "bayyinah": 98,
    "zalzalah": 99, "adiyat": 100, "qariah": 101, "takathur": 102, "asr": 103, "humazah": 104,
    "fil": 105, "feel": 105, "quraysh": 106, "quraish": 106, "maun": 107, "kawthar": 108,
    "kauthar": 108, "kafirun": 109, "kafiroon": 109, "nasr": 110, "masad": 111, "lahab": 111,
    "ikhlas": 112, "falaq": 113, "nas": 114,
}
# Surah names that are also ordinary English words / letter names: only matched after the
# word "surah"/"sura" so "the sad truth" or "the letter qaf" isn't read as a surah reference.
AMBIGUOUS_ALIASES = {"sad", "qaf", "nur", "fil", "tin", "ala", "cow", "cave", "mary", "joseph", "noah",
                     "opening", "muhammad", "hud", "saba", "nas", "asr", "fath", "rum", "tur", "jinn",
                     "insan", "dahr", "qadr", "layl", "shams", "fajr", "sharh", "feel", "saff", "mulk",
                     "najm", "qamar", "balad", "duha", "nasr", "hajj", "yunus", "yusuf", "ibrahim",
                     "nuh", "maryam", "luqman", "rahman",  # prophets / a name of Allah
                     "azab"}  # common spelling of Al-Ahzab, but also the word for "punishment"

NAMED_AYAHS = {
    "ayat al-kursi": (2, 255), "ayatul kursi": (2, 255), "ayat ul kursi": (2, 255),
    "ayat al kursi": (2, 255), "ayatal kursi": (2, 255), "verse of the throne": (2, 255),
    "throne verse": (2, 255), "ayat an-nur": (24, 35), "light verse": (24, 35), "verse of light": (24, 35),
}

REF_RE = re.compile(r"\b(\d{1,3})\s*[:.]\s*(\d{1,3})\b")
SURAH_AYAH_RE = re.compile(r"\b(?:surah|sura|surat|chapter)\s+(\d{1,3})\b(?:[^\d]{0,20}?\b(?:ayah|ayat|aya|verse)\s+(\d{1,3}))?", re.IGNORECASE)
AYAH_OF_RE = re.compile(r"\b(?:ayah|ayat|aya|verse)\s+(\d{1,3})\b", re.IGNORECASE)

STOPWORDS = set("""a an and are as at be but by did do does for from has have he her his how i in into is it its
me my of on or our quran say says said she so tell that the their them then there these they this to
us was we were what when where which who why will with you your about explain meaning mean surah ayah
verse allah god many much number ayahs verses surahs name names story""".split())

# "[2:255]" -- the citation format the LLM is instructed to use.
CITATION_RE = re.compile(r"\[(\d{1,3}):(\d{1,3})\]")


def _norm_name(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\b(?:surah|sura|surat)\b", " ", text)
    text = re.sub(r"\b(?:al|an|ar|as|at|ad|adh|ash|az|aal|i)[-\s]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"aa", "a", text)
    text = re.sub(r"ee", "i", text)
    text = re.sub(r"oo", "u", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z']+", text.lower()) if t not in STOPWORDS and len(t) > 2]


@dataclass
class GroundingContext:
    surah_facts: List[dict] = field(default_factory=list)
    references: List[AyahReference] = field(default_factory=list)
    retrieved: List[AyahReference] = field(default_factory=list)

    @property
    def has_explicit_quran_reference(self) -> bool:
        return bool(self.surah_facts or self.references)


class QuranCorpus:
    def __init__(self) -> None:
        self.surahs: List[dict] = json.loads((DATA_DIR / "surahs.json").read_text(encoding="utf-8"))
        self.arabic: Dict[str, List[str]] = json.loads((DATA_DIR / "quran_ar.json").read_text(encoding="utf-8"))
        self.english: Dict[str, List[str]] = json.loads((DATA_DIR / "quran_en.json").read_text(encoding="utf-8"))

        aliases = dict(SURAH_ALIASES)
        for s in self.surahs:
            aliases.setdefault(_norm_name(s["name"]), s["number"])
        self._alias_res: List[Tuple[re.Pattern, int]] = []
        for alias, number in sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
            body = re.escape(alias).replace(r"\ ", r"[\s-]?")
            if alias in AMBIGUOUS_ALIASES:
                pattern = rf"\b(?:surah|sura|surat)\s+(?:al|an|ar|as|at|ad|ash|az)?[-\s]?{body}\b"
            else:
                pattern = rf"\b{body}\b"
            self._alias_res.append((re.compile(pattern, re.IGNORECASE), number))

        # BM25 index over the English translation.
        self._docs: List[Tuple[int, int]] = []
        self._tfs: List[Counter] = []
        df: Counter = Counter()
        for s in self.surahs:
            for i, text in enumerate(self.english[str(s["number"])], start=1):
                tokens = _tokenize(text)
                tf = Counter(tokens)
                self._docs.append((s["number"], i))
                self._tfs.append(tf)
                df.update(tf.keys())
        n = len(self._docs)
        self._avgdl = sum(sum(tf.values()) for tf in self._tfs) / n
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    # --- lookups ------------------------------------------------------------------------

    def surah(self, number: int) -> Optional[dict]:
        return self.surahs[number - 1] if 1 <= number <= len(self.surahs) else None

    def ayah(self, surah: int, ayah: int) -> Optional[AyahReference]:
        s = self.surah(surah)
        if s is None or not 1 <= ayah <= s["ayah_count"]:
            return None
        return AyahReference(surah=surah, ayah=ayah, surah_name=s["name"],
                             arabic=self.arabic[str(surah)][ayah - 1],
                             translation=self.english[str(surah)][ayah - 1])

    def mentioned_surahs(self, question: str) -> List[int]:
        found: List[int] = []
        for m in SURAH_AYAH_RE.finditer(question):
            number = int(m.group(1))
            if self.surah(number) and number not in found:
                found.append(number)
        normalized = _norm_name(question)
        for pattern, number in self._alias_res:
            if (pattern.search(question) or pattern.search(normalized)) and number not in found:
                found.append(number)
        return found[:3]

    def explicit_references(self, question: str, surahs: List[int]) -> List[AyahReference]:
        refs: List[AyahReference] = []
        lowered = question.lower()
        for name, (s, a) in NAMED_AYAHS.items():
            if name in lowered:
                refs.append(self.ayah(s, a))
        for m in REF_RE.finditer(question):
            refs.append(self.ayah(int(m.group(1)), int(m.group(2))))
        for m in SURAH_AYAH_RE.finditer(question):
            if m.group(2):
                refs.append(self.ayah(int(m.group(1)), int(m.group(2))))
        if len(surahs) == 1:  # "ayah 5 of Al-Kahf"
            for m in AYAH_OF_RE.finditer(question):
                refs.append(self.ayah(surahs[0], int(m.group(1))))
        unique: List[AyahReference] = []
        for r in refs:
            if r is not None and all((u.surah, u.ayah) != (r.surah, r.ayah) for u in unique):
                unique.append(r)
        return unique[:5]

    def search(self, question: str, k: int = 3, min_score: float = 6.0) -> List[AyahReference]:
        """BM25 over the translation. min_score keeps weak, one-common-word matches out."""
        query = set(_tokenize(question))
        if not query:
            return []
        k1, b = 1.5, 0.75
        scored = []
        for idx, tf in enumerate(self._tfs):
            common = query.intersection(tf)
            if not common:
                continue
            dl = sum(tf.values())
            score = sum(self._idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self._avgdl)) for t in common)
            scored.append((score, idx))
        scored.sort(reverse=True)
        return [self.ayah(*self._docs[idx]) for score, idx in scored[:k] if score >= min_score]

    def ground(self, question: str) -> GroundingContext:
        surahs = self.mentioned_surahs(question)
        refs = self.explicit_references(question, surahs)
        # A question about a named surah/ayah is answered from those facts; keyword retrieval
        # would only add noise.
        retrieved = [] if (refs or surahs) else [r for r in self.search(question) if r is not None]
        return GroundingContext(surah_facts=[self.surah(n) for n in surahs], references=refs, retrieved=retrieved)

    def cited_references(self, answer: str, ctx: GroundingContext) -> List[AyahReference]:
        """References to show with an answer: the ones the user asked about, plus every
        [s:a] the answer cites -- resolved against the real text, so a hallucinated
        citation (an ayah number that doesn't exist) is dropped rather than displayed."""
        refs = list(ctx.references)
        for m in CITATION_RE.finditer(answer):
            r = self.ayah(int(m.group(1)), int(m.group(2)))
            if r is not None and all((u.surah, u.ayah) != (r.surah, r.ayah) for u in refs):
                refs.append(r)
        return refs[:5]

    @staticmethod
    def facts_for_prompt(ctx: GroundingContext) -> str:
        lines = []
        for s in ctx.surah_facts:
            lines.append(f"Surah {s['number']} {s['name']} (\"{s['english_name']}\"): {s['ayah_count']} ayahs, "
                         f"{s['revelation']}, number {s['revelation_order']} in order of revelation.")
        for r in ctx.references:
            lines.append(f"[{r.surah}:{r.ayah}] {r.arabic} -- \"{r.translation}\"")
        if ctx.retrieved:
            lines.append("Possibly relevant ayahs found by keyword search (cite only those that actually answer the question):")
            for r in ctx.retrieved:
                lines.append(f"[{r.surah}:{r.ayah}] \"{r.translation}\"")
        return "\n".join(lines)
