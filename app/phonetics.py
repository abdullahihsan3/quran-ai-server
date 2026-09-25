"""Deterministic pronunciation knowledge: letters (makhraj/sifat), tajweed rules, word breakdowns.

Why this is not left to the LLM: pronunciation facts are a closed, well-established set
(the 17 makharij, the sifat, the tajweed rules), and a generative model -- especially one
small enough to self-host -- will occasionally state a wrong articulation point with total
confidence. So the facts come from curated data, are returned to the app as structured
cards, and are handed to the LLM only as ground truth to explain around.
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from .config import DATA_DIR
from .schemas import PhoneticCard

# --- Arabic script ------------------------------------------------------------------------

FATHA, DAMMA, KASRA = "\u064E", "\u064F", "\u0650"
FATHATAN, DAMMATAN, KASRATAN = "\u064B", "\u064C", "\u064D"
SHADDA = "\u0651"
SUKUN = {"\u0652", "\u06E1"}  # standard sukun + Uthmani "small high dotless head of khah"
DAGGER_ALIF = "\u0670"
MARKS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670\u06E1\u06DF\u06E0\u06E2\u06E5\u06E6\u06ED")

ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u0671]")
ARABIC_TOKEN_RE = re.compile(r"[\u0621-\u064A\u0671\u064B-\u0655\u0670\u06DF-\u06ED]+")

CONSONANTS = {
    "ء": "ʼ", "أ": "ʼ", "إ": "ʼ", "ؤ": "ʼ", "ئ": "ʼ",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ",
    "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y",
}
HAMZA_CARRIERS = {"ء", "أ", "إ", "ؤ", "ئ"}

# Variant spellings folded onto the knowledge-base letter for lookup only.
LOOKUP_FOLD = {"أ": "ء", "إ": "ء", "ؤ": "ء", "ئ": "ء", "آ": "ا", "ٱ": "ا", "ى": "ا", "ة": "ه"}

VOWEL_NAMES = {FATHA: "fatḥah (a)", DAMMA: "ḍammah (u)", KASRA: "kasrah (i)",
               FATHATAN: "tanwīn fatḥ (an)", DAMMATAN: "tanwīn ḍamm (un)", KASRATAN: "tanwīn kasr (in)"}

# Words that indicate the user is asking about sound/articulation. Latin letter names such as
# "ra" or "ta" and plain-English rule aliases such as "echo" are only matched when one of these
# is present, so "Surah Ta-Ha" or "an echo of the past" don't produce phonetics cards.
PRONUNCIATION_CONTEXT = re.compile(
    r"\b(pronounc\w*|pronunciation|articulat\w*|makhraj|makharij|sound\w*|letter\w*|tajw[ie]e?d|"
    r"recit\w*|say|said|saying|throat|tongue|lips?|nasal|heavy|light|tafkh\w*|phonetic\w*|vowel\w*)\b",
    re.IGNORECASE,
)
GENERIC_RULE_ALIASES = {
    "echo", "echoing", "bouncing", "elongation", "prolongation", "long vowel", "lengthening",
    "merging", "assimilation", "hiding", "concealment", "hidden", "conversion", "changing",
    "clear pronunciation", "clarity", "heavy letters", "light letters", "emphatic", "emphasis",
    "full mouth", "stopping", "pause", "pausing", "doubled letter", "double consonant",
    "gemination", "nasal sound", "nasalization", "nasalisation", "sun letters", "moon letters",
    "mad",
}

MAX_CARDS = 6
MAX_QUOTED_ARABIC_WORDS = 3


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


class PhoneticsKnowledgeBase:
    def __init__(self) -> None:
        self.letters: Dict[str, dict] = {entry["letter"]: entry for entry in _load("arabic_letters.json")}
        self.rules: List[dict] = _load("tajweed_rules.json")
        self._letter_alias_res: List[Tuple[re.Pattern, dict]] = [
            (re.compile(rf"(?<![\w']){re.escape(alias)}(?![\w'])", re.IGNORECASE), entry)
            for entry in self.letters.values()
            for alias in sorted(entry["aliases"], key=len, reverse=True)
        ]
        self._rule_alias_res: List[Tuple[re.Pattern, bool, dict]] = [
            (re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE), alias.lower() in GENERIC_RULE_ALIASES, rule)
            for rule in self.rules
            for alias in rule["aliases"]
        ]

    # --- lookup -------------------------------------------------------------------------

    def cards_for(self, question: str) -> List[PhoneticCard]:
        """Phonetic cards for every letter, word and tajweed rule the question mentions."""
        text = question.replace("’", "'").replace("‘", "'").replace("`", "'")
        has_context = bool(PRONUNCIATION_CONTEXT.search(text))
        cards: List[PhoneticCard] = []
        seen: set = set()

        def add(key: str, card: Optional[PhoneticCard]) -> None:
            if card is not None and key not in seen and len(cards) < MAX_CARDS:
                seen.add(key)
                cards.append(card)

        # 1. Arabic script typed by the user: a single letter -> letter card, else a word card.
        #    Word breakdowns only make sense for a word QUOTED in a question ("how do I say
        #    رَبِّ?"). A question written in Arabic would otherwise get a card per word.
        tokens = ARABIC_TOKEN_RE.findall(text)
        quoting = len(tokens) <= MAX_QUOTED_ARABIC_WORDS
        for token in tokens:
            base = strip_marks(token)
            if not base:
                continue
            if len(base) == 1:
                add("L" + fold(base), self.letter_card(base))
            elif quoting:
                add("W" + token, self.word_card(token))

        # 2. Tajweed rules by name.
        for pattern, is_generic, rule in self._rule_alias_res:
            if (has_context or not is_generic) and pattern.search(text):
                add("R" + rule["id"], self.rule_card(rule))

        # 3. Letters by Latin name ("ain", "qaf") -- only in a pronunciation context.
        if has_context:
            for pattern, entry in self._letter_alias_res:
                if pattern.search(text):
                    add("L" + entry["letter"], self.letter_card(entry["letter"]))
        return cards

    def letter_card(self, letter: str) -> Optional[PhoneticCard]:
        entry = self.letters.get(fold(letter))
        if entry is None:
            return None
        return PhoneticCard(
            kind="letter", arabic=entry["letter"], title=entry["name"],
            transliteration=entry["transliteration"], ipa=entry["ipa"], makhraj=entry["makhraj"],
            characteristics=entry["characteristics"], description=f"Articulated from: {entry['makhraj']}.",
            tip=entry["tip"], common_mistake=entry.get("common_mistake"),
        )

    @staticmethod
    def rule_card(rule: dict) -> PhoneticCard:
        characteristics = [f"Letters: {' '.join(rule['letters'])}"] if rule["letters"] else []
        if rule.get("duration"):
            characteristics.append(f"Duration: {rule['duration']}")
        return PhoneticCard(
            kind="rule", arabic=rule["arabic"], title=rule["name"], description=rule["description"],
            characteristics=characteristics, tip=rule.get("tip"), examples=rule.get("examples", []),
        )

    def word_card(self, word: str) -> PhoneticCard:
        clusters = split_clusters(word)
        diacritized = any(marks for _, marks in clusters)
        breakdown = [describe_cluster(base, marks, self.letters) for base, marks in clusters]
        if diacritized:
            return PhoneticCard(
                kind="word", arabic=word, title="Word breakdown", transliteration=romanize(word),
                description="Letter-by-letter reading. The transliteration is an approximate guide; "
                            "listen to a qualified reciter for exact sound.",
                examples=breakdown,
            )
        # Without harakat a word has several valid readings -- don't guess one.
        return PhoneticCard(
            kind="word", arabic=word, title="Word breakdown",
            description="This word has no harakat (vowel marks), so its exact reading can't be "
                        "determined. Copy it from the Quran text in the app to get a full breakdown.",
            examples=breakdown,
        )

    def facts_for_prompt(self, cards: List[PhoneticCard]) -> str:
        """Plain-text ground truth given to the LLM so its prose matches the cards."""
        lines = []
        for c in cards:
            parts = [f"[{c.kind}] {c.arabic} {c.title}"]
            if c.transliteration:
                parts.append(f"transliteration: {c.transliteration}")
            if c.ipa:
                parts.append(f"IPA: {c.ipa}")
            if c.makhraj:
                parts.append(f"makhraj: {c.makhraj}")
            if c.characteristics:
                parts.append("characteristics: " + "; ".join(c.characteristics))
            if c.description and c.kind != "letter":
                parts.append(c.description)
            if c.common_mistake:
                parts.append(f"common mistake: {c.common_mistake}")
            if c.examples:
                parts.append("examples: " + "; ".join(c.examples))
            lines.append(" | ".join(parts))
        return "\n".join(lines)


# --- Arabic helpers -----------------------------------------------------------------------

def fold(letter: str) -> str:
    return LOOKUP_FOLD.get(letter, letter)


def strip_marks(text: str) -> str:
    return "".join(ch for ch in text if ARABIC_LETTER_RE.match(ch))


def split_clusters(word: str) -> List[Tuple[str, str]]:
    """[(base letter, its combining marks)] -- e.g. 'بِّ' -> ('ب', 'ِّ')."""
    clusters: List[Tuple[str, str]] = []
    for ch in word:
        if ch in MARKS and clusters:
            base, marks = clusters[-1]
            clusters[-1] = (base, marks + ch)
        elif ARABIC_LETTER_RE.match(ch) or ch == "آ":
            clusters.append((ch, ""))
    return clusters


def describe_cluster(base: str, marks: str, letters: Dict[str, dict]) -> str:
    entry = letters.get(fold(base))
    name = entry["name"] if entry else base
    parts = [name]
    if SHADDA in marks:
        parts.append("shaddah (doubled)")
    for mark, vowel in VOWEL_NAMES.items():
        if mark in marks:
            parts.append(vowel)
    if DAGGER_ALIF in marks:
        parts.append("dagger alif (long ā)")
    if SUKUN & set(marks):
        parts.append("sukūn (no vowel)")
    return f"{base}{marks} — " + " + ".join(parts)


def _vowel(marks: str) -> str:
    if FATHATAN in marks:
        return "an"
    if DAMMATAN in marks:
        return "un"
    if KASRATAN in marks:
        return "in"
    if DAGGER_ALIF in marks:
        return "ā"
    if FATHA in marks:
        return "a"
    if DAMMA in marks:
        return "u"
    if KASRA in marks:
        return "i"
    return ""


def _lengthen(out: List[str], short: str, long: str) -> bool:
    if out and out[-1].endswith(short):
        out[-1] = out[-1][: -len(short)] + long
        return True
    return False


def romanize(word: str) -> str:
    """Approximate scholarly transliteration of a fully diacritized word.

    Handles harakat, tanwin, shaddah, the three madd letters, dagger alif, hamzah carriers,
    and sun/moon-letter assimilation of the article. It is a reading aid, not a phonetic
    transcription: it does not model madd length tiers or ghunnah.
    """
    clusters = split_clusters(word)
    out: List[str] = []
    i = 0
    n = len(clusters)
    sun_letter_next = False
    while i < n:
        base, marks = clusters[i]
        vowel = _vowel(marks)

        # Definite article at the start of the word: ال / ٱل
        if i == 0 and base in ("ا", "ٱ") and n > 2 and clusters[1][0] == "ل" and not _vowel(clusters[1][1]):
            if SHADDA in clusters[2][1]:
                out.append("a")          # sun letter: the lām is silent, the next letter doubles
                sun_letter_next = clusters[2][0] != "ل"   # ٱللَّه is "allāh", not "al-lāh"
            else:
                out.append("al-")        # moon letter: the lām is pronounced
            i += 2
            continue

        if base in ("ا", "ٱ"):
            if i == 0:
                out.append(vowel or ("a" if base == "ا" else "i"))
            elif out and out[-1].endswith("an") and FATHATAN in clusters[i - 1][1]:
                pass                      # alif carrying tanwin fath (ـًا) is silent
            elif not _lengthen(out, "a", "ā"):
                out.append("ā")
        elif base == "آ":
            out.append("ʼā" if i else "ā")
        elif base == "ى":
            if vowel:
                out.append("y" + vowel)
            elif not _lengthen(out, "a", "ā"):
                out.append("ā")
        elif base == "ة":
            out.append("t" + vowel if vowel else "h")
        elif base in ("و", "ي") and not vowel and SHADDA not in marks and not (SUKUN & set(marks)) \
                and _lengthen(out, "u" if base == "و" else "i", "ū" if base == "و" else "ī"):
            pass                          # madd letter: lengthened the preceding vowel
        else:
            consonant = CONSONANTS.get(base, "")
            if base in HAMZA_CARRIERS:
                if i == 0:
                    consonant = ""        # word-initial hamzah is conventionally unwritten
                if not vowel:
                    vowel = {"إ": "i", "أ": "a", "ؤ": "u"}.get(base, "")
            if SHADDA in marks:
                # ar-raḥmān: the assimilated article is written joined to the doubled letter
                consonant = consonant + ("-" if sun_letter_next else "") + consonant
            sun_letter_next = False
            out.append(consonant + vowel)
        i += 1
    return "".join(out)


# --- Tajweed rules occurring in an ayah ---------------------------------------------------
#
# Deterministic detection over the diacritized (Simple-script) text, so the Recite reading
# mode tells the user exactly which rules occur at which words, rather than the model
# guessing. Covers the rules that are fully determined by the written text; it does not
# judge how the user actually recites (that's the QuranASR feature).

THROAT_LETTERS = set("ءهعحغخ")
IDGHAM_WITH_GHUNNAH = set("ينمو")
IDGHAM_WITHOUT_GHUNNAH = set("لر")
IKHFA_LETTERS = set("تثجدذزسشصضطظفقك")
QALQALAH_LETTERS = set("قطبجد")
TANWEEN = {FATHATAN, DAMMATAN, KASRATAN}


def _is_sakin(marks: str) -> bool:
    return bool(SUKUN & set(marks))


def _has_vowel(marks: str) -> bool:
    return any(m in marks for m in (FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, DAGGER_ALIF))


def _is_madd_letter(clusters: List[Tuple[str, str]], i: int) -> bool:
    """Alif/alif maqsura after fatha, wāw after ḍammah, yāʼ after kasrah -- with no vowel of
    their own (a wāw/yāʼ with sukun after a *fatha* is a diphthong, not madd)."""
    if i == 0:
        return False
    base, marks = clusters[i]
    prev_marks = clusters[i - 1][1]
    if _has_vowel(marks) or SHADDA in marks:
        return False
    if base in ("ا", "ى"):
        # وَالْأَرْضَ, فَاللَّهُ: after a one-letter prefix, an alif before ل is the article's
        # silent hamzat al-waṣl, not a madd letter (unlike the real madd in الضَّالِّينَ).
        if i == 1 and clusters[0][0] in ("و", "ف", "ب", "ك", "ل") and i + 1 < len(clusters) \
                and clusters[i + 1][0] == "ل":
            return False
        return FATHA in prev_marks or DAGGER_ALIF in prev_marks
    if base == "و":
        return DAMMA in prev_marks
    if base == "ي":
        return KASRA in prev_marks
    return False


def tajweed_in_ayah(text: str) -> List[Tuple[str, str, str]]:
    """[(rule_id, word, note)] for each tajweed rule occurrence in an ayah, in reading order."""
    words = [w for w in text.split() if ARABIC_LETTER_RE.search(w)]
    parsed = [split_clusters(w) for w in words]
    found: List[Tuple[str, str, str]] = []

    def next_word_start(w: int) -> Optional[str]:
        """First pronounced letter of the next word, or None at the end of the ayah or before
        hamzat al-waṣl (a bare alif), where the preceding sound is joined by a helping vowel."""
        if w + 1 >= len(parsed) or not parsed[w + 1]:
            return None
        base, marks = parsed[w + 1][0]
        if base in ("ا", "ٱ") and not marks:
            return None
        return fold(base)

    for w, clusters in enumerate(parsed):
        word = words[w]
        last_word = w == len(parsed) - 1
        n = len(clusters)

        # Lām shamsiyyah: ال + a letter carrying shaddah -- or, as in الَّذِينَ, the article's
        # lām written merged into a single doubled لّ.
        if n > 2 and clusters[0][0] in ("ا", "ٱ") and clusters[1][0] == "ل" and not _has_vowel(clusters[1][1]) \
                and SHADDA in clusters[2][1]:
            found.append(("lam_shamsiyyah", word, f"the lām is silent; {clusters[2][0]} is doubled"))
        elif n > 1 and clusters[0][0] in ("ا", "ٱ") and clusters[1][0] == "ل" and SHADDA in clusters[1][1]:
            found.append(("lam_shamsiyyah", word, "the article's lām merges into the doubled لّ"))

        for i, (base, marks) in enumerate(clusters):
            word_final = i == n - 1 or all(b in ("ا", "ى") and not m for b, m in clusters[i + 1:])
            nxt = fold(clusters[i + 1][0]) if i + 1 < n and not word_final else (None if last_word else next_word_start(w))
            same_word = i + 1 < n and not word_final

            # Nūn sākinah / tanwīn.
            is_noon_sakinah = base == "ن" and (_is_sakin(marks) or (not marks and i == n - 1))
            if (is_noon_sakinah or TANWEEN & set(marks)) and nxt:
                what = "tanwīn" if TANWEEN & set(marks) else "nūn sākinah"
                if nxt in THROAT_LETTERS:
                    found.append(("izhar", word, f"{what} before the throat letter {nxt}: pronounce the n clearly"))
                elif nxt == "ب":
                    found.append(("iqlab", word, f"{what} before ب: the n becomes a hidden m with ghunnah"))
                elif nxt in IDGHAM_WITH_GHUNNAH | IDGHAM_WITHOUT_GHUNNAH and not same_word:
                    kind = "with ghunnah" if nxt in IDGHAM_WITH_GHUNNAH else "without ghunnah"
                    found.append(("idgham", word, f"{what} merges into {nxt} ({kind})"))
                elif nxt in IKHFA_LETTERS:
                    found.append(("ikhfa", word, f"{what} before {nxt}: hidden n with ghunnah"))

            # Mīm sākinah.
            if base == "م" and _is_sakin(marks) and nxt:
                if nxt == "ب":
                    found.append(("meem_sakinah", word, "ikhfāʼ shafawī: mīm sākinah before ب, lips lightly closed with ghunnah"))
                elif nxt == "م":
                    found.append(("meem_sakinah", word, "idghām shafawī: mīm sākinah merges into م with ghunnah"))
                elif nxt in ("و", "ف"):
                    found.append(("meem_sakinah", word, f"iẓhār shafawī: keep the mīm clear before {nxt}"))

            # Ghunnah on a doubled nūn / mīm.
            if base in ("ن", "م") and SHADDA in marks:
                found.append(("ghunnah", word, f"{base}ّ: hold the nasal sound for 2 counts"))

            # Qalqalah: sukūn mid-ayah (minor); the final letter when stopping (major).
            if base in QALQALAH_LETTERS:
                if last_word and i == n - 1:
                    found.append(("qalqalah", word, f"major (kubrā) on {base} when stopping at the end of the ayah"))
                elif _is_sakin(marks):
                    found.append(("qalqalah", word, f"minor (ṣughrā) on {base}ْ"))

            # Madd beyond the natural 2 counts. The small (dagger) alif, as in أُولَٰئِكَ,
            # is a madd letter written above the preceding letter.
            if _is_madd_letter(clusters, i) or DAGGER_ALIF in marks:
                after = clusters[i + 1] if i + 1 < n else None
                if after and after[0] in HAMZA_CARRIERS:
                    found.append(("madd", word, "madd muttaṣil: hamzah after the madd letter in the same word, 4–5 counts"))
                elif after and (SHADDA in after[1] or _is_sakin(after[1])):
                    found.append(("madd", word, "madd lāzim: a permanent sukūn/shaddah after the madd letter, 6 counts"))
                elif after is None and not last_word and w + 1 < len(parsed) and parsed[w + 1] \
                        and parsed[w + 1][0][0] in HAMZA_CARRIERS:
                    found.append(("madd", word, "madd munfaṣil: the next word starts with hamzah, 2, 4 or 5 counts"))
                elif last_word and i == n - 2:
                    found.append(("madd", word, "madd ʿāriḍ lis-sukūn when stopping: 2, 4 or 6 counts"))
    return found


def pausal(roman: str) -> str:
    """The form read when stopping (waqf): the final short vowel or tanwin drops, and tanwin
    fath becomes a long ā -- أَحَدٌ /aḥadun/ is read /aḥad/ at the end of an ayah."""
    if roman.endswith("an"):
        return roman[:-2] + "ā"
    if roman.endswith(("un", "in")):
        return roman[:-2]
    if roman.endswith(("a", "i", "u")) and not roman.endswith(("ā", "ī", "ū")):
        return roman[:-1]
    return roman


def ayah_transliteration(text: str) -> List[Tuple[str, str]]:
    """[(word, transliteration)] for an ayah, the last word in its pausal form."""
    words = [w for w in text.split() if ARABIC_LETTER_RE.search(w)]
    pairs = [(w, romanize(w)) for w in words]
    if pairs:
        pairs[-1] = (pairs[-1][0], pausal(pairs[-1][1]))
    return pairs
