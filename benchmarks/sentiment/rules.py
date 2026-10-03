"""Small, explainable Vietnamese/English sentiment baseline for the benchmark."""

from __future__ import annotations

import unicodedata


LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")

POSITIVE_PHRASES = (
    "cam on",
    "rat hai long",
    "hai long voi",
    "khong co gi de phan nan",
    "rat ro rang",
    "de hieu",
    "rat tot",
    "tuyet voi",
    "hoat dong tot",
    "xu ly tot",
    "tron tru",
    "rat huu ich",
    "huu ich",
    "than thien",
    "hieu qua",
    "chuyen nghiep",
    "deu chay",
    "cam thay yen tam",
    "toi vui",
    "rat thich",
    "appreciate",
    "thanks",
    "thank you",
    "worked perfectly",
    "helpful",
    "patient",
    "great job",
    "working again",
    "solved it",
    "not bad",
    "awesome",
    "excellent",
)

NEGATIVE_PHRASES = (
    "khong hai long",
    "khong tot",
    "khong huu ich",
    "that vong",
    "rat buc",
    "buc minh",
    "phat cau",
    "kho chiu",
    "te qua",
    "chan that",
    "rat lo",
    "qua met moi",
    "khong giai quyet",
    "phan hoi cham",
    "lien tuc bi chuyen",
    "treo suot",
    "frustrat",
    "not happy",
    "not helpful",
    "disappointed",
    "annoying",
    "annoyed",
    "useless",
    "terrible",
    "worse",
    "ignored my issue",
    "cannot believe",
    "can't believe",
    "khong the tin duoc",
    "cham that",
    "cho ca ngay",
    "dung gui cau tra loi tu dong nua",
    "phien phuc",
)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.split())


def classify(text: str) -> tuple[str, dict[str, list[str]]]:
    normalized = normalize(text)
    positive_hits = [phrase for phrase in POSITIVE_PHRASES if phrase in normalized]
    negative_hits = [phrase for phrase in NEGATIVE_PHRASES if phrase in normalized]
    if len(negative_hits) > len(positive_hits):
        label = "NEGATIVE"
    elif len(positive_hits) > len(negative_hits):
        label = "POSITIVE"
    else:
        label = "NEUTRAL"
    return label, {"positive": positive_hits, "negative": negative_hits}
