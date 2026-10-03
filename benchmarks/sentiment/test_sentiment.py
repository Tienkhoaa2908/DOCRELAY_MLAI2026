from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from benchmark import load_cases, score
from rules import classify


class SyntheticCorpusTests(unittest.TestCase):
    def test_corpus_is_balanced_unique_and_synthetic(self) -> None:
        cases = load_cases()
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(Counter(case["label"] for case in cases), {
            "NEGATIVE": 20,
            "NEUTRAL": 20,
            "POSITIVE": 20,
        })
        self.assertTrue(all(case["text"].strip() for case in cases))

    def test_case_labels_are_valid(self) -> None:
        cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
        self.assertTrue(
            all(case["label"] in {"NEGATIVE", "NEUTRAL", "POSITIVE"} for case in cases)
        )


class RuleBaselineTests(unittest.TestCase):
    def test_clear_emotion_phrases(self) -> None:
        examples = {
            "Cảm ơn IT, VPN đã kết nối lại sau hướng dẫn.": "POSITIVE",
            "Tôi rất bực vì VPN vẫn lỗi dù đã làm đúng hướng dẫn.": "NEGATIVE",
            "This is frustrating; I still cannot log in.": "NEGATIVE",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(classify(text)[0], expected)

    def test_technical_errors_and_negated_emotion_stay_neutral(self) -> None:
        for text in (
            "VPN không kết nối được; ứng dụng hiển thị lỗi 809 từ sáng.",
            "Tôi không bực, chỉ cần biết cách cấu hình VPN.",
            "Server staging trả 503 từ 09:30; production vẫn hoạt động.",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify(text)[0], "NEUTRAL")

    def test_metric_calculation(self) -> None:
        cases = [
            {"id": "a", "text": "", "label": "NEGATIVE"},
            {"id": "b", "text": "", "label": "NEUTRAL"},
            {"id": "c", "text": "", "label": "POSITIVE"},
        ]
        metrics = score(cases, ["NEGATIVE", "NEGATIVE", "POSITIVE"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"]["NEUTRAL"]["NEGATIVE"], 1)


if __name__ == "__main__":
    unittest.main()
