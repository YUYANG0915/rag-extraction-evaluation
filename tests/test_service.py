"""Small contract checks that use synthetic records and no model service."""

import json
import unittest

from not_every_leak_counts.service.run_service import context, worlds


class ServiceContract(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "case-1",
            "support_id": "doc-a",
            "target_indices": [0],
            "question": "Give the case label for doc-a.",
            "target_values": ["A-label"],
        }
        self.pool = [
            {"id": "doc-a", "text": "A-label; A-code", "markers": ["A-label", "A-code"]},
            {"id": "doc-b", "text": "B-label; B-code", "markers": ["B-label", "B-code"]},
        ]

    def test_worlds_preserve_only_the_requested_value(self):
        first, second = worlds(self.case, self.pool)
        self.assertEqual(first[0]["markers"][0], second[0]["markers"][0])
        self.assertNotEqual(first[0]["markers"][1], second[0]["markers"][1])
        self.assertNotEqual(first[1]["markers"], second[1]["markers"])

    def test_ideal_context_contains_only_requested_value(self):
        first, second = worlds(self.case, self.pool)
        for world in (first, second):
            evidence = context("ideal", self.case, world, {}, set(), 1)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(json.loads(evidence[0]["text"]), {"case_label": "A-label"})


if __name__ == "__main__":
    unittest.main()
