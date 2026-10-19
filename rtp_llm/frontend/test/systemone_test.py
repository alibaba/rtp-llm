import copy
import unittest

from rtp_llm.frontend.systemone import parse_request, to_answers


class SystemOneRequestTest(unittest.TestCase):
    def request(self, **overrides):
        value = {"state": "synthetic", "questions": {"check": {"type": "noul"}}}
        value.update(overrides)
        return value

    def test_nested_rendering_and_codec_record_shape(self):
        request = self.request(
            state={
                "document": "sample",
                "attributes": {"ready": True, "empty": None},
                "items": [{"name": "alpha", "values": [1, False]}, "plain", None],
            },
            questions={
                "pick": {
                    "type": "choice",
                    "instructions": [{"task": "choose"}, "   carefully"],
                    "criteria": {"first": None, "second": {"detail": ["x", 3]}},
                    "label": 1,
                }
            },
            extra="ignored",
        )
        before = copy.deepcopy(request)
        model, record, meta = parse_request(request)
        self.assertEqual(model, "kev-latest")
        # Golden rendering from Kev api.py at 1f04cfa28e6a708485b28a9585010b9e436b776e.
        self.assertEqual(
            record,
            {
                "state": (
                    "document: sample\nattributes:\n  ready: True\n  empty: \nitems:\n"
                    "  - name: alpha\n    values:\n      - 1\n      - False\n"
                    "  - plain\n  - "
                ),
                "questions": [
                    {
                        "instr": "- task: choose\n- carefully",
                        "options": ["first", "second: detail:\n  - x\n  - 3"],
                    }
                ],
            },
        )
        self.assertEqual(
            meta, [{"id": "pick", "type": "choice", "keys": ["first", "second"]}]
        )
        self.assertEqual(request, before)

    def test_all_json_state_types_and_optional_instructions(self):
        for state, rendered in (
            (None, ""),
            ("", ""),
            (False, "False"),
            (True, "True"),
            (0, "0"),
            (-12, "-12"),
            (1.25, "1.25"),
            ([], ""),
            ({}, ""),
            (["a", None], "- a\n- "),
        ):
            with self.subTest(state=state):
                _, record, _ = parse_request(self.request(state=state))
                self.assertEqual(record["state"], rendered)
                self.assertEqual(record["questions"][0]["instr"], "")

    def test_question_and_choice_order_and_model_are_preserved(self):
        model, record, meta = parse_request(
            self.request(
                model=" custom-model ",
                questions={
                    "z": {
                        "type": "choice",
                        "criteria": {"zulu": None, "alpha": "", "middle": False},
                    },
                    "a": {"type": "score", "criteria": ["low", "high"]},
                },
            )
        )
        self.assertEqual(model, " custom-model ")
        self.assertEqual([item["id"] for item in meta], ["z", "a"])
        self.assertEqual(meta[0]["keys"], ["zulu", "alpha", "middle"])
        self.assertEqual(
            record["questions"][0]["options"], ["zulu", "alpha", "middle: False"]
        )
        self.assertEqual(meta[1]["legend"], {"0": "low", "1": "high"})

    def test_noul_uses_false_true_order_and_no_yes_option_names(self):
        for criteria, expected in (
            (None, ["no", "yes"]),
            ({}, ["no", "yes"]),
            ({"true": "present", "false": "absent"}, ["no: absent", "yes: present"]),
            ({"true": "", "false": None, "unused": "ignored"}, ["no", "yes"]),
        ):
            with self.subTest(criteria=criteria):
                _, record, meta = parse_request(
                    self.request(
                        questions={"check": {"type": "noul", "criteria": criteria}}
                    )
                )
                self.assertEqual(record["questions"][0]["options"], expected)
                self.assertEqual(meta[0]["keys"], ["false", "true"])
                self.assertEqual(
                    to_answers([[0.2, 0.8]], meta),
                    {"check": {"type": "noul", "noul": 0.8}},
                )

    def test_question_and_option_limits(self):
        questions = {str(i): {"type": "noul"} for i in range(64)}
        self.assertEqual(len(parse_request(self.request(questions=questions))[2]), 64)
        questions["64"] = {"type": "noul"}
        with self.assertRaisesRegex(ValueError, "1..64"):
            parse_request(self.request(questions=questions))
        for count in (1, 32):
            _, record, _ = parse_request(
                self.request(
                    questions={"score": {"type": "score", "criteria": ["x"] * count}}
                )
            )
            self.assertEqual(len(record["questions"][0]["options"]), count)
        with self.assertRaisesRegex(ValueError, "1..32"):
            parse_request(
                self.request(
                    questions={"score": {"type": "score", "criteria": ["x"] * 33}}
                )
            )
        with self.assertRaisesRegex(ValueError, "1..1"):
            parse_request(self.request(), max_options=1)
        for limit in (0, -1, True, 2.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                parse_request(self.request(), max_options=limit)

    def test_invalid_request_shapes(self):
        invalid = [
            None,
            [],
            {"questions": {"q": {"type": "noul"}}},
            self.request(questions={}),
            self.request(questions=[]),
            self.request(questions={"q": []}),
            self.request(questions={"q": {}}),
            self.request(questions={"q": {"type": "other"}}),
            self.request(questions={"q": {"type": "choice", "criteria": []}}),
            self.request(questions={"q": {"type": "choice", "criteria": {}}}),
            self.request(questions={"q": {"type": "score", "criteria": {}}}),
            self.request(questions={"q": {"type": "score", "criteria": []}}),
            self.request(questions={"q": {"type": "noul", "criteria": []}}),
            self.request(state={1: "nonstring key"}),
            self.request(state=("not", "json")),
        ]
        invalid.extend(
            self.request(model=value) for value in (None, "", " \t", 1, [], {})
        )
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                parse_request(request)

    def test_empty_rendered_options_rejected_but_false_and_zero_remain_options(self):
        for description in (None, "", {}, []):
            with (
                self.subTest(description=description),
                self.assertRaisesRegex(ValueError, "nonempty"),
            ):
                parse_request(
                    self.request(
                        questions={"q": {"type": "score", "criteria": [description]}}
                    )
                )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            parse_request(
                self.request(
                    questions={"q": {"type": "choice", "criteria": {"": None}}}
                )
            )
        _, record, _ = parse_request(
            self.request(
                questions={"q": {"type": "score", "criteria": [False, 0, " "]}}
            )
        )
        self.assertEqual(record["questions"][0]["options"], ["False", "0", " "])

    def test_nonfinite_json_and_cycles_rejected(self):
        for number in (float("nan"), float("inf"), -float("inf")):
            for request in (
                self.request(state={"nested": [number]}),
                self.request(
                    questions={"q": {"type": "noul", "instructions": [number]}}
                ),
                self.request(
                    questions={"q": {"type": "choice", "criteria": {"x": number}}}
                ),
                self.request(questions={"q": {"type": "score", "criteria": [number]}}),
                self.request(extra=number),
            ):
                with (
                    self.subTest(request=request),
                    self.assertRaisesRegex(ValueError, "finite"),
                ):
                    parse_request(request)
        cycle = []
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, "circular"):
            parse_request(self.request(state=cycle))

    def test_dates_are_not_augmented(self):
        state = "Start 2026-01-01; end 2026-01-03."
        _, record, _ = parse_request(self.request(state=state))
        self.assertEqual(record["state"], state)


class SystemOneAnswersTest(unittest.TestCase):
    def metadata(self, kind, criteria):
        return parse_request(
            {"state": None, "questions": {"q": {"type": kind, "criteria": criteria}}}
        )[2]

    def test_score_expectation_and_confidence(self):
        meta = self.metadata("score", ["low", "medium", "high"])
        self.assertEqual(
            to_answers([[0.15, 0.25, 0.6]], meta),
            {
                "q": {
                    "type": "score",
                    "score": 1.45,
                    "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0.15, "1": 0.25, "2": 0.6},
                    "confidence": 0.175,
                }
            },
        )

    def test_only_confidence_normalizes_and_all_scalars_round_to_four_places(self):
        choice = self.metadata("choice", {"a": None, "b": None})
        answer = to_answers([[0.1, 0.3]], choice)["q"]
        self.assertEqual(answer["confidence"], 0.5)
        self.assertEqual(answer["probabilities"], {"a": 0.1, "b": 0.3})
        score = self.metadata("score", ["low", "high"])
        self.assertEqual(to_answers([[0.1, 0.3]], score)["q"]["score"], 0.3)
        answer = to_answers([[0.123456, 0.876544]], choice)["q"]
        self.assertEqual(answer["probabilities"], {"a": 0.1235, "b": 0.8765})
        self.assertEqual(answer["confidence"], 0.7531)
        self.assertEqual(
            to_answers([[0.123456, 0.876544]], score)["q"]["score"], 0.8765
        )

    def test_ties_choose_first_and_score_mode_uses_first_maximum(self):
        choice = self.metadata("choice", {"z": None, "a": None})
        result = to_answers([[0.5, 0.5]], choice)["q"]
        self.assertEqual(result["choice"], "z")
        self.assertEqual(result["confidence"], 0.0)
        score = self.metadata("score", ["a", "b", "c", "d"])
        result = to_answers([[0.4, 0.4, 0.15, 0.05]], score)["q"]
        self.assertEqual(result["score"], 0.85)
        self.assertEqual(result["confidence"], 0.15)

    def test_zero_mass_and_single_option_confidence(self):
        for kind, criteria in (
            ("choice", {"a": None, "b": None}),
            ("score", ["a", "b"]),
        ):
            result = to_answers([[0.0, 0.0]], self.metadata(kind, criteria))["q"]
            self.assertEqual(result["confidence"], 0.0)
        for mass in (0.0, 1.0):
            for kind, criteria in (("choice", {"only": None}), ("score", ["only"])):
                result = to_answers([[mass]], self.metadata(kind, criteria))["q"]
                self.assertEqual(result["confidence"], 1.0)
                self.assertEqual(
                    result.get("choice", result.get("score")),
                    "only" if kind == "choice" else 0.0,
                )

    def test_answer_order_follows_request(self):
        _, _, meta = parse_request(
            {"state": "", "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}}}
        )
        answers = to_answers([[0.2, 0.8], [0.9, 0.1]], meta)
        self.assertEqual(list(answers), ["z", "a"])
        self.assertEqual(answers["a"]["noul"], 0.1)

    def test_invalid_probability_shapes_or_values_are_not_silently_truncated(self):
        meta = self.metadata("choice", {"a": None, "b": None})
        invalid = (
            [],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.1]],
            [[0.1, 0.9, 0.0]],
            [[False, 1.0]],
            [[-0.1, 1.1]],
            [[float("nan"), 1.0]],
            [[float("inf"), 1.0]],
            [["0.1", 0.9]],
        )
        for probabilities in invalid:
            with (
                self.subTest(probabilities=probabilities),
                self.assertRaises(ValueError),
            ):
                to_answers(probabilities, meta)


if __name__ == "__main__":
    unittest.main()
