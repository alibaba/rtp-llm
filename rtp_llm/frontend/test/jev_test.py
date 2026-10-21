import asyncio
import copy
import math
import string
import unittest

from rtp_llm.frontend.jev import (
    JevExecutor,
    JevHeadSpec,
    JevRequestError,
    JevRowResult,
    build_row,
    groups,
    parse_decide_request,
    parse_systemone_request,
    probabilities,
)


def configuration(max_options=256):
    labels = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]
    kinds = {}
    offset = 0
    for kind, names in (
        ("noul", ["false", "true"]),
        ("score", [str(i) for i in range(6)]),
        ("choice", labels[:max_options]),
    ):
        kinds[kind] = {
            "row_indices": list(range(offset, offset + len(names))),
            "label_names": names,
            "bias": [0.0] * len(names),
            "temperature": 1.0,
        }
        offset += len(names)
    return {
        "schema_version": 1,
        "mode": "system1",
        "model_type": "jev",
        "num_labels": offset,
        "kinds": kinds,
    }


def choice(count, **extra):
    return {
        "kind": "choice",
        "question": "pick",
        "options": [str(i) for i in range(count)],
        **extra,
    }


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.spec = JevHeadSpec.from_config(configuration())

    def test_exact_text_and_json_framing(self):
        request = parse_decide_request(
            choice(2, state={"中文": [1, "裙子"], "ready": True}), self.spec
        )
        row = build_row(request, self.spec, request.options)
        self.assertEqual(
            "".join(part["text"] for part in row.content),
            '[kind] choice\n[state] {"中文": [1, "裙子"], "ready": true}'
            "\n[question] pick\n[options]\nA) 0\nB) 1\n[decision]:",
        )
        self.assertFalse(row.has_images)

    def test_native_noul_and_score_options_ignore_supplied_options(self):
        for kind, options in (("noul", ["false", "true"]), ("score", list("012345"))):
            parsed = parse_decide_request(
                {"kind": kind, "question": "q", "options": ["ignored"]}, self.spec
            )
            self.assertEqual(list(parsed.options), options)
            text = build_row(parsed, self.spec, parsed.options).content[-1]["text"]
            self.assertEqual(
                text,
                "\n[question] q\n[options]\n" + "\n".join(options) + "\n[decision]:",
            )

    def test_multimodal_order_and_input_not_mutated(self):
        state = [
            "before",
            {"image": "data:image/png;base64,AAA="},
            {"type": "text", "text": "between"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/2.png", "detail": "high"},
            },
            {"metadata": "after"},
            3,
        ]
        original = copy.deepcopy(state)
        parsed = parse_decide_request(choice(2, state=state), self.spec)
        row = build_row(parsed, self.spec, parsed.options)
        self.assertTrue(row.has_images)
        self.assertEqual(
            [part["type"] for part in row.content],
            ["text", "text", "image_url", "text", "image_url", "text", "text", "text"],
        )
        self.assertEqual(row.content[4], state[3])
        self.assertEqual(row.content[5]["text"], '{"metadata": "after"}')
        row.content[4]["image_url"]["detail"] = "low"
        self.assertEqual(state, original)
        self.assertEqual(parsed.parts[3]["image_url"]["detail"], "high")

    def test_option_boundaries_and_actual_label_limit(self):
        for count in (2, 16, 17, 128, 129, 256):
            self.assertEqual(
                len(parse_decide_request(choice(count), self.spec).options), count
            )
        for count in (0, 1, 257):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "2-256"):
                parse_decide_request(choice(count), self.spec)
        small = JevHeadSpec.from_config(configuration(26))
        with self.assertRaisesRegex(ValueError, "2-26"):
            parse_decide_request(choice(27), small)

    def test_schema_and_nonfinite_inputs_fail_before_execution(self):
        invalid = [
            None,
            [],
            {},
            choice(2, kind="other"),
            choice(2, question=None),
            choice(2, options=[1, 2]),
            choice(2, strategy="invalid"),
            choice(2, state=None),
            choice(2, state=float("nan")),
            choice(2, state={1: "bad"}),
            choice(2, state=[{"type": "text"}]),
            choice(2, state=[{"image": 3}]),
            choice(2, state=[{"type": "image_url", "image_url": "url"}]),
            choice(2, state=[{"type": "video_url", "video_url": "url"}]),
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                parse_decide_request(request, self.spec)
        state = []
        state.append(state)
        with self.assertRaisesRegex(ValueError, "circular"):
            parse_decide_request(choice(2, state=state), self.spec)
        with self.assertRaisesRegex(ValueError, "max_images"):
            parse_decide_request(choice(2, state=[{"image": "url"}] * 9), self.spec)

    def test_system2_controls_rejected_explicitly(self):
        for controls in (
            {"thinking": "auto"},
            {"thinking": "on"},
            {"system2_only": True},
            {"system2_only": "false"},
            {"think_budget": 10},
            {"reasoning_effort": "low"},
            {"return_reasoning": True},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                parse_decide_request(choice(2, **controls), self.spec)
        for value in ("default", "off"):
            self.assertEqual(
                parse_decide_request(choice(2, thinking=value), self.spec).strategy,
                "single",
            )

    def test_kind_selection_bias_temperature_and_no_padded_labels(self):
        config = configuration()
        config["kinds"]["choice"]["bias"] = [2.0, 0.0] + [0.0] * 14
        config["kinds"]["choice"]["temperature"] = 2.0
        spec = JevHeadSpec.from_config(config)
        logits = [10000.0] * spec.num_labels
        logits[8], logits[9] = 0.0, 2.0
        self.assertEqual(
            probabilities(JevRowResult(logits, 10), "choice", 2, spec), [0.5, 0.5]
        )
        self.assertEqual(spec.kinds["choice"].bias[16:], (0.0,) * 240)
        for logits in (
            [0.0],
            [float("nan")] * spec.num_labels,
            [float("inf")] * spec.num_labels,
        ):
            with self.assertRaises(ValueError):
                probabilities(JevRowResult(logits, 10), "choice", 2, spec)

    def test_invalid_head_metadata_rejected(self):
        cases = []
        for key, value in (
            ("schema_version", 2),
            ("mode", "system2"),
            ("num_labels", True),
            ("default_strategy", "auto"),
            ("max_images", 0),
            ("protocol", "other"),
        ):
            config = configuration()
            config[key] = value
            cases.append(config)
        for key, value in (
            ("row_indices", [8, 8]),
            ("temperature", 0),
            ("temperature", float("nan")),
            ("bias", [0]),
            ("label_names", ["same"] * 256),
            ("max_options", 255),
            ("native_slots", 17),
        ):
            config = configuration()
            config["kinds"]["choice"][key] = value
            cases.append(config)
        config = configuration()
        config["kinds"]["noul"]["label_names"].reverse()
        cases.append(config)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                JevHeadSpec.from_config(config)

    def test_systemone_preserves_ids_descriptions_and_custom_scores(self):
        raw = {
            "model": " business-model ",
            "state": ["x", {"image": "url"}],
            "questions": {
                "pick": {
                    "type": "choice",
                    "criteria": {"red": "warm", "blue": {"tag": "cold"}},
                },
                "judge": {
                    "type": "noul",
                    "instructions": "check",
                    "criteria": {"true": "matches", "false": "does not match"},
                },
                "rate": {"type": "score", "criteria": ["bad", "good", "great"]},
            },
        }
        before = copy.deepcopy(raw)
        model, records, metadata = parse_systemone_request(raw, self.spec)
        self.assertEqual(model, " business-model ")
        self.assertEqual([item["id"] for item in metadata], ["pick", "judge", "rate"])
        self.assertEqual(records[0].options, ("red: warm", "blue: tag: cold"))
        self.assertEqual(records[1].options, ("false", "true"))
        self.assertEqual(
            records[1].question,
            "check\n[criteria]\nfalse: does not match\ntrue: matches",
        )
        self.assertEqual(records[2].kind, "choice")
        self.assertEqual(metadata[2]["effective_kind"], "choice")
        self.assertEqual(metadata[2]["adaptation"], "score_rubric_as_choice")
        self.assertTrue(all(record.has_images for record in records))
        self.assertEqual(raw, before)

    def test_systemone_limits_singletons_and_controls(self):
        for kind, criteria in (("choice", {"only": None}), ("score", ["only"])):
            _, records, metadata = parse_systemone_request(
                {
                    "state": None,
                    "questions": {"q": {"type": kind, "criteria": criteria}},
                },
                self.spec,
            )
            self.assertTrue(records[0].deterministic)
            self.assertEqual(metadata[0]["adaptation"], "deterministic")
        for request in (
            {"state": "", "questions": {}},
            {"state": "", "questions": {str(i): {"type": "noul"} for i in range(65)}},
            {
                "state": "",
                "questions": {"q": {"type": "score", "criteria": ["x"] * 257}},
            },
            {"state": "", "questions": {"q": {"type": "noul", "thinking": "on"}}},
        ):
            with self.assertRaises(ValueError):
                parse_systemone_request(request, self.spec)


class StrategyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.spec = JevHeadSpec.from_config(configuration())
        self.rows = []

    async def uniform(self, row):
        self.rows.append(row)
        return JevRowResult([0.0] * self.spec.num_labels, 10)

    async def identity(self, row):
        self.rows.append(row)
        logits = [0.0] * self.spec.num_labels
        for index, option in enumerate(row.options):
            logits[8 + index] = float(option) / 10
        return JevRowResult(logits, 10)

    async def test_single_all_boundaries_one_real_request_and_zero_decode(self):
        executor = JevExecutor(self.spec, self.uniform)
        for count in (2, 16, 17, 128, 129, 256):
            result = await executor.decide(choice(count))
            self.assertEqual(result["num_model_requests"], 1)
            self.assertEqual(result["probabilities"], [1 / count] * count)
            self.assertEqual(result["choice_index"], 0)
            self.assertEqual(
                result["usage"],
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                    "vision_tokens": 0,
                    "cache_reuse_tokens": 0,
                },
            )
            self.assertEqual(
                result["adaptation"], "native" if count <= 16 else "single"
            )
            self.assertNotIn("thinking", result)

    async def test_native_kinds_and_small_choice_do_not_expand_strategies(self):
        executor = JevExecutor(self.spec, self.uniform)
        for kind in ("noul", "score", "choice"):
            for strategy in ("auto", "single", "permute", "tournament"):
                result = await executor.decide(choice(16, kind=kind, strategy=strategy))
                self.assertEqual(result["num_model_requests"], 1)
                self.assertEqual(result["adaptation"], "native")

    async def test_permutation_maps_full_distribution_back_to_original_options(self):
        executor = JevExecutor(self.spec, self.identity, max_concurrency=2)
        for count in (17, 128, 129, 256):
            result = await executor.decide(choice(count, strategy="permute"))
            weights = [math.exp(i / 10 - (count - 1) / 10) for i in range(count)]
            expected = [w / sum(weights) for w in weights]
            for actual, target in zip(result["probabilities"], expected):
                self.assertAlmostEqual(actual, target, places=14)
            self.assertEqual(result["choice_index"], count - 1)
            self.assertEqual(result["num_model_requests"], 4)
            self.assertEqual(result["usage"]["prompt_tokens"], 40)

    async def test_permutation_seed_depends_on_question_not_global_random(self):
        executor = JevExecutor(self.spec, self.uniform)
        await executor.decide(choice(17, strategy="permute", question="中文问题"))
        first = [row.options for row in self.rows]
        self.rows.clear()
        await executor.decide(choice(17, strategy="permute", question="中文问题"))
        self.assertEqual(first, [row.options for row in self.rows])
        self.assertEqual(first[0], tuple(str(i) for i in range(17)))
        # random.Random(crc32(UTF-8 question)), first official sample order.
        self.assertEqual(
            first[1],
            tuple(
                str(i)
                for i in [2, 12, 9, 14, 16, 13, 15, 6, 10, 0, 5, 11, 8, 4, 1, 7, 3]
            ),
        )
        self.rows.clear()
        await executor.decide(choice(17, strategy="permute", question="another"))
        self.assertNotEqual(first[1:], [row.options for row in self.rows][1:])

    async def test_tournament_ties_and_all_losers_keep_probability(self):
        executor = JevExecutor(self.spec, self.uniform)
        result = await executor.decide(choice(17, strategy="tournament"))
        self.assertEqual([len(row.options) for row in self.rows], [9, 8, 16])
        self.assertEqual(
            self.rows[-1].options, tuple(str(i) for i in range(17) if i != 8)
        )
        # Official reconstruction: 16 finalist masses 17/288, loser mass 1/18.
        self.assertAlmostEqual(result["probabilities"][8], 1 / 18)
        for i in range(17):
            if i != 8:
                self.assertAlmostEqual(result["probabilities"][i], 17 / 288)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0)
        self.assertEqual(result["choice_index"], 0)
        self.assertEqual(result["num_model_requests"], 3)

    async def test_tournament_wide_boundaries(self):
        executor = JevExecutor(self.spec, self.uniform)
        for count in (128, 129, 256):
            self.rows.clear()
            result = await executor.decide(choice(count, strategy="tournament"))
            self.assertEqual(result["num_model_requests"], math.ceil(count / 16) + 1)
            self.assertEqual(len(result["probabilities"]), count)
            self.assertTrue(all(value > 0 for value in result["probabilities"]))
            self.assertAlmostEqual(sum(result["probabilities"]), 1.0)
            self.assertEqual(len(self.rows[-1].options), 16)
        self.assertEqual(result["probabilities"], [1 / 256] * 256)
        self.assertEqual(
            self.rows[-1].options, tuple(str(i) for i in range(0, 256, 16))
        )

    async def test_info_does_not_claim_system2_or_fake_vllm_counts(self):
        info = JevExecutor(self.spec, self.uniform, max_concurrency=3).info()
        self.assertEqual(info["max_options"], 256)
        self.assertEqual(info["native_choice_options"], 16)
        self.assertEqual(info["defaults"]["thinking"], "off")
        self.assertFalse(info["reasoning_controls"]["enable_thinking"])
        self.assertEqual(info["accounting"]["completion_tokens"], 0)
        self.assertEqual(info["capabilities"]["max_concurrent_rows"], 3)

    async def test_systemone_uses_choice_calibration_and_preserves_answer_shape(self):
        config = configuration()
        config["kinds"]["choice"]["temperature"] = 2.0
        config["kinds"]["score"]["temperature"] = 10.0
        spec = JevHeadSpec.from_config(config)

        async def run(row):
            self.rows.append(row)
            logits = [0.0] * spec.num_labels
            logits[8], logits[9], logits[10] = (
                2 * math.log(0.15),
                2 * math.log(0.25),
                2 * math.log(0.6),
            )
            return JevRowResult(logits, 7)

        result = await JevExecutor(spec, run).systemone(
            {
                "state": ["state", {"image": "url"}],
                "questions": {
                    "rate": {"type": "score", "criteria": ["bad", "good", "great"]},
                    "only": {"type": "choice", "criteria": {"unique": "single"}},
                },
            }
        )
        self.assertEqual(result["answers"]["rate"]["score"], 1.45)
        self.assertEqual(
            result["answers"]["rate"]["probabilities"], {"0": 0.15, "1": 0.25, "2": 0.6}
        )
        self.assertEqual(result["answers"]["only"]["choice"], "unique")
        self.assertEqual(result["answers"]["only"]["probabilities"], {"unique": 1.0})
        self.assertEqual(result["num_model_requests"], 1)
        self.assertEqual(result["jev"]["questions"]["rate"]["effective_kind"], "choice")
        self.assertEqual(self.rows[0].kind, "choice")
        self.assertTrue(self.rows[0].has_images)

    async def test_systemone_all_singletons_need_no_model(self):
        result = await JevExecutor(self.spec, self.uniform).systemone(
            {
                "state": "x",
                "questions": {
                    "c": {"type": "choice", "criteria": {"one": "only"}},
                    "s": {"type": "score", "criteria": ["only"]},
                },
            }
        )
        self.assertEqual(result["num_model_requests"], 0)
        self.assertEqual(result["usage"]["total_tokens"], 0)
        self.assertEqual(result["answers"]["s"]["score"], 0)
        self.assertEqual(self.rows, [])

    async def test_shared_concurrency_bound_across_questions_and_requests(self):
        active, peak = 0, 0

        async def run(row):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return JevRowResult([0.0] * self.spec.num_labels, 10)
            finally:
                active -= 1

        executor = JevExecutor(self.spec, run, max_concurrency=2)
        multi = {
            "state": "",
            "strategy": "tournament",
            "questions": {
                str(i): {
                    "type": "choice",
                    "criteria": {str(j): "x" for j in range(256)},
                }
                for i in range(3)
            },
        }
        results = await asyncio.gather(
            executor.systemone(multi), executor.decide(choice(256, strategy="permute"))
        )
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual([item["num_model_requests"] for item in results], [51, 4])
        self.assertEqual(
            results[0]["jev"]["questions"]["0"]["adaptation"], "tournament"
        )

    async def test_cancel_drains_all_rows_and_releases_semaphore(self):
        active = set()
        started = asyncio.Event()

        async def run(row):
            task = asyncio.current_task()
            active.add(task)
            if len(active) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active.remove(task)

        executor = JevExecutor(self.spec, run, max_concurrency=2)
        task = asyncio.create_task(executor.decide(choice(256, strategy="tournament")))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, set())
        executor.run_row = self.uniform
        result = await executor.decide(choice(2))
        self.assertEqual(result["num_model_requests"], 1)

    async def test_timeout_and_failure_cancel_siblings(self):
        for mode in ("timeout", "failure"):
            active = set()
            all_started = asyncio.Event()

            async def run(row):
                task = asyncio.current_task()
                active.add(task)
                if len(active) == 2:
                    all_started.set()
                try:
                    await all_started.wait()
                    if mode == "failure" and row.options[0] == "0":
                        raise RuntimeError("engine failure")
                    await asyncio.Event().wait()
                finally:
                    active.remove(task)

            executor = JevExecutor(
                self.spec, run, max_concurrency=2, timeout_seconds=0.05
            )
            with self.assertRaises(
                asyncio.TimeoutError if mode == "timeout" else RuntimeError
            ):
                await executor.decide(choice(256, strategy="tournament"))
            self.assertEqual(active, set())

    async def test_bad_usage_is_not_reported_as_success(self):
        async def run(row):
            return JevRowResult([0.0] * self.spec.num_labels, -1)

        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            await JevExecutor(self.spec, run).decide(choice(2))

    async def test_client_errors_are_distinct_from_invalid_engine_outputs(self):
        executor = JevExecutor(self.spec, self.uniform)
        for invalid in (choice(1), [], "bad", None):
            with self.subTest(invalid=invalid), self.assertRaises(JevRequestError):
                await executor.decide(invalid)
        with self.assertRaises(JevRequestError):
            await executor.systemone({"state": "", "questions": {}})

        async def invalid_engine(row):
            return JevRowResult([0.0], 1)

        with self.assertRaises(ValueError) as raised:
            await JevExecutor(self.spec, invalid_engine).decide(choice(2))
        self.assertNotIsInstance(raised.exception, JevRequestError)


class GroupTest(unittest.TestCase):
    def test_balanced_contiguous_groups(self):
        for count in (17, 128, 129, 256):
            partition = groups(count)
            self.assertEqual(
                [index for group in partition for index in group], list(range(count))
            )
            self.assertLessEqual(max(map(len, partition)), 16)
            self.assertLessEqual(max(map(len, partition)) - min(map(len, partition)), 1)


if __name__ == "__main__":
    unittest.main()
