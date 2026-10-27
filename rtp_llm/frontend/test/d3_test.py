"""D3 protocol tests: official prompt, question, answer and image contracts."""

import asyncio
import base64
import io
import math
import unittest
import unittest.mock
from dataclasses import replace
from types import SimpleNamespace

from rtp_llm.frontend.d3 import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_BATCH_TOKENS,
    MAX_OPTIONS,
    SYSTEM_PROMPT,
    D3Executor,
    D3RequestError,
    _bounded_batches,
    answer_codes,
    candidate_codes,
    image_visual_tokens,
    messages,
    normalize_question,
    product_answer,
    user_prompt,
)
from rtp_llm.models.downstream_modules.d3_config import D3Config


class FakeTokenizer:
    """Deterministic chat-template tokenizer; a maximal run of capitals is one token."""

    def __init__(self):
        self.runs = {}
        self.templates = []
        self.rendered = []

    def get_real_tokenizer(self):
        return self

    def _token_of(self, run):
        if run not in self.runs:
            self.runs[run] = 100000 + len(self.runs)
        return self.runs[run]

    def _encode(self, text):
        ids, run = [], ""
        for char in text + "\0":
            if char.isascii() and char.isupper():
                run += char
                continue
            if run:
                ids.append(self._token_of(run))
                run = ""
            if char != "\0":
                ids.append(1000 + ord(char))
        return ids

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self._encode(item) for item in text]}
        return {"input_ids": self._encode(text)}

    def encode(self, text, add_special_tokens=True):
        return self._encode(text)

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
        **_,
    ):
        self.templates.append((add_generation_prompt, enable_thinking, conversation))
        parts = []
        for message in conversation:
            content = message["content"]
            if isinstance(content, list):
                body = "".join(
                    (
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        if part["type"] == "image"
                        else (
                            "<|vision_start|><|video_pad|><|vision_end|>"
                            if part["type"] == "video"
                            else part["text"]
                        )
                    )
                    for part in content
                )
            else:
                body = content
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        self.rendered = getattr(self, "rendered", []) + [rendered]
        return rendered


def image_url(width, height, fmt="PNG"):
    from PIL import Image

    image = Image.new("RGB", (width, height), (96, 160, 224))
    with io.BytesIO() as buffer:
        image.save(buffer, format=fmt)
        data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"


def config(tokenizer=None, **overrides):
    codes, token_ids = answer_codes(tokenizer or FakeTokenizer())
    values = dict(
        format_version=1,
        format_id="d3-code-readout-v1",
        codes=tuple(codes),
        token_ids=tuple(token_ids),
        temperature=1.0,
        attention_mode="noncausal_full_attention",
        pooling="last",
        max_length=None,
        readout_dtype="float32",
        hidden_size=2560,
    )
    values.update(overrides)
    return D3Config(**values)


def executor(tokenizer=None, run_rows=None, **overrides):
    return D3Executor(
        config(
            tokenizer,
            **{
                k: v for k, v in overrides.items() if k in D3Config.__dataclass_fields__
            },
        ),
        run_rows or (lambda rows: []),
        tokenizer=tokenizer or FakeTokenizer(),
        model_name="d3-mini",
        max_row_tokens=overrides.get("max_row_tokens"),
        batch_rows=overrides.get("batch_rows", DEFAULT_BATCH_ROWS),
        batch_tokens=overrides.get("batch_tokens", DEFAULT_BATCH_TOKENS),
        timeout_seconds=overrides.get("timeout_seconds"),
    )


class PromptContractTest(unittest.TestCase):
    def test_user_prompt_exact_choice_text(self):
        question = {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"red": "warm", "blue": None},
        }
        self.assertEqual(
            user_prompt({"k": 1, "list": ["a", 2]}, question, ["A", "B"]),
            'State:\n{"k": 1, "list": ["a", 2]}\n\nQuestion:\nPick one\n\nOptions:\n'
            "A: red: warm\nB: blue\n\nReply with only the code of the best option.",
        )

    def test_user_prompt_empty_state_and_default_instructions(self):
        for state in (None, ""):
            with self.subTest(state=state):
                self.assertIn(
                    "(empty)", user_prompt(state, {"type": "noul"}, ["A", "B"])
                )
        self.assertEqual(
            user_prompt("s", {"type": "noul"}, ["A", "B"]),
            "State:\ns\n\nQuestion:\nChoose the best matching option.\n\nOptions:\n"
            "A: No / false\nB: Yes / true\n\nReply with only the code of the best option.",
        )

    def test_noul_criteria_descriptions(self):
        question = {
            "type": "noul",
            "criteria": {"false": "absent", "true": "present"},
        }
        self.assertEqual(
            user_prompt("s", question, ["A", "B"]),
            "State:\ns\n\nQuestion:\nChoose the best matching option.\n\nOptions:\n"
            "A: absent\nB: present\n\nReply with only the code of the best option.",
        )

    def test_score_levels_are_bare_option_texts_in_level_order(self):
        question = {"type": "score", "criteria": ["bad", {"grade": "ok"}, "good"]}
        normalized = normalize_question(question)
        self.assertEqual(normalized.kind, "score")
        self.assertEqual(
            user_prompt("s", normalized.rendered, ["A", "B", "C"]),
            "State:\ns\n\nQuestion:\nChoose the best matching option.\n\nOptions:\n"
            'A: bad\nB: {"grade": "ok"}\nC: good\n\nReply with only the code of the best option.',
        )

    def test_messages_carry_the_official_system_prompt(self):
        conversation = messages("s", {"type": "noul"}, ["A", "B"])
        self.assertEqual(conversation[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(conversation[1]["role"], "user")

    def test_instructions_render_as_json(self):
        question = {
            "type": "choice",
            "instructions": ["step", 2],
            "criteria": {"x": None},
        }
        self.assertIn('Question:\n["step", 2]', user_prompt("s", question, ["A"]))


class QuestionTest(unittest.TestCase):
    def test_choice_accepts_one_to_255_options(self):
        self.assertEqual(
            len(
                normalize_question({"type": "choice", "criteria": {"only": None}}).keys
            ),
            1,
        )
        big = {f"option-{i}": i for i in range(MAX_OPTIONS)}
        normalized = normalize_question({"type": "choice", "criteria": big})
        self.assertEqual(len(normalized.keys), MAX_OPTIONS)
        self.assertEqual(normalized.rendered["criteria"], big)

    def test_invalid_questions(self):
        cases = [
            "text",
            [],
            {"type": "unknown"},
            {"type": "choice"},
            {"type": "choice", "criteria": {}},
            {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}},
            {"type": "choice", "criteria": {"": None}},
            {"type": "choice", "criteria": {"a": float("nan")}},
            {"type": "noul", "criteria": {"maybe": "x"}},
            {"type": "noul", "criteria": {"false": float("inf")}},
            {"type": "score"},
            {"type": "score", "criteria": ["one"]},
            {"type": "score", "criteria": [f"level{i}" for i in range(11)]},
            {"type": "score", "criteria": ["same", "same"]},
            {"type": "score", "criteria": ["", "x"]},
        ]
        for question in cases:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    normalize_question(question)

    def test_score_and_noul_question_metadata(self):
        score = normalize_question({"type": "score", "criteria": ["low", "high"]})
        self.assertEqual(score.keys, ["0", "1"])
        self.assertEqual(score.descriptions, ["low", "high"])
        noul = normalize_question({"type": "noul"})
        self.assertEqual(noul.keys, ["false", "true"])
        self.assertEqual(noul.descriptions, ["No / false", "Yes / true"])


class AnswerCodesTest(unittest.TestCase):
    def test_all_candidates_are_single_tokens_so_first_255_are_used(self):
        tokenizer = FakeTokenizer()
        codes, token_ids = answer_codes(tokenizer)
        self.assertEqual(codes, candidate_codes()[:MAX_OPTIONS])
        self.assertEqual(len(token_ids), MAX_OPTIONS)
        self.assertEqual(len(set(token_ids)), MAX_OPTIONS)

    def test_executor_rejects_config_codes_that_differ_from_the_tokenizer(self):
        tokenizer = FakeTokenizer()
        good = config(tokenizer)
        D3Executor(good, lambda rows: [], tokenizer=tokenizer, model_name="d3-mini")
        wrong = replace(good, codes=("code-a",) + good.codes[1:])
        with self.assertRaisesRegex(ValueError, "answer codes or token ids"):
            D3Executor(
                wrong, lambda rows: [], tokenizer=tokenizer, model_name="d3-mini"
            )


class ExecutorConfigTest(unittest.TestCase):
    def test_bad_configs_are_rejected(self):
        good = config()
        cases = {
            "format_version": replace(good, format_version=2),
            "format_id": replace(good, format_id="other-v1"),
            "codes": replace(good, codes=good.codes[:-1]),
            "token_ids": replace(good, token_ids=(0,) + good.token_ids[1:]),
            "temperature": replace(good, temperature=0.0),
            "attention_mode": replace(good, attention_mode="bidirectional"),
            "pooling": replace(good, pooling="mean"),
            "max_length": replace(good, max_length=0),
            "readout_dtype": replace(good, readout_dtype="float16"),
            "hidden_size": replace(good, hidden_size=0),
            "missing": SimpleNamespace(
                **{k: v for k, v in good.__dict__.items() if k != "pooling"}
            ),
        }
        for name, bad in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    D3Executor(
                        bad,
                        lambda rows: [],
                        tokenizer=FakeTokenizer(),
                        model_name="d3-mini",
                    )

    def test_bad_executor_arguments(self):
        for overrides in (
            {"max_row_tokens": 0},
            {"batch_rows": 0},
            {"batch_rows": 65},
            {"batch_tokens": 0},
            {"timeout_seconds": 0},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    executor(**overrides)


class AnswerTest(unittest.TestCase):
    def choice_question(self, criteria):
        return normalize_question({"type": "choice", "criteria": criteria})

    def test_choice_answer_with_entropy_confidence(self):
        answer = product_answer(
            self.choice_question({"red": None, "blue": None}), [0.25, 0.75]
        )
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "blue")
        self.assertEqual(answer["probabilities"], {"red": 0.25, "blue": 0.75})
        self.assertAlmostEqual(answer["confidence"], 0.18872187554086717, places=15)

    def test_single_option_choice_is_certain(self):
        answer = product_answer(self.choice_question({"only": None}), [1.0])
        self.assertEqual(answer["choice"], "only")
        self.assertEqual(answer["probabilities"], {"only": 1.0})
        self.assertEqual(answer["confidence"], 1.0)

    def test_noul_answer_has_no_confidence_field(self):
        question = normalize_question({"type": "noul"})
        self.assertEqual(
            product_answer(question, [0.3, 0.7]), {"type": "noul", "noul": 0.7}
        )

    def test_score_answer_is_level_expectation_with_canonical_legend(self):
        question = normalize_question(
            {"type": "score", "criteria": ["low", {"grade": "ok"}, "high"]}
        )
        answer = product_answer(question, [0.2, 0.5, 0.3])
        self.assertEqual(answer["type"], "score")
        self.assertAlmostEqual(answer["score"], 0.2 * 0 + 0.5 * 1 + 0.3 * 2, places=15)
        self.assertEqual(answer["probabilities"], {"0": 0.2, "1": 0.5, "2": 0.3})
        self.assertEqual(
            answer["legend"], {"0": "low", "1": '{"grade":"ok"}', "2": "high"}
        )
        self.assertAlmostEqual(answer["confidence"], 0.06276943678387048, places=15)

    def test_answer_probabilities_are_renormalized_once(self):
        answer = product_answer(self.choice_question({"a": None, "b": None}), [1, 3])
        self.assertEqual(answer["probabilities"], {"a": 0.25, "b": 0.75})


class ProbabilitiesTest(unittest.TestCase):
    def test_masked_softmax_with_temperature_applied_once(self):
        tokenizer = FakeTokenizer()
        engine = executor(tokenizer)
        engine.config = replace(engine.config, temperature=2.0)
        logits = [2.0, 4.0] + [1000.0] * (MAX_OPTIONS - 2)
        result = SimpleNamespace(logits=logits, prompt_tokens=3)
        values = engine.probabilities(result, 2)
        expected = 1.0 / (1.0 + math.exp(1.0))
        self.assertAlmostEqual(values[0], expected, places=12)
        self.assertAlmostEqual(values[1], 1 - expected, places=12)

    def test_nonfinite_or_wrong_sized_logits_are_invalid_model_output(self):
        engine = executor()
        for logits, count in (
            ([0.0] * 254, 2),
            ([0.0, float("nan")] + [0.0] * 253, 2),
            ([float("inf")] + [0.0] * 254, 1),
        ):
            with self.subTest(logits=logits[:3]):
                with self.assertRaises(ValueError):
                    engine.probabilities(
                        SimpleNamespace(logits=logits, prompt_tokens=1), count
                    )


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.engine = executor(self.tokenizer)

    def prepare(self, request):
        return self.engine.prepare(request)

    def test_whole_request_rejections(self):
        cases = [
            "text",
            {"state": "s"},
            {"questions": {}},
            {"questions": []},
            {
                "questions": {"q": {"type": "choice", "criteria": {"a": None}}},
                "extra": 1,
            },
            {"questions": {"q": {"type": "noul"}}, "videos": "clip.mp4"},
            {"questions": {3: {"type": "noul"}}},
            {
                "questions": {"q": {"type": "noul"}},
                "images": "data:image/png;base64,AAA=",
            },
            {"questions": {"q": {"type": "noul"}}, "images": [42]},
            {
                "questions": {"q": {"type": "noul"}},
                "images": ["https://example.invalid/x.png"],
            },
            {
                "questions": {"q": {"type": "noul"}},
                "images": ["data:image/gif;base64,AAA="],
            },
            {
                "questions": {"q": {"type": "noul"}},
                "images": ["data:image/png;base64,!!!"],
            },
            {
                "questions": {"q": {"type": "noul"}},
                "images": ["data:image/png;base64," + "A" * 11000000],
            },
        ]
        for request in cases:
            with self.subTest(request=request):
                with self.assertRaises(D3RequestError) as caught:
                    self.prepare(request)
                self.assertIsInstance(caught.exception.detail, str)

    def test_non_json_and_circular_states_are_rejected(self):
        with self.assertRaises(D3RequestError):
            self.prepare({"state": float("nan"), "questions": {"q": {"type": "noul"}}})
        circular = {"text": "loop"}
        circular["self"] = circular
        with self.assertRaises(D3RequestError):
            self.prepare({"state": circular, "questions": {"q": {"type": "noul"}}})

    def test_videos_are_validated_with_official_reasons(self):
        for url, reason in [
            ("data:video/avi;base64,AAAA", "video data URLs take the types"),
            ("not-a-url", "videos must be given as a data:video"),
            ("data:video/mp4;base64,!!!", "valid base64"),
            ("data:video/mp4,AAAA", "expected a base64 data URL"),
        ]:
            with self.subTest(url=url[:40]):
                with self.assertRaises(D3RequestError) as caught:
                    self.prepare(
                        {
                            "questions": {"q": {"type": "noul"}},
                            "videos": [url],
                        }
                    )
                self.assertIn(reason, str(caught.exception))

    def test_video_bytes_limit_refuses_before_decoding(self):
        # A data URL whose base64 text alone exceeds the 32,000,000 byte limit
        # is refused before any decode, exactly like the official server.
        huge = "data:video/mp4;base64," + "A" * (4 * math.ceil(32_000_000 / 3) + 1)
        with self.assertRaises(D3RequestError) as caught:
            self.prepare({"questions": {"q": {"type": "noul"}}, "videos": [huge]})
        self.assertIn("byte limit", str(caught.exception))

    def test_invalid_questions_return_the_official_detail(self):
        request = {
            "questions": {
                "good": {"type": "noul"},
                "bad": {"type": "choice", "criteria": {}},
                "worse": "not a question",
            }
        }
        with self.assertRaises(D3RequestError) as caught:
            self.prepare(request)
        self.assertEqual(
            caught.exception.detail,
            {
                "invalid_questions": {
                    "bad": "choice criteria must map 1 to 255 option keys to descriptions",
                    "worse": "a question is an object with a type",
                }
            },
        )

    def test_text_only_literal_media_tokens_are_allowed(self):
        # Official contract: the text path has NO placeholder check — a
        # text-only request whose state carries a literal media token is
        # answered normally (nothing is injected into it).
        for state in ("<|image_pad|>", "safe <|video_pad|> tail"):
            with self.subTest(state=state):
                prepared = self.prepare(
                    {"state": state, "questions": {"q": {"type": "noul"}}}
                )
                self.assertEqual(dict(prepared.errors), {})
                self.assertEqual([row.question_id for row in prepared.rows], ["q"])

    def test_media_placeholder_injection_is_a_whole_request_rejection(self):
        # Official contract: on media paths a literal media token (either
        # direction) is an invalid_question, and EVERY invalid_question —
        # including this render-phase one — rejects the WHOLE request with the
        # invalid_questions detail before anything runs; the valid question in
        # the same request does not run either.
        cases = {
            "image-token-in-image-request": "before <|image_pad|> after",
            "video-token-in-image-request": "before <|video_pad|> after",
        }
        for name, instructions in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(D3RequestError) as caught:
                    self.prepare(
                        {
                            "state": "s",
                            "images": [image_url(64, 64)],
                            "questions": {
                                "bad": {"type": "noul", "instructions": instructions},
                                "good": {"type": "noul"},
                            },
                        }
                    )
                detail = caught.exception.detail
                self.assertIn("bad", detail["invalid_questions"])
                self.assertNotIn("good", detail["invalid_questions"])

    def test_rows_carry_prompt_tokens_and_option_counts_in_request_order(self):
        prepared = self.prepare(
            {
                "state": "context",
                "questions": {
                    "first": {"type": "choice", "criteria": {"a": None, "b": None}},
                    "second": {"type": "noul"},
                },
            }
        )
        self.assertEqual(
            [row.question_id for row in prepared.rows], ["first", "second"]
        )
        self.assertEqual([row.option_count for row in prepared.rows], [2, 2])
        self.assertEqual(
            [row.planned_tokens for row in prepared.rows],
            [len(row.token_ids) for row in prepared.rows],
        )
        prompt = self.tokenizer.apply_chat_template(
            messages(
                "context",
                {"type": "choice", "criteria": {"a": None, "b": None}},
                ["A", "B"],
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertEqual(
            prepared.rows[0].token_ids,
            self.tokenizer.encode(prompt, add_special_tokens=False),
        )
        self.assertTrue(self.tokenizer.templates)
        self.assertTrue(
            all(
                flag and thinking is False
                for flag, thinking, _ in self.tokenizer.templates
            )
        )

    def test_over_limit_question_errors_per_question_without_truncation(self):
        small = {"type": "choice", "criteria": {"only": None}, "instructions": "small"}
        large = {
            "type": "choice",
            "instructions": "large",
            "criteria": {f"option-{i}": f"description {i}" for i in range(200)},
        }
        unlimited = self.prepare({"questions": {"small": small, "large": large}})
        limit = [
            row.planned_tokens for row in unlimited.rows if row.question_id == "small"
        ][0]
        engine = executor(self.tokenizer, max_row_tokens=limit)
        prepared = engine.prepare({"questions": {"small": small, "large": large}})
        self.assertEqual([row.question_id for row in prepared.rows], ["small"])
        error = prepared.errors["large"]
        self.assertEqual(error["error"], "max_length_exceeded")
        self.assertEqual(error["type"], "choice")
        self.assertIn("nothing was truncated", error["message"])

    def test_config_max_length_limits_rows_the_same_way(self):
        question = {"type": "noul"}
        unlimited = self.prepare({"questions": {"q": question}})
        limit = unlimited.rows[0].planned_tokens - 1
        engine = executor(self.tokenizer)
        engine.config = replace(engine.config, max_length=limit)
        prepared = engine.prepare({"questions": {"q": question}})
        self.assertEqual(prepared.rows, [])
        self.assertEqual(prepared.errors["q"]["error"], "max_length_exceeded")


class ImageTest(unittest.TestCase):
    def test_image_row_counts_visual_tokens_and_placeholder_ids(self):
        tokenizer = FakeTokenizer()
        engine = executor(tokenizer)
        prepared = engine.prepare(
            {"questions": {"q": {"type": "noul"}}, "images": [image_url(1024, 1024)]}
        )
        self.assertEqual(len(prepared.rows), 1)
        row = prepared.rows[0]
        self.assertEqual(len(row.media), 1)
        self.assertEqual(row.media[0].visual_tokens, 1024)
        self.assertEqual(row.planned_tokens, len(row.token_ids) - 1 + 1024)
        self.assertEqual(tokenizer.rendered[-1].count("<|image_pad|>"), 1)

    def test_image_prompt_places_every_placeholder_before_the_text(self):
        tokenizer = FakeTokenizer()
        engine = executor(tokenizer)
        prepared = engine.prepare(
            {
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "images": [image_url(64, 64), image_url(448, 336)],
            }
        )
        row = prepared.rows[0]
        self.assertEqual([media.visual_tokens for media in row.media], [64, 140])
        self.assertEqual(row.planned_tokens, len(row.token_ids) - 2 + 64 + 140)
        content = tokenizer.templates[-1][2][1]["content"]
        self.assertEqual([part["type"] for part in content[:2]], ["image", "image"])
        self.assertTrue(content[2]["text"].startswith("State:\ns\n"))

    def test_bad_images_are_rejected_with_official_reasons(self):
        engine = executor(FakeTokenizer())
        cases = {
            "data:image/png;base64,AAA=": "could not be identified",
            "data:image/png;base64,"
            + base64.b64encode(b"not an image").decode(): "could not be identified",
            "data:image/tiff;base64,"
            + base64.b64encode(b"x").decode(): "image data URLs take the types",
            "file:///tmp/x.png": "strict mode: images must be given as a data: URL string",
            "data:image/png;base64,"
            + base64.b64encode(b"x" * 8000001).decode(): "byte limit",
        }
        for url, reason in cases.items():
            with self.subTest(url=url[:48]):
                with self.assertRaises(D3RequestError) as caught:
                    engine.prepare(
                        {"questions": {"q": {"type": "noul"}}, "images": [url]}
                    )
                self.assertIn(reason, str(caught.exception))

    def test_too_many_source_pixels_and_extreme_aspect_ratios_are_refused(self):
        engine = executor(FakeTokenizer())
        with self.assertRaises(D3RequestError) as caught:
            engine.prepare(
                {
                    "questions": {"q": {"type": "noul"}},
                    "images": [image_url(4096, 4096)],
                }
            )
        self.assertIn("pixels; at most 16,000,000 are accepted", str(caught.exception))
        with self.assertRaises(D3RequestError) as caught:
            engine.prepare(
                {"questions": {"q": {"type": "noul"}}, "images": [image_url(6404, 32)]}
            )
        self.assertIn("image rejected by the processor", str(caught.exception))

    def test_visual_token_planning_follows_the_processor_resize(self):
        # Expected values follow the Qwen-VL smart_resize: round to 32, clamp to
        # [65536, 1638400] pixels, one token per 32x32 block.
        self.assertEqual(
            [
                image_visual_tokens(64, 64),
                image_visual_tokens(1024, 1024),
                image_visual_tokens(448, 336),
                image_visual_tokens(5000, 100),
                image_visual_tokens(200, 100),
            ],
            [64, 1024, 140, 468, 72],
        )

    def test_image_placeholder_injection_rejects_the_whole_request(self):
        # Render-phase invalid_questions (literal placeholder tokens on media
        # paths) reject the WHOLE request before anything runs — official 422
        # invalid_questions, never a 200 with per-question errors.
        engine = executor(FakeTokenizer())
        with self.assertRaises(D3RequestError) as caught:
            engine.prepare(
                {
                    "state": "<|image_pad|>",
                    "questions": {"q": {"type": "noul"}},
                    "images": [image_url(64, 64)],
                }
            )
        self.assertIn("q", caught.exception.detail["invalid_questions"])


class BatchingTest(unittest.TestCase):
    def row(self, qid, planned):
        return SimpleNamespace(
            question_id=qid,
            token_ids=[1] * planned,
            option_count=2,
            question=None,
            media=(),
            planned_tokens=planned,
        )

    def test_batches_are_bounded_by_rows_and_padded_tokens(self):
        rows = [self.row(f"q{i}", 100) for i in range(20)]
        batches = _bounded_batches(rows, batch_rows=8, batch_tokens=16384)
        self.assertEqual([len(batch) for batch in batches], [8, 8, 4])
        self.assertEqual(
            [row.question_id for batch in batches for row in batch],
            [f"q{i}" for i in range(20)],
        )
        wide = [self.row("small", 10), self.row("huge", 9000), self.row("tiny", 5)]
        batches = _bounded_batches(wide, batch_rows=8, batch_tokens=10000)
        self.assertEqual(
            [[row.question_id for row in batch] for batch in batches],
            [["small"], ["huge"], ["tiny"]],
        )

    def test_single_row_above_the_bound_still_runs(self):
        batches = _bounded_batches(
            [self.row("q", 50000)], batch_rows=8, batch_tokens=100
        )
        self.assertEqual(
            [[row.question_id for row in batch] for batch in batches], [["q"]]
        )


class SystemOneFlowTest(unittest.TestCase):
    def test_full_request_lifecycle_with_official_answers(self):
        tokenizer = FakeTokenizer()
        seen = []

        async def run_rows(rows):
            seen.append([row.question_id for row in rows])
            return [
                SimpleNamespace(
                    logits=[
                        2.0 if index == row.option_count - 1 else 0.0
                        for index in range(MAX_OPTIONS)
                    ],
                    prompt_tokens=row.planned_tokens,
                )
                for row in rows
            ]

        engine = executor(tokenizer, run_rows=run_rows)
        request = {
            "model": "ignored-name",
            "state": {"items": ["a", "b"]},
            "questions": {
                "pick": {"type": "choice", "criteria": {"red": None, "blue": None}},
                "yn": {"type": "noul"},
                "grade": {"type": "score", "criteria": ["low", "mid", "high"]},
            },
        }
        result = asyncio.run(engine.systemone(request))
        self.assertEqual(seen, [["pick", "yn", "grade"]])
        self.assertEqual(result["model"], "d3-mini")
        self.assertEqual(set(result), {"model", "answers", "usage"})
        self.assertEqual(list(result["answers"]), ["pick", "yn", "grade"])
        pick = result["answers"]["pick"]
        self.assertEqual(pick["choice"], "blue")
        expected_blue = 1.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(pick["probabilities"]["blue"], expected_blue, places=12)
        self.assertAlmostEqual(
            result["answers"]["yn"]["noul"], expected_blue, places=12
        )
        self.assertEqual(set(result["answers"]["yn"]), {"type", "noul"})
        grade = result["answers"]["grade"]
        expected_score = 1.0 / (2.0 + math.exp(2.0)) + 2.0 * math.exp(2.0) / (
            2.0 + math.exp(2.0)
        )
        self.assertAlmostEqual(grade["score"], expected_score, places=12)
        self.assertIn("legend", grade)
        self.assertEqual(
            result["usage"],
            {
                "input_tokens": sum(
                    row.planned_tokens for row in engine.prepare(request).rows
                ),
                "output_tokens": 0,
            },
        )

    def test_run_row_count_mismatch_is_a_server_error(self):
        async def run_rows(rows):
            return []

        engine = executor(run_rows=run_rows)
        with self.assertRaises(RuntimeError):
            asyncio.run(engine.systemone({"questions": {"q": {"type": "noul"}}}))


class VideoTest(unittest.TestCase):
    """The official video math: sampling, resize budgets, timestamps, groups."""

    def test_sample_indices_follow_the_official_formula(self):
        import importlib.util

        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy not installed (Bazel runs carry it)")
        from rtp_llm.frontend.d3 import video_sample_indices

        # Official d3_runtime.video_sample_indices (pinned reference, lines
        # 575-581): np.linspace(0, total-1, count).round().astype(int) — the
        # LAST index is total-1, never total, and 24 frames at 24 fps clamp to
        # 4 frames that round to [0, 8, 15, 23] (7.67 -> 8, 15.33 -> 15).
        self.assertEqual(video_sample_indices(10, 24.0), [0, 3, 6, 9])
        self.assertEqual(video_sample_indices(24, 24.0), [0, 8, 15, 23])
        self.assertEqual(
            video_sample_indices(9600, 24.0),
            video_sample_indices(9600, 24.0),
        )
        long = video_sample_indices(9600, 24.0)
        self.assertEqual(len(long), 32)
        self.assertEqual(long[0], 0)
        self.assertEqual(long[-1], 9599)
        # A clip shorter than the minimum still reads every frame once.
        self.assertEqual(video_sample_indices(2, 24.0), [0, 1])
        self.assertEqual(video_sample_indices(1, 24.0), [0])

    def test_timestamps_average_each_two_frames(self):
        from rtp_llm.frontend.d3 import video_timestamps

        self.assertEqual(video_timestamps([0, 24], 24.0), [0.5])
        self.assertEqual(video_timestamps([0, 12, 24, 36], 24.0), [0.25, 1.25])
        # An odd count repeats the last frame's time, exactly like the official
        # _calculate_timestamps padding.
        self.assertEqual(video_timestamps([0, 10, 20], 10.0), [0.5, 2.0])

    def test_resize_dims_cap_each_frame_like_the_official_processor(self):
        from rtp_llm.frontend.d3 import video_resize_dims

        # A 1080p clip at the 32-frame maximum: every frame is capped at
        # 196 tokens of pixels (200,704 px), so the budget matches the cap.
        height, width = video_resize_dims(32, 1080, 1920)
        self.assertLessEqual(height * width, 200_704)
        self.assertEqual(height % 32, 0)
        self.assertEqual(width % 32, 0)
        # Few frames keep the same per-frame cap (the clip budget shrinks with
        # the frame count).
        for frames in (4, 8, 16, 32):
            h, w = video_resize_dims(frames, 1080, 1920)
            self.assertLessEqual(h * w, 200_704, frames)
            self.assertGreaterEqual(h * w, 200_704 - 2 * 32 * 1920, frames)
        # A small clip is scaled up to at least the 4,096-pixel floor.
        h, w = video_resize_dims(4, 8, 8)
        self.assertGreaterEqual(h * w * 4, 4_096)
        self.assertEqual(h % 32, 0)
        # Frames below the factor are scaled up before rounding.
        h, w = video_resize_dims(4, 10, 20)
        self.assertEqual(h, 32)
        self.assertEqual(w, 64)

    def test_group_tokens_are_one_per_32_by_32(self):
        from rtp_llm.frontend.d3 import video_group_tokens

        self.assertEqual(video_group_tokens(1024, 1024), 1024)
        self.assertEqual(video_group_tokens(160, 160), 25)

    def test_expanded_video_text_marks_every_two_frames(self):
        from rtp_llm.frontend.d3 import D3Video, expanded_video_text

        video = D3Video(
            frames=None,
            fps=24.0,
            indices=[0, 12, 24, 36],
            total_frames=48,
            height=160,
            width=160,
            timestamps=[0.25, 0.75],
            group_tokens=25,
            key="k",
        )
        self.assertEqual(
            expanded_video_text(video),
            "<0.2 seconds><|vision_start|>"
            + "<|video_pad|>" * 25
            + "<|vision_end|>"
            + "<0.8 seconds><|vision_start|>"
            + "<|video_pad|>" * 25
            + "<|vision_end|>",
        )

    def test_video_messages_place_media_before_the_text(self):
        from rtp_llm.frontend.d3 import normalize_question, video_messages

        conversation = video_messages(
            "s", normalize_question({"type": "noul"}).rendered, ["A", "B"], 1, 2
        )
        self.assertEqual(conversation[0]["role"], "system")
        content = conversation[1]["content"]
        self.assertEqual(
            [part["type"] for part in content], ["image", "video", "video", "text"]
        )

    def test_multi_video_placeholder_expansion_is_indexed_per_video(self):
        # A 1-token group's expansion is itself a placeholder-shaped span, so
        # first-occurrence replacement would let video 0's expansion eat video
        # 1's placeholder (independent-review P2). The expansion is indexed per
        # video; both the wrapped and the bare placeholder forms are covered.
        # Official contract: only the BARE token is replaced, so the template's
        # outer vision markers stay around the wrapped form's whole expansion.
        from rtp_llm.frontend.d3 import (
            VIDEO_TOKEN,
            VISION_END,
            VISION_START,
            D3Video,
            _expand_video_placeholders,
        )

        def video(key, tokens, timestamps):
            return D3Video(
                frames=None,
                fps=24.0,
                indices=[0, 12, 24, 36],
                total_frames=48,
                height=32,
                width=32 * tokens,
                timestamps=timestamps,
                group_tokens=tokens,
                key=key,
            )

        tiny = video("a", 1, [0.25, 0.75])  # 2 groups x 1 token (the trigger)
        large = video("b", 9, [1.0, 3.0])  # 2 groups x 9 tokens
        wrapped = (
            "A "
            + VISION_START
            + VIDEO_TOKEN
            + VISION_END
            + " B "
            + VISION_START
            + VIDEO_TOKEN
            + VISION_END
            + " C"
        )
        bare = "A " + VIDEO_TOKEN + " B " + VIDEO_TOKEN + " C"
        expected = {
            # The wrapped form keeps its outer markers around the expansion
            # (official get_text_with_replacements replaces the bare token
            # only — 2 extra tokens per video, the request-level deficit).
            "wrapped": (
                "A "
                + VISION_START
                + "<0.2 seconds>"
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + "<0.8 seconds>"
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + VISION_END
                + " B "
                + VISION_START
                + "<1.0 seconds>"
                + VISION_START
                + VIDEO_TOKEN * 9
                + VISION_END
                + "<3.0 seconds>"
                + VISION_START
                + VIDEO_TOKEN * 9
                + VISION_END
                + VISION_END
                + " C"
            ),
            # The bare form is replaced in place; nothing else is added.
            "bare": (
                "A "
                + "<0.2 seconds>"
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + "<0.8 seconds>"
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + " B "
                + "<1.0 seconds>"
                + VISION_START
                + VIDEO_TOKEN * 9
                + VISION_END
                + "<3.0 seconds>"
                + VISION_START
                + VIDEO_TOKEN * 9
                + VISION_END
                + " C"
            ),
        }
        for name, text in {"wrapped": wrapped, "bare": bare}.items():
            with self.subTest(form=name):
                expanded = _expand_video_placeholders(text, [tiny, large])
                self.assertEqual(expanded, expected[name])
                # Order and no nesting: every timestamp group stays with its
                # own video position (the old bug nested video 1's expansion
                # inside video 0's first group and reordered timestamps).
                self.assertEqual(expanded.count(VIDEO_TOKEN), 2 * 1 + 2 * 9)
                self.assertLess(
                    expanded.index("<0.2 seconds>"), expanded.index("<0.8 seconds>")
                )
                self.assertLess(
                    expanded.index("<0.8 seconds>"), expanded.index("<1.0 seconds>")
                )
                self.assertLess(
                    expanded.index("<1.0 seconds>"), expanded.index("<3.0 seconds>")
                )

    def test_private_use_unicode_is_preserved_across_expansion(self):
        # Official never rejects ordinary Unicode (private-use included): the
        # indexed expansion preserves arbitrary characters verbatim while
        # every video's groups stay in that video's own placeholder position.
        from rtp_llm.frontend.d3 import (
            VIDEO_TOKEN,
            VISION_END,
            VISION_START,
            D3Video,
            _expand_video_placeholders,
        )

        def video(key, tokens, timestamps):
            return D3Video(
                frames=None,
                fps=24.0,
                indices=[0, 12, 24, 36],
                total_frames=48,
                height=32,
                width=32 * tokens,
                timestamps=timestamps,
                group_tokens=tokens,
                key=key,
            )

        tiny = video("a", 1, [0.25, 0.75])  # 2 groups x 1 token (the trigger)
        large = video("b", 9, [1.0, 3.0])
        text = (
            "\ue000A "
            + VISION_START
            + VIDEO_TOKEN
            + VISION_END
            + " B\ue001 "
            + VISION_START
            + VIDEO_TOKEN
            + VISION_END
            + " C\ue000"
        )
        expanded = _expand_video_placeholders(text, [tiny, large])
        # Every private-use character survives verbatim, and each video's
        # groups sit in that video's own placeholder position, in order — the
        # template's outer markers stay around each expansion.
        self.assertTrue(
            expanded.startswith("\ue000A " + VISION_START + "<0.2 seconds>")
        )
        self.assertIn(" B\ue001 " + VISION_START + "<1.0 seconds>", expanded)
        self.assertTrue(expanded.endswith(VISION_END + " C\ue000"))
        self.assertLess(
            expanded.index("<0.2 seconds>"), expanded.index("<0.8 seconds>")
        )
        self.assertLess(
            expanded.index("<0.8 seconds>"), expanded.index("<1.0 seconds>")
        )
        self.assertLess(
            expanded.index("<1.0 seconds>"), expanded.index("<3.0 seconds>")
        )
        self.assertEqual(expanded.count(VIDEO_TOKEN), 2 * 1 + 2 * 9)

    def test_expansion_matches_the_official_replacement_reference(self):
        # Independent reference: ProcessorMixin.get_text_with_replacements
        # (transformers 5.17, processing_utils.py) scans the ORIGINAL text with
        # re.finditer over the BARE token, replaces the i-th occurrence with
        # the i-th video's replacement, and never rescans what it inserted;
        # Qwen3VLProcessor.replace_video_token (processing_qwen3_vl.py) emits
        # one "<t seconds>" + vision-wrapped pads per 2-frame group. The
        # frontend expansion must equal that reference byte for byte — outer
        # template markers included, multi-video 1-token groups included.
        import re

        from rtp_llm.frontend.d3 import (
            IMAGE_TOKEN,
            VIDEO_TOKEN,
            VISION_END,
            VISION_START,
            D3Video,
            _expand_video_placeholders,
        )

        def official(text, replacements):
            pattern = re.compile(re.escape(VIDEO_TOKEN))
            out, last, pending = [], 0, iter(replacements)
            for match in pattern.finditer(text):
                out.append(text[last : match.start()])
                out.append(next(pending))
                last = match.end()
            out.append(text[last:])
            return "".join(out)

        def video(key, tokens, timestamps):
            return D3Video(
                frames=None,
                fps=24.0,
                indices=[0, 12, 24, 36],
                total_frames=48,
                height=32,
                width=32 * tokens,
                timestamps=timestamps,
                group_tokens=tokens,
                key=key,
            )

        def replacement(item):
            # replace_video_token's exact per-group string.
            return "".join(
                f"<{t:.1f} seconds>"
                + VISION_START
                + VIDEO_TOKEN * item.group_tokens
                + VISION_END
                for t in item.timestamps
            )

        tiny = video("a", 1, [0.25, 0.75])  # 2 groups x 1 token (the trigger)
        large = video("b", 9, [1.0, 3.0])
        cases = {
            "single wrapped": (
                "X " + VISION_START + VIDEO_TOKEN + VISION_END + " Y",
                [tiny],
            ),
            "multi 1-token groups": (
                "A "
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + " B "
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + " C",
                [tiny, large],
            ),
            "bare tokens": (
                "A " + VIDEO_TOKEN + " B " + VIDEO_TOKEN + " C",
                [tiny, large],
            ),
            "image and video": (
                "X "
                + VISION_START
                + IMAGE_TOKEN
                + VISION_END
                + " mid "
                + VISION_START
                + VIDEO_TOKEN
                + VISION_END
                + " Y",
                [tiny],
            ),
            "private use unicode": (
                "\ue000" + VISION_START + VIDEO_TOKEN + VISION_END + "\ue001",
                [tiny],
            ),
        }
        for name, (text, videos) in cases.items():
            with self.subTest(case=name):
                expanded = _expand_video_placeholders(text, videos)
                self.assertEqual(
                    expanded, official(text, [replacement(item) for item in videos])
                )
        # The video pass never touches image placeholders: the official call
        # without image replacements leaves them in place for the engine.
        mixed = _expand_video_placeholders(cases["image and video"][0], [tiny])
        self.assertEqual(mixed.count(IMAGE_TOKEN), 1)
        self.assertLess(mixed.index(IMAGE_TOKEN), mixed.index("<0.2 seconds>"))

    def test_video_prompt_accounts_the_full_expanded_sequence(self):
        # Full expanded-sequence accounting: the row must carry exactly the
        # tokenization of the official expansion — every group's timestamp and
        # pad tokens AND the template's outer vision markers (the
        # wrapper-stripping bug was the measured 2-tokens-per-video
        # request-level deficit).
        import torch

        from rtp_llm.frontend.d3 import VIDEO_TOKEN, VISION_END, VISION_START, D3Video

        tokenizer = FakeTokenizer()
        engine = executor(tokenizer, run_rows=_no_rows())
        video = D3Video(
            frames=torch.zeros(4, 3, 64, 64, dtype=torch.uint8),
            fps=24.0,
            indices=[0, 8, 15, 23],
            total_frames=48,
            height=64,
            width=64,
            timestamps=[0.25, 0.75],
            group_tokens=4,
            key="k",
        )
        with unittest.mock.patch("rtp_llm.frontend.d3.load_video", return_value=video):
            prepared = engine.prepare(
                {
                    "questions": {"q": {"type": "noul"}},
                    "videos": ["data:video/mp4;base64,AAAA"],
                }
            )
        self.assertEqual(len(prepared.rows), 1)
        row = prepared.rows[0]
        prompt = tokenizer.rendered[-1]
        # The chat template renders ONE bare video placeholder; the official
        # replacement (get_text_with_replacements) swaps only that occurrence
        # for the video's expansion and keeps every other character — the
        # template's outer vision markers included. Reconstruct the official
        # result independently here (replace_video_token group grammar +
        # timestamps [0.25, 0.75] -> "<0.2 seconds>"/"<0.8 seconds>").
        self.assertEqual(prompt.count(VIDEO_TOKEN), 1)
        expansion = (
            "<0.2 seconds>"
            + VISION_START
            + VIDEO_TOKEN * 4
            + VISION_END
            + "<0.8 seconds>"
            + VISION_START
            + VIDEO_TOKEN * 4
            + VISION_END
        )
        expected = prompt.replace(VIDEO_TOKEN, expansion, 1)
        span = VISION_START + expansion + VISION_END
        self.assertIn(span, expected)
        self.assertEqual(expected.count(VIDEO_TOKEN), 2 * 4)
        self.assertEqual(expected.count(VISION_START), 3)  # 2 groups + the outer one
        self.assertEqual(expected.count(VISION_END), 3)
        # The row is exactly the tokenization of that official expansion, and
        # a video row spends all of it: len(ids) planned tokens.
        self.assertEqual(
            row.token_ids, tokenizer.encode(expected, add_special_tokens=False)
        )
        self.assertEqual(row.planned_tokens, len(row.token_ids))
        # One engine media entry per 2-frame group, in prompt order.
        self.assertEqual([entry.visual_tokens for entry in row.media], [4, 4])

    def test_mixed_prompt_accounts_images_and_the_expanded_video_sequence(self):
        # Image + video: the image placeholder stays one token for the engine
        # while the video expansion (outer markers included) is already in the
        # text — planned = len(ids) - n_images + image visual tokens.
        import torch

        from rtp_llm.frontend.d3 import (
            IMAGE_TOKEN,
            VIDEO_TOKEN,
            VISION_END,
            VISION_START,
            D3Video,
        )

        tokenizer = FakeTokenizer()
        engine = executor(tokenizer, run_rows=_no_rows())
        video = D3Video(
            frames=torch.zeros(4, 3, 64, 64, dtype=torch.uint8),
            fps=24.0,
            indices=[0, 8, 15, 23],
            total_frames=48,
            height=64,
            width=64,
            timestamps=[0.25, 0.75],
            group_tokens=4,
            key="k",
        )
        with unittest.mock.patch("rtp_llm.frontend.d3.load_video", return_value=video):
            prepared = engine.prepare(
                {
                    "questions": {"q": {"type": "noul"}},
                    "images": [image_url(1024, 1024)],
                    "videos": ["data:video/mp4;base64,AAAA"],
                }
            )
        self.assertEqual(len(prepared.rows), 1)
        row = prepared.rows[0]
        prompt = tokenizer.rendered[-1]
        # The template renders the image and the BARE video placeholder; the
        # official replacement swaps only the bare video token for the video's
        # expansion, so images stay one engine token while the expansion is
        # already in the text. Reconstruct the official result independently.
        self.assertEqual(prompt.count(IMAGE_TOKEN), 1)
        self.assertEqual(prompt.count(VIDEO_TOKEN), 1)
        # Images first, then the video, both outer-wrapped by the template.
        self.assertLess(prompt.index(IMAGE_TOKEN), prompt.index(VIDEO_TOKEN))
        expansion = (
            "<0.2 seconds>"
            + VISION_START
            + VIDEO_TOKEN * 4
            + VISION_END
            + "<0.8 seconds>"
            + VISION_START
            + VIDEO_TOKEN * 4
            + VISION_END
        )
        expected = prompt.replace(VIDEO_TOKEN, expansion, 1)
        span = VISION_START + expansion + VISION_END
        self.assertIn(span, expected)
        self.assertEqual(expected.count(VIDEO_TOKEN), 2 * 4)
        # image pair + video outer pair + 2 group pairs
        self.assertEqual(expected.count(VISION_START), 1 + 1 + 2)
        self.assertEqual(expected.count(VISION_END), 1 + 1 + 2)
        self.assertEqual(
            row.token_ids, tokenizer.encode(expected, add_special_tokens=False)
        )
        self.assertEqual(row.planned_tokens, len(row.token_ids) - 1 + 1024)
        self.assertEqual(
            [entry.kind for entry in row.media], ["image", "video_group", "video_group"]
        )

    def test_request_video_budget_is_a_whole_request_error(self):
        from rtp_llm.frontend.d3 import VIDEO_MAX_TOKENS, D3Video

        engine = executor(run_rows=_no_rows())
        big = D3Video(
            frames=None,
            fps=24.0,
            indices=list(range(32)),
            total_frames=480,
            height=1024,
            width=1024,
            timestamps=[0.5] * 16,
            group_tokens=2048,
            key="k",
        )
        with unittest.mock.patch("rtp_llm.frontend.d3.load_video", return_value=big):
            with self.assertRaises(D3RequestError) as caught:
                engine.prepare(
                    {
                        "questions": {"q": {"type": "noul"}},
                        "videos": ["data:video/mp4;base64,AAAA"],
                    }
                )
        self.assertIn(f"over the {VIDEO_MAX_TOKENS:,} tokens", str(caught.exception))


def _no_rows():
    async def run_rows(rows):
        return []

    return run_rows


if __name__ == "__main__":
    unittest.main()
