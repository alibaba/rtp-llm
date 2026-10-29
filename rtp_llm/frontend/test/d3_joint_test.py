"""Joint inline-readout protocol tests (frontend executor, official contract)."""

import asyncio
import math
import unittest

from rtp_llm.frontend.d3 import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_BATCH_TOKENS,
    JOINT_MARKER,
    JOINT_SYSTEM_PROMPT,
    MAX_OPTIONS,
    D3RequestError,
    D3RowResult,
    D3Executor,
    _softmax,
    candidate_codes,
    joint_answer,
    joint_messages,
    joint_user_text,
    normalize_question,
)
from rtp_llm.models.downstream_modules.d3_config import D3Config, D3JointConfig


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
        self.rendered.append(rendered)
        return rendered


class JointTokenizer(FakeTokenizer):
    """FakeTokenizer whose marker and chat tokens are single tokens."""

    all_special_tokens = (JOINT_MARKER, "<|im_start|>", "<|im_end|>")

    def __init__(self):
        super().__init__()
        self.special_ids = {
            token: 900_000 + index
            for index, token in enumerate(self.all_special_tokens)
        }

    def _encode(self, text):
        ids: list[int] = []
        plain = ""
        specials = sorted(self.all_special_tokens, key=len, reverse=True)
        index = 0
        while index < len(text):
            for token in specials:
                if text.startswith(token, index):
                    if plain:
                        ids.extend(super()._encode(plain))
                        plain = ""
                    ids.append(self.special_ids[token])
                    index += len(token)
                    break
            else:
                plain += text[index]
                index += 1
        if plain:
            ids.extend(super()._encode(plain))
        return ids

    def convert_ids_to_tokens(self, token_id):
        for token, value in self.special_ids.items():
            if value == token_id:
                return token
        return None


def joint_config(tokenizer=None, **overrides):
    tokenizer = tokenizer or JointTokenizer()
    codes = candidate_codes()[:MAX_OPTIONS]
    token_ids = [
        tokenizer.encode(code, add_special_tokens=False)[0] for code in codes
    ]
    values = dict(
        format_version=2,
        format_id="joint-question-inline-readout-v1",
        codes=tuple(codes),
        token_ids=tuple(token_ids),
        temperature=1.0,
        attention_mode="noncausal_full_attention",
        pooling="last",
        max_length=262144,
        readout_dtype="float32",
        hidden_size=2560,
        joint=D3JointConfig(
            marker=JOINT_MARKER,
            readout_offset=0,
            max_questions=None,
            max_pixels=None,
        ),
        max_pixels=None,
    )
    values.update(
        {
            key: value
            for key, value in overrides.items()
            if key in D3Config.__dataclass_fields__
        }
    )
    return D3Config(**values)


def joint_executor(tokenizer=None, run_rows=None, **overrides):
    return D3Executor(
        joint_config(tokenizer, **overrides),
        run_rows or (lambda rows: []),
        tokenizer=tokenizer or JointTokenizer(),
        model_name="pplx-decider",
        max_row_tokens=overrides.get("max_row_tokens"),
        batch_rows=overrides.get("batch_rows", DEFAULT_BATCH_ROWS),
        batch_tokens=overrides.get("batch_tokens", DEFAULT_BATCH_TOKENS),
        timeout_seconds=overrides.get("timeout_seconds"),
    )


def image_url(width, height, fmt="PNG"):
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (width, height), (96, 160, 224))
    with io.BytesIO() as buffer:
        image.save(buffer, format=fmt)
        data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"


def choice_question(**criteria):
    return {"type": "choice", "criteria": criteria}


class JointPromptTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = JointTokenizer()
        self.escapes = tuple(
            sorted(
                {*self.tokenizer.all_special_tokens, JOINT_MARKER},
                key=len,
                reverse=True,
            )
        )

    def question(self, kind="choice"):
        if kind == "noul":
            return normalize_question({"type": "noul"})
        return normalize_question(
            {"type": "choice", "criteria": {"alpha": "first", "beta": None}}
        )

    def test_joint_user_text_matches_the_official_layout(self):
        text = joint_user_text(
            "the state",
            [self.question(), self.question("noul")],
            ("A", "B"),
            self.escapes,
        )
        self.assertEqual(
            text,
            "State:\nthe state\n\n"
            "Question 0:\nChoose the best matching option.\nOptions:\n"
            "A: alpha: first\nB: beta\n"
            + JOINT_MARKER
            + "\n\n"
            "Question 1:\nChoose the best matching option.\nOptions:\n"
            "A: No / false\nB: Yes / true\n"
            + JOINT_MARKER,
        )

    def test_joint_user_text_uses_question_instructions_and_escapes(self):
        question = normalize_question(
            {
                "type": "choice",
                "instructions": "Pick for <|fim_suffix|> analysis",
                "criteria": {"a": "x <|im_start|> y", "b": None},
            }
        )
        text = joint_user_text(
            "state with <|fim_suffix|> inside",
            [question],
            ("A", "B"),
            self.escapes,
        )
        self.assertIn("Question 0:\nPick for ＜|fim_suffix|＞ analysis", text)
        self.assertIn("A: a: x ＜|im_start|＞ y\nB: b\n" + JOINT_MARKER, text)
        self.assertIn("State:\nstate with ＜|fim_suffix|＞ inside", text)

    def test_joint_messages_place_images_before_the_text(self):
        questions = [self.question()]
        messages = joint_messages(
            "s", questions, ("A", "B"), self.escapes, 2
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": JOINT_SYSTEM_PROMPT})
        content = messages[1]["content"]
        self.assertEqual(
            content,
            [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": joint_user_text("s", questions, ("A", "B"), self.escapes)},
            ],
        )


class JointAnswerTest(unittest.TestCase):
    def test_choice_margin_confidence(self):
        question = normalize_question(choice_question(a=None, b=None))
        answer = joint_answer(question, [0.7, 0.3])
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "a")
        self.assertAlmostEqual(answer["confidence"], 0.4, places=12)

    def test_single_option_choice_confidence_is_one(self):
        question = normalize_question(choice_question(only=None))
        answer = joint_answer(question, [1.0])
        self.assertEqual(answer["choice"], "only")
        self.assertEqual(answer["confidence"], 1.0)

    def test_noul_reports_true_probability(self):
        question = normalize_question({"type": "noul"})
        answer = joint_answer(question, [0.25, 0.75])
        self.assertEqual(answer["noul"], 0.75)

    def test_score_distance_confidence(self):
        question = normalize_question(
            {"type": "score", "criteria": ["low", "mid", "high"]}
        )
        answer = joint_answer(question, [0.1, 0.6, 0.3])
        self.assertEqual(answer["type"], "score")
        self.assertAlmostEqual(answer["score"], 1.2, places=12)
        self.assertAlmostEqual(answer["confidence"], 0.4, places=12)
        self.assertEqual(answer["legend"], {"0": "low", "1": "mid", "2": "high"})


class JointPrepareTest(unittest.TestCase):
    def request(self, **questions):
        return {"state": "s", "questions": questions or {"q": choice_question(a=None, b=None)}}

    def test_one_chunk_holds_every_question(self):
        executor = joint_executor()
        prepared = executor.prepare_joint(
            {
                "state": "s",
                "questions": {
                    "first": choice_question(a=None, b=None),
                    "second": {"type": "noul"},
                },
            }
        )
        self.assertEqual(len(prepared.rows), 1)
        row = prepared.rows[0]
        self.assertEqual(row.question_ids, ("first", "second"))
        self.assertEqual(row.option_counts, (2, 2))
        markers = [
            index
            for index, token in enumerate(row.token_ids)
            if token == executor.marker_id
        ]
        self.assertEqual(len(markers), 2)

    def test_over_limit_requests_bisect_into_chunks(self):
        executor = joint_executor(max_row_tokens=450)
        prepared = executor.prepare_joint(
            {
                "state": "a somewhat longer state to force a split",
                "questions": {
                    "a": choice_question(x=None, y=None),
                    "b": choice_question(x=None, y=None),
                    "c": choice_question(x=None, y=None),
                    "d": choice_question(x=None, y=None),
                },
            }
        )
        self.assertGreater(len(prepared.rows), 1)
        seen = [qid for row in prepared.rows for qid in row.question_ids]
        self.assertEqual(seen, ["a", "b", "c", "d"])
        for row in prepared.rows:
            self.assertLessEqual(row.planned_tokens, 450)
            self.assertEqual(
                len(
                    [
                        token
                        for token in row.token_ids
                        if token == executor.marker_id
                    ]
                ),
                len(row.question_ids),
            )

    def test_single_question_over_limit_fails_the_request(self):
        executor = joint_executor(max_row_tokens=8)
        with self.assertRaises(D3RequestError):
            executor.prepare_joint(self.request())

    def test_videos_are_rejected(self):
        executor = joint_executor()
        with self.assertRaises(D3RequestError):
            executor.prepare_joint(
                {"state": "s", "questions": {"q": choice_question(a=None)}, "videos": ["x"]}
            )

    def test_score_levels_beyond_ten_are_valid(self):
        executor = joint_executor()
        levels = [f"level {index}" for index in range(12)]
        prepared = executor.prepare_joint(
            {"state": "s", "questions": {"q": {"type": "score", "criteria": levels}}}
        )
        self.assertEqual(len(prepared.rows), 1)
        self.assertEqual(prepared.rows[0].option_counts, (12,))

    def test_invalid_questions_reject_the_whole_request(self):
        executor = joint_executor()
        with self.assertRaises(D3RequestError) as caught:
            executor.prepare_joint(
                {
                    "state": "s",
                    "questions": {"q": {"type": "unknown"}},
                    "extra": 1,
                }
            )
        self.assertIn("not understood", str(caught.exception))

    def test_images_ride_on_every_chunk(self):
        url = image_url(64, 64)
        executor = joint_executor(max_row_tokens=450)
        questions = {
            f"q{index}": choice_question(a=None, b=None) for index in range(4)
        }
        prepared = executor.prepare_joint(
            {
                "state": "a longer state to force splits",
                "questions": questions,
                "images": [url],
            }
        )
        self.assertGreater(len(prepared.rows), 1)
        for row in prepared.rows:
            # Every chunk keeps the full state and the image.
            self.assertEqual(len(row.media), 1)
            # The image placeholder is one token in token_ids; the visual
            # budget enters planned_tokens.
            self.assertGreater(row.planned_tokens, len(row.token_ids))
            self.assertEqual(
                len(
                    [
                        token
                        for token in row.token_ids
                        if token == executor.marker_id
                    ]
                ),
                len(row.question_ids),
            )
        rendered = "".join(executor.tokenizer.rendered)
        # Every chunk render carries exactly one image placeholder.
        self.assertEqual(rendered.count("<|image_pad|>"), len(executor.tokenizer.rendered))


class JointExecutorTest(unittest.TestCase):
    def run_rows_from_logits(self, calls, blocks):
        """Return a run_rows callback: each row gets its questions' 255-logit blocks."""

        async def run_rows(rows):
            calls.append([tuple(row.question_ids) for row in rows])
            results = []
            widest = max(len(row.question_ids) for row in rows)
            for row in rows:
                logits: list[float] = []
                for index in range(widest):
                    if index < len(blocks) and index < len(row.question_ids):
                        logits.extend(blocks[index])
                    else:
                        logits.extend([0.0] * MAX_OPTIONS)
                results.append(
                    D3RowResult(logits=logits, prompt_tokens=row.planned_tokens)
                )
            return results

        return run_rows

    def test_answers_slice_each_questions_own_block(self):
        calls: list = []
        blocks = [[5.0] + [0.0] * 254, [1.0] + [0.0] * 254]
        executor = joint_executor(
            run_rows=self.run_rows_from_logits(calls, blocks), temperature=1.0
        )
        response = asyncio.run(
            executor.systemone(
                {
                    "state": "s",
                    "questions": {
                        "first": choice_question(a=None, b=None),
                        "second": choice_question(x=None, y=None),
                    },
                }
            )
        )
        self.assertEqual(response["model"], "pplx-decider")
        expected_first = _softmax([5.0, 0.0])
        expected_second = _softmax([1.0, 0.0])
        self.assertEqual(response["answers"]["first"]["choice"], "a")
        self.assertAlmostEqual(
            response["answers"]["first"]["probabilities"]["a"],
            expected_first[0],
            places=12,
        )
        self.assertEqual(response["answers"]["second"]["choice"], "x")
        self.assertAlmostEqual(
            response["answers"]["second"]["probabilities"]["x"],
            expected_second[0],
            places=12,
        )
        self.assertEqual(response["usage"]["output_tokens"], 0)
        self.assertEqual(response["usage"]["sequences"], 1)
        self.assertEqual(
            response["usage"]["input_tokens"],
            sum(len(row.token_ids) for row in executor.prepare_joint(
                {
                    "state": "s",
                    "questions": {
                        "first": choice_question(a=None, b=None),
                        "second": choice_question(x=None, y=None),
                    },
                }
            ).rows),
        )

    def test_bisected_requests_count_every_chunk_once(self):
        calls: list = []
        blocks = [[3.0] + [0.0] * 254]
        executor = joint_executor(
            run_rows=self.run_rows_from_logits(calls, blocks),
            max_row_tokens=450,
            batch_rows=1,
        )
        questions = {
            f"q{index}": choice_question(a=None, b=None) for index in range(4)
        }
        response = asyncio.run(
            executor.systemone({"state": "a longer state to force splits", "questions": questions})
        )
        self.assertGreater(len(calls), 1)
        self.assertEqual(response["usage"]["sequences"], len(calls))
        for index in range(4):
            self.assertEqual(response["answers"][f"q{index}"]["choice"], "a")

    def test_temperature_scales_before_softmax(self):
        calls: list = []
        blocks = [[math.log(3.0)] + [0.0] * 254]
        executor = joint_executor(
            run_rows=self.run_rows_from_logits(calls, blocks), temperature=2.0
        )
        response = asyncio.run(
            executor.systemone(
                {"state": "s", "questions": {"q": choice_question(a=None, b=None)}}
            )
        )
        expected = _softmax([math.log(3.0) / 2.0, 0.0])
        self.assertAlmostEqual(
            response["answers"]["q"]["probabilities"]["a"], expected[0], places=12
        )


if __name__ == "__main__":
    unittest.main()
