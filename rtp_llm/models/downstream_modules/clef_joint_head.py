"""Clef joint-schema decision head (port of Cloudflare/clef joint_schema_model.py).

The head reads the backbone's final hidden states of ONE packed sequence per
record -- system prompt + STATE: <state> + SCHEMA FIELDS (one FIELD block per
question with its option JSON spans) + an assistant suffix whose final ":"
token is the global anchor.  A small transformer routes evidence from the
state memory to every option of every question and scores them jointly.

Span transport: the renderer appends ``[sentinel]<decimal table>[sentinel]``
after the suffix (strictly after every consumed position, so real-token
hidden states are unchanged); the handler backward-scans for the sentinel
pair and validates the table before decoding geometry.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as functional

# Transcribed from joint_schema_model.py:24-29 with hex verification; the
# think markers are the pipe-less added tokens 248068/248069 -- never render
# them through tooling that may corrupt the bytes (see the clef plan §0).
CLEF_SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
CLEF_OPEN_THINK = "\x3cthink\x3e"  # tokenizer id 248068
CLEF_CLOSE_THINK = "\x3c/think\x3e"  # tokenizer id 248069
CLEF_SUFFIX = (
    "\n<|im_end|>\n<|im_start|>assistant\n"
    + CLEF_OPEN_THINK
    + "\n\n"
    + CLEF_CLOSE_THINK
    + "\n\nJOINT SCHEMA DECISIONS:"
)
CLEF_SCHEMA_HEADER = "\n\nSCHEMA FIELDS:\n"
CLEF_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
CLEF_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

CLEF_QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
CLEF_NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}

# Span-table field widths (decimal digits per field).
TABLE_BODY_LEN_WIDTH = 6
TABLE_N_QUESTIONS_WIDTH = 3
TABLE_TYPE_WIDTH = 1
TABLE_SPAN_WIDTH = 6
TABLE_N_OPTIONS_WIDTH = 3


def render_json(value: Any) -> str:
    """Reference render(): identity for strings, compact sorted JSON else."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def clef_question_options(question: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Reference question_options(): option (id, description) per type."""
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = dict(CLEF_NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted(
            (str(key), value) for key, value in question["criteria"].items()
        )
    if question_type == "score":
        return [(str(index), value) for index, value in enumerate(question["criteria"])]
    raise ValueError(f"unknown clef question type: {question_type!r}")


@dataclass(frozen=True)
class ClefEncodedQuestion:
    question_id: str
    question_type: int
    question_span: Tuple[int, int]
    option_spans: Tuple[Tuple[int, int], ...]
    option_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ClefEncodedRecord:
    input_ids: Tuple[int, ...]
    body_length: int  # tokens before the appended span table
    questions: Tuple[ClefEncodedQuestion, ...]


def _encode_fragment(tokenizer: Any, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _validate_clef_questions(items: List[Tuple[str, Dict[str, Any]]]) -> None:
    for question_id, question in items:
        qtype = question.get("type")
        if qtype not in CLEF_QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if qtype != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")


def _encode_question(
    tokens,
    question_index: int,
    question_id: str,
    question: Dict[str, Any],
    base_offset: int,
) -> Tuple[ClefEncodedQuestion, List[int]]:
    """Encode one FIELD block; returns (question geometry, schema ids).

    Spans are absolute positions within the schema segment (base_offset =
    tokens already emitted for the schema header and earlier fields); the
    caller shifts them by the prefix+state offset.
    """
    schema_ids: List[int] = []
    schema_ids.extend(
        tokens(
            f"\nFIELD {question_index + 1}\nID: {question_id}\n"
            f"TYPE: {question['type']}\nINSTRUCTION: "
        )
    )
    question_start = base_offset + len(schema_ids)
    instructions = question.get("instructions")
    if instructions is None or instructions == "":
        instructions = str(question_id)
    schema_ids.extend(tokens(render_json(instructions)))
    question_end = base_offset + len(schema_ids)
    schema_ids.extend(tokens("\nALLOWED OPTIONS:\n"))

    option_spans: List[Tuple[int, int]] = []
    option_ids: List[str] = []
    for option_index, (option_id, description) in enumerate(
        clef_question_options(question)
    ):
        schema_ids.extend(tokens(f"OPTION {option_index + 1}: "))
        option_start = base_offset + len(schema_ids)
        semantics: Dict[str, Any] = {"option_id": option_id}
        if description is not None:
            semantics["description"] = description
        schema_ids.extend(tokens(render_json(semantics)))
        option_spans.append((option_start, base_offset + len(schema_ids)))
        option_ids.append(option_id)
        schema_ids.extend(tokens("\n"))
    schema_ids.extend(tokens("END FIELD\n"))
    return (
        ClefEncodedQuestion(
            question_id=str(question_id),
            question_type=CLEF_QUESTION_TYPES[str(question["type"])],
            question_span=(question_start, question_end),
            option_spans=tuple(option_spans),
            option_ids=tuple(option_ids),
        ),
        schema_ids,
    )


def encode_clef_record(
    tokenizer: Any,
    record: Dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: Optional[int] = None,
    sentinel_token_id: int = 248055,
    table_digit_token_ids: Optional[List[int]] = None,
    escape: bool = True,
) -> ClefEncodedRecord:
    """Pack one record into tokens + geometry, with the span table appended.

    Fragments are tokenized independently (never the joined string): BPE
    merges at fragment edges would shift ids and invalidate spans.
    """
    if table_digit_token_ids is None:
        table_digit_token_ids = _resolve_digit_ids(tokenizer)

    def tokens(text: str) -> List[int]:
        if escape:
            text = text.replace("<|", "< |").replace("|>", "| >")
        return _encode_fragment(tokenizer, text)

    questions_input = record.get("questions")
    if isinstance(questions_input, dict):
        items = list(questions_input.items())
    else:
        items = [
            (str(q.get("id", index)), q)
            for index, q in enumerate(questions_input or [])
        ]
    if not items:
        raise ValueError("clef records need at least one question")
    _validate_clef_questions(items)

    schema_ids: List[int] = tokens(CLEF_SCHEMA_HEADER)
    questions: List[ClefEncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(items):
        encoded_question, question_ids = _encode_question(
            tokens, question_index, str(question_id), question, len(schema_ids)
        )
        questions.append(encoded_question)
        schema_ids.extend(question_ids)

    # Trusted literals (prefix/suffix) encode raw so their special tokens stay
    # intact; user-influenced fragments go through the escaping tokenizer.
    prefix_ids = _encode_fragment(
        tokenizer,
        "<|im_start|>system\n"
        + CLEF_SYSTEM_PROMPT
        + "<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _encode_fragment(tokenizer, CLEF_SUFFIX)
    state_ids = tokens(render_json(record.get("state", "")))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; "
            f"maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        ClefEncodedQuestion(
            question_id=q.question_id,
            question_type=q.question_type,
            question_span=(q.question_span[0] + schema_offset, q.question_span[1] + schema_offset),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in q.option_spans
            ),
            option_ids=q.option_ids,
        )
        for q in questions
    )
    body = prefix_ids + state_ids + schema_ids + suffix_ids
    if not body:
        raise ValueError("record produced no model input")
    table = encode_span_table(
        body_length=len(body),
        questions=shifted,
        sentinel_token_id=sentinel_token_id,
        table_digit_token_ids=table_digit_token_ids,
    )
    return ClefEncodedRecord(
        input_ids=tuple(body + table),
        body_length=len(body),
        questions=shifted,
    )


def _resolve_digit_ids(tokenizer: Any) -> List[int]:
    ids = []
    for digit in "0123456789":
        encoded = _encode_fragment(tokenizer, digit)
        if len(encoded) != 1:
            raise ValueError(
                f"digit {digit!r} is not a single token in this tokenizer"
            )
        ids.append(encoded[0])
    return ids


def encode_span_table(
    body_length: int,
    questions: Tuple[ClefEncodedQuestion, ...],
    sentinel_token_id: int,
    table_digit_token_ids: List[int],
) -> List[int]:
    """[sentinel] body_len n_q (type s e n_opts (s e)*) [sentinel]."""

    def number(value: int, width: int) -> List[int]:
        text = str(value).zfill(width)
        if len(text) != width:
            raise ValueError(f"span-table field overflows {width} digits: {value}")
        return [table_digit_token_ids[int(ch)] for ch in text]

    table: List[int] = [sentinel_token_id]
    table += number(body_length, TABLE_BODY_LEN_WIDTH)
    table += number(len(questions), TABLE_N_QUESTIONS_WIDTH)
    for question in questions:
        table += number(question.question_type, TABLE_TYPE_WIDTH)
        table += number(question.question_span[0], TABLE_SPAN_WIDTH)
        table += number(question.question_span[1], TABLE_SPAN_WIDTH)
        table += number(len(question.option_spans), TABLE_N_OPTIONS_WIDTH)
        for start, end in question.option_spans:
            table += number(start, TABLE_SPAN_WIDTH)
            table += number(end, TABLE_SPAN_WIDTH)
    table.append(sentinel_token_id)
    return table


@dataclass(frozen=True)
class DecodedTable:
    body_length: int
    questions: Tuple[ClefEncodedQuestion, ...]


def _decode_table_question(take, body_length: int) -> ClefEncodedQuestion:
    qtype = take(TABLE_TYPE_WIDTH)
    if qtype not in CLEF_QUESTION_TYPES.values():
        raise ValueError("clef span table has an unknown question type")
    start = take(TABLE_SPAN_WIDTH)
    end = take(TABLE_SPAN_WIDTH)
    n_options = take(TABLE_N_OPTIONS_WIDTH)
    if not (0 <= start < end <= body_length) or n_options < 1:
        raise ValueError("clef span table question geometry is invalid")
    spans = []
    for _ in range(n_options):
        ostart = take(TABLE_SPAN_WIDTH)
        oend = take(TABLE_SPAN_WIDTH)
        if not (start <= ostart < oend <= body_length):
            raise ValueError("clef span table option geometry is invalid")
        spans.append((ostart, oend))
    return ClefEncodedQuestion(
        question_id="",
        question_type=qtype,
        question_span=(start, end),
        option_spans=tuple(spans),
        option_ids=(),
    )


def decode_span_table(
    segment_ids: List[int],
    sentinel_token_id: int,
    table_digit_token_ids: List[int],
) -> Tuple[DecodedTable, int]:
    """Backward-scan the last sentinel pair and validate the table strictly.

    Returns (table, table_open_position).  Raises ValueError on any forgery
    signature: the last two sentinels must bracket a digit-only span that
    parses into geometrically consistent spans.
    """
    digit_to_value = {tid: i for i, tid in enumerate(table_digit_token_ids)}
    sentinel_positions = [
        i for i, t in enumerate(segment_ids) if t == sentinel_token_id
    ]
    if len(sentinel_positions) < 2:
        raise ValueError("clef span table not found in the record segment")
    table_open, table_close = sentinel_positions[-2], sentinel_positions[-1]
    table_end = table_open + 1
    raw = segment_ids[table_end:table_close]
    if not raw or any(t not in digit_to_value for t in raw):
        raise ValueError("clef span table contains non-digit tokens")
    digits = [digit_to_value[t] for t in raw]
    pos = 0

    def take(width: int) -> int:
        nonlocal pos
        if pos + width > len(digits):
            raise ValueError("clef span table truncated")
        value = 0
        for d in digits[pos:pos + width]:
            value = value * 10 + d
        pos += width
        return value

    body_length = take(TABLE_BODY_LEN_WIDTH)
    n_questions = take(TABLE_N_QUESTIONS_WIDTH)
    if not 1 <= n_questions <= 999:
        raise ValueError("clef span table has no questions")
    questions = [
        _decode_table_question(take, body_length) for _ in range(n_questions)
    ]
    if pos != len(digits):
        raise ValueError("clef span table has trailing digits")
    return DecodedTable(body_length=body_length, questions=tuple(questions)), table_open


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefJointHead(torch.nn.Module):
    """State-dict compatible port of clef's JointSchemaHead."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width, heads=heads, feedforward=feedforward, dropout=dropout
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        sequence_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        questions: Tuple[ClefEncodedQuestion, ...],
        output_embedding_weight: torch.Tensor,
    ) -> List[torch.Tensor]:
        """One record: sequence_hidden [seq, hidden] over real tokens only."""
        normalized_hidden = self.hidden_norm(sequence_hidden)
        memory = self.memory_projection(normalized_hidden).unsqueeze(0)
        global_vector = normalized_hidden[-1]
        question_vectors = torch.stack(
            [self._mean_span(normalized_hidden, q.question_span) for q in questions]
        )
        type_ids = torch.tensor(
            [q.question_type for q in questions], device=sequence_hidden.device
        )
        option_contexts: List[torch.Tensor] = []
        lexical_options: List[torch.Tensor] = []
        option_counts: List[int] = []
        for question in questions:
            context_vectors = torch.stack(
                [self._mean_span(normalized_hidden, span) for span in question.option_spans]
            )
            lexical_vectors = [
                output_embedding_weight[input_ids[start:end]].mean(dim=0)
                for start, end in question.option_spans
            ]
            lexical = torch.stack(lexical_vectors)
            option_contexts.append(context_vectors)
            lexical_options.append(lexical)
            option_counts.append(len(question.option_spans))

        option_queries = [
            self.option_context_projection(context)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors[i]).unsqueeze(0)
            for i, (context, lexical) in enumerate(zip(option_contexts, lexical_options))
        ]
        routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
        for layer in self.evidence_layers:
            routed_options = layer(routed_options, memory)
        routed_options = routed_options[0]
        split_options = list(torch.split(routed_options, option_counts, dim=0))

        base_fields = self.question_projection(question_vectors)
        option_summaries = []
        for field, options in zip(base_fields, split_options):
            routing_weights = torch.softmax(
                torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0
            )
            option_summaries.append(
                torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
            )
        fields = (
            base_fields
            + self.option_summary_norm(torch.stack(option_summaries))
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(type_ids)
        )
        fields = fields.unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        record_logits: List[torch.Tensor] = []
        for index, (field, question, lexical, routed) in enumerate(
            zip(fields, questions, lexical_options, split_options)
        ):
            anchor = functional.normalize(
                question_vectors[index] + global_vector, dim=-1
            )
            lexical_anchor = functional.normalize(lexical, dim=-1)
            prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
            prior = prior_scale * torch.matmul(lexical_anchor, anchor)
            options = self.option_norm(routed)
            repeated_field = field.unsqueeze(0).expand_as(options)
            cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
            features = torch.cat(
                [
                    repeated_field,
                    options,
                    repeated_field * options,
                    torch.abs(repeated_field - options),
                ],
                dim=-1,
            )
            residual = self.residual_scorer(features).squeeze(-1)
            joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
            joint = joint_scale * cosine + residual
            record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
        return record_logits


def systemone_answer(
    question: Dict[str, Any], probabilities: Dict[str, float]
) -> Dict[str, Any]:
    """Reference systemone_answer(): typed answer per question."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {
                option: round(probabilities[option], 4) for option in options
            },
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(
            sum(index * probabilities[level] for index, level in enumerate(levels)), 4
        ),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {
            level: round(probabilities[level], 4) for level in levels
        },
    }
