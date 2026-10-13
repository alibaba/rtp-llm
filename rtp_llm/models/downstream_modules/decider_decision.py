"""Decider (Mapika/decider-35b-a3b) decision encoding: state-first layout.

Port of the checkpoint's decider/prompt.py ``build`` (state_first) and
model.py ``slot_logits``: one packed sequence carries the context, then one
block per question (``Question k: ...\\nOptions:\\n(A) ...``) ending in its
``Answer k: (`` slot.  The readout is the backbone lm_head rows at the 255
single-token option letters, taken at each slot's final " (" token; the
question's own options mask the logits first-K and a calibrated temperature
is applied before the softmax.

Slot transport reuses the clef sentinel span table: the renderer appends
``[sentinel]<table>[sentinel]`` after the last slot (strictly after every
consumed position), the handler backward-scans and decodes each question's
slot position from its question span.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DECIDER_MAX_OPTIONS = 255
DECIDER_NARROW = 10  # A..J render inline "(A) text"; beyond that the wide form
DECIDER_LETTERS = "ABCDEFGHIJ"
DECIDER_CONTEXT_HEADER = "Context:\n"

# The cache holds the tokenizer itself as part of the key so an id() reuse
# after garbage collection can never hit another tokenizer's table.
LABELS_CACHE: Dict[Tuple[int, Any], Tuple[List[str], List[int]]] = {}


def decider_label_table(tokenizer: Any) -> Tuple[List[str], List[int]]:
    """255 single-token uppercase labels (A..Z, AA..) — prompt.py label_table."""
    cache_key = (id(tokenizer), tokenizer)
    if cache_key in LABELS_CACHE:
        return LABELS_CACHE[cache_key]
    import string

    upper = string.ascii_uppercase
    names = list(upper) + [a + b for a in upper for b in upper]
    out: List[Tuple[str, int]] = []
    for name in names:
        ids = tokenizer(name, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append((name, ids[0]))
        if len(out) == DECIDER_MAX_OPTIONS:
            break
    if len(out) != DECIDER_MAX_OPTIONS:
        raise ValueError(
            f"the checkpoint tokenizer only yields {len(out)} single-token labels"
        )
    table = ([n for n, _ in out], [i for _, i in out])
    LABELS_CACHE[cache_key] = table
    return table


@dataclass(frozen=True)
class DeciderEncodedRecord:
    input_ids: Tuple[int, ...]
    body_length: int
    slots: Tuple[int, ...]  # absolute position of each slot's final token


def encode_decider_record(
    tokenizer: Any,
    context: str,
    questions: List[Dict[str, Any]],
    max_ctx_tokens: int = 1536,
    sentinel_token_id: int = 248055,
    table_digit_token_ids: Optional[List[int]] = None,
    escape: bool = True,
) -> DeciderEncodedRecord:
    """One packed record; per-question slot positions ride the span table."""
    from rtp_llm.models.downstream_modules.clef_joint_head import (
        encode_span_table,
        _resolve_digit_ids,
    )
    from rtp_llm.models.downstream_modules.decision_module import (
        ClefEncodedQuestion,
    )

    if table_digit_token_ids is None:
        table_digit_token_ids = _resolve_digit_ids(tokenizer)
    labels, label_ids = decider_label_table(tokenizer)

    def tokens(text: str) -> List[int]:
        if escape:
            text = text.replace("<|", "< |").replace("|>", "| >")
        return tokenizer(text, add_special_tokens=False).input_ids

    if not questions:
        raise ValueError("decider records need at least one question")
    for question in questions:
        options = question.get("options") or []
        if not 2 <= len(options) <= DECIDER_MAX_OPTIONS:
            raise ValueError(
                f"decider questions need 2..{DECIDER_MAX_OPTIONS} options, "
                f"got {len(options)}"
            )

    multi = len(questions) > 1
    ids: List[int] = tokens(DECIDER_CONTEXT_HEADER) + tokens(context)
    ids = ids[: len(tokens(DECIDER_CONTEXT_HEADER)) + max_ctx_tokens]

    slots: List[int] = []
    encoded_questions: List[ClefEncodedQuestion] = []
    for k, question in enumerate(questions):
        options = question["options"]
        head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {question['question']}\nOptions:"
        tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
        piece: List[int] = tokens(head)
        if len(options) <= DECIDER_NARROW:
            lines = "".join(
                f"\n({DECIDER_LETTERS[j]}) {option}" for j, option in enumerate(options)
            )
            piece += tokens(lines)
        else:
            # wide form: "\n(" + <one label token> + ") <option text>"
            open_ids = tokens("\n(")
            for j, option in enumerate(options):
                piece += open_ids + [label_ids[j]] + tokens(f") {option}")
        piece += tokens(tail)
        # the slot is the final " (" token of "Answer k: ("
        slot_abs = len(ids) + len(piece) - 1
        slots.append(slot_abs)
        encoded_questions.append(
            ClefEncodedQuestion(
                question_id=str(question.get("id", k)),
                question_type=1,
                # the span table validates at least one option span per
                # question; decider reuses the slot position for both (the
                # handler reads question_span[0]).
                question_span=(slot_abs, slot_abs + 1),
                option_spans=((slot_abs, slot_abs + 1),),
                option_ids=(),
            )
        )
        ids.extend(piece)

    body_length = len(ids)
    table = encode_span_table(
        body_length=body_length,
        questions=tuple(encoded_questions),
        sentinel_token_id=sentinel_token_id,
        table_digit_token_ids=table_digit_token_ids,
    )
    return DeciderEncodedRecord(
        input_ids=tuple(ids + table),
        body_length=body_length,
        slots=tuple(slots),
    )
