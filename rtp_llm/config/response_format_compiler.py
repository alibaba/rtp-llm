import copy
import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import (
    GRAMMAR_FIELD_NAMES,
    GrammarConstraint,
    has_bounded_region,
)
from rtp_llm.config.response_format import ResponseFormat, normalize_think_tag

if TYPE_CHECKING:
    from rtp_llm.config.generate_config import ThinkingMode


@dataclass(frozen=True)
class ReasoningFormat:
    """Resolved model/frontend syntax for the reasoning section."""

    tag_begin: Union[str, List[str], Dict[str, Any]]
    tag_end: Union[str, List[str], Dict[str, Any]]
    suffix: str = ""
    no_think_excludes: Tuple[str, ...] = ()
    # DISABLED hardening: the prompt is not inside an open think block, so the
    # reply must not open one. Compiles the no-think branch only (see
    # ResponseFormatPlan.compile).
    enforce_no_think: bool = False

    @classmethod
    def from_generate_env_config(cls, generate_env_config: Any) -> "ReasoningFormat":
        raw_end_tag = generate_env_config.think_end_tag
        normalized_end_tag = (
            normalize_think_tag(str(raw_end_tag)) if raw_end_tag is not None else ""
        )
        raw_token_id = generate_env_config.think_end_token_id
        token_id = -1 if raw_token_id is None else int(raw_token_id)
        if token_id != -1:
            return cls(
                tag_begin="",
                tag_end={"type": "token", "token": token_id},
                no_think_excludes=(normalized_end_tag,) if normalized_end_tag else (),
            )

        if raw_end_tag is None:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "think_end_tag is required when think_end_token_id is not set",
            )
        return cls(tag_begin="", tag_end=normalized_end_tag)

    def prefix_format(self, max_thinking_tokens: int) -> Dict[str, Any]:
        think_tag = {
            "type": "tag",
            "begin": self.tag_begin,
            "content": {
                "type": "any_text",
                "max_tokens": max_thinking_tokens,
            },
            "end": self.tag_end,
        }
        if not self.suffix:
            return think_tag
        return {
            "type": "sequence",
            "elements": [
                think_tag,
                {"type": "const_string", "value": self.suffix},
            ],
        }


@dataclass(frozen=True)
class ResponseFormatPlan:
    """Pure result of compiling the response format for one request."""

    final_constraint: Optional[GrammarConstraint]
    engine_constraint: Optional[GrammarConstraint]

    @classmethod
    def compile(
        cls,
        config: Any,
        reasoning_format: Optional[ReasoningFormat] = None,
    ) -> "ResponseFormatPlan":
        from rtp_llm.config.generate_config import ThinkingMode

        final_constraint = _resolve_final_constraint(config)
        thinking_mode = _resolved_thinking_mode(config)

        if thinking_mode in (ThinkingMode.ENABLED, ThinkingMode.ADAPTIVE):
            if config.has_num_beams() or config.num_return_sequences > 1:
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "thinking mode does not support beam search or "
                    "num_return_sequences > 1 because it uses "
                    "grammar-constrained decoding",
                )
            if reasoning_format is None:
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "reasoning_format is required for enabled or adaptive thinking",
                )
            final_format = (
                final_constraint.final_format_node()
                if final_constraint is not None
                else {"type": "any_text"}
            )
            if thinking_mode == ThinkingMode.ADAPTIVE:
                envelope = _adaptive_reasoning_envelope(
                    reasoning_format,
                    config.max_thinking_tokens,
                    final_format,
                )
            else:
                envelope = _reasoning_envelope(
                    reasoning_format,
                    config.max_thinking_tokens,
                    final_format,
                )
            engine_constraint = GrammarConstraint(
                "structural_tag",
                envelope,
            ).normalized()
        else:
            if final_constraint is not None and (
                config.has_num_beams() or config.num_return_sequences > 1
            ):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "grammar-constrained decoding does not support beam search or "
                    "num_return_sequences > 1",
                )
            engine_constraint = final_constraint
            if (
                thinking_mode == ThinkingMode.DISABLED
                and reasoning_format is not None
                and reasoning_format.enforce_no_think
            ):
                # Thinking is off and the prompt did not put the model inside a
                # think block, yet a hybrid reasoning checkpoint may still open
                # <think> by itself and spend the caller's whole max_new_tokens
                # inside it. The renderer can only re-route that text -- the
                # answer is gone once the budget is spent -- so keep the
                # boundary tags out of the grammar instead. This is the no-think
                # branch ADAPTIVE already compiles, on its own.
                #
                # The hardening is best-effort: it must never turn a servable
                # request into a failure, so every shape the envelope cannot wrap
                # keeps the caller's own grammar (or none) and is logged once.
                if config.has_num_beams() or config.num_return_sequences > 1:
                    _warn_skipped_no_think(
                        "multiple_sequences",
                        "skipping the no-think constraint: grammar-constrained "
                        "decoding does not support beam search or "
                        "num_return_sequences > 1",
                    )
                elif _has_bounded_final_format(final_constraint):
                    # A structural_tag carrying any_text/any_tokens max_tokens
                    # cannot be nested in the envelope; the caller's budgeted
                    # grammar stays in charge, exactly as before the hardening.
                    _warn_skipped_no_think(
                        "bounded_final_format",
                        "skipping the no-think constraint: the caller's "
                        "structural_tag already bounds an any_text/any_tokens "
                        "region, which the no-think envelope cannot wrap",
                    )
                else:
                    # Any other shape the envelope cannot wrap -- a legacy
                    # structural_tag ({"structures","triggers"}, no "format" node)
                    # is one -- must keep the caller's grammar too. Calling
                    # final_format_node() on it raises, and the hardening is
                    # best-effort, so it yields rather than turning a request that
                    # was servable before this branch existed into a failure.
                    try:
                        final_format = (
                            final_constraint.final_format_node()
                            if final_constraint is not None
                            else {"type": "any_text"}
                        )
                        engine_constraint = GrammarConstraint(
                            "structural_tag",
                            _no_think_only_envelope(reasoning_format, final_format),
                        ).normalized()
                    except FtRuntimeException:
                        engine_constraint = final_constraint
                        _warn_skipped_no_think(
                            "unwrappable_final_format",
                            "skipping the no-think constraint: the caller's grammar "
                            "is in a shape the no-think envelope cannot wrap, so it "
                            "stays in charge",
                        )

        return cls(final_constraint, engine_constraint)

    def apply_to_config(self, config: Any) -> None:
        config.response_format = None
        if self.engine_constraint is None:
            GrammarConstraint.clear_from_config(config)
        else:
            self.engine_constraint.apply_to_config(config)
        validate_engine_ready(config)


def _resolved_thinking_mode(config: Any) -> "ThinkingMode":
    """Resolve the legacy boolean without importing GenerateConfig at module load."""

    from rtp_llm.config.generate_config import ThinkingMode

    if config.thinking_mode == ThinkingMode.ADAPTIVE:
        return ThinkingMode.ADAPTIVE
    return ThinkingMode.ENABLED if config.in_think_mode else ThinkingMode.DISABLED


def _uses_reasoning_envelope(config: Any) -> bool:
    from rtp_llm.config.generate_config import ThinkingMode

    return _resolved_thinking_mode(config) in (
        ThinkingMode.ENABLED,
        ThinkingMode.ADAPTIVE,
    )


def _resolve_final_constraint(config: Any) -> Optional[GrammarConstraint]:
    response_format = config.response_format
    direct_constraints = GrammarConstraint.collect_from_config(config)

    if response_format is not None:
        if not isinstance(response_format, ResponseFormat):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "response_format must be a validated ResponseFormat object",
            )
        if direct_constraints:
            names = ", ".join(c.name for c in direct_constraints)
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"response_format conflicts with grammar field(s): {names}",
            )
        constraint = GrammarConstraint.from_response_format(response_format)
        return constraint.normalized() if constraint is not None else None

    for constraint in direct_constraints:
        constraint.validate_not_empty()
    if len(direct_constraints) > 1:
        names = " / ".join(GRAMMAR_FIELD_NAMES)
        raise FtRuntimeException(
            ExceptionType.UNSUPPORTED_OPERATION,
            f"only one grammar constraint ({names}) may be set per request",
        )
    if not direct_constraints:
        return None
    return direct_constraints[0].normalized()


def _reasoning_sequence(
    reasoning_format: ReasoningFormat,
    max_thinking_tokens: int,
    final_format: Dict[str, Any],
) -> Dict[str, Any]:
    reasoning_prefix = reasoning_format.prefix_format(max_thinking_tokens)
    if reasoning_prefix.get("type") == "sequence":
        elements = list(reasoning_prefix["elements"]) + [final_format]
    else:
        elements = [reasoning_prefix, final_format]
    return {"type": "sequence", "elements": elements}


def _reasoning_envelope(
    reasoning_format: ReasoningFormat,
    max_thinking_tokens: int,
    final_format: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": "structural_tag",
        "format": _reasoning_sequence(
            reasoning_format, max_thinking_tokens, final_format
        ),
    }


def _add_no_think_excludes(
    final_format: Dict[str, Any], reasoning_format: ReasoningFormat
) -> Dict[str, Any]:
    """Keep think boundary tags out of an unconstrained no-think branch."""

    result = copy.deepcopy(final_format)
    if result.get("type") not in ("any_text", "triggered_tags"):
        return result
    excludes = list(result.get("excludes") or [])
    boundaries = (
        reasoning_format.tag_begin,
        reasoning_format.tag_end,
        *reasoning_format.no_think_excludes,
    )
    for boundary in boundaries:
        if isinstance(boundary, str) and boundary and boundary not in excludes:
            excludes.append(boundary)
    if excludes:
        result["excludes"] = excludes
    return result


def _no_think_only_envelope(
    reasoning_format: ReasoningFormat,
    final_format: Dict[str, Any],
) -> Dict[str, Any]:
    """Answer-only grammar: free text, minus the think boundary tags.

    Unlike the ADAPTIVE envelope this has no think branch and no budget, so the
    model cannot re-open a think block at all -- EOS stays samplable throughout.
    """
    return {
        "type": "structural_tag",
        "format": _add_no_think_excludes(final_format, reasoning_format),
    }


def _has_bounded_final_format(
    final_constraint: Optional[GrammarConstraint],
) -> bool:
    """Whether the caller's grammar bounds a region the envelope cannot wrap.

    ``GrammarConstraint.final_format_node`` refuses such a structural_tag (the
    reasoning envelopes cannot nest a bounded any_text/any_tokens), so the
    no-think hardening must yield before that call instead of failing the request.
    """

    if final_constraint is None or final_constraint.name != "structural_tag":
        return False
    return has_bounded_region(final_constraint.value)


@functools.lru_cache(maxsize=None)
def _warn_skipped_no_think(reason: str, message: str) -> None:
    """Log one skipped hardening per reason per process.

    The conditions are request- or deployment-shaped, so a caller that always
    sends them would otherwise log a line per request. Tests clear the cache to
    observe the record regardless of execution order.
    """

    logging.warning(message)


def _adaptive_reasoning_envelope(
    reasoning_format: ReasoningFormat,
    max_thinking_tokens: int,
    final_format: Dict[str, Any],
) -> Dict[str, Any]:
    if not reasoning_format.tag_begin:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            "adaptive thinking requires a non-empty think start tag",
        )
    return {
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                _reasoning_sequence(
                    reasoning_format, max_thinking_tokens, final_format
                ),
                _add_no_think_excludes(final_format, reasoning_format),
            ],
        },
    }


def prepare_response_format(
    config: Any,
    reasoning_format: Optional[ReasoningFormat] = None,
) -> Optional[GrammarConstraint]:
    """Compile and install the engine constraint at request entry."""

    if config._reasoning_envelope_applied:
        final_constraint = config._reasoning_final_constraint
        if _uses_reasoning_envelope(config):
            validate_engine_ready(config)
            return final_constraint
        restore_final_constraint(config, final_constraint)
        return final_constraint

    plan = ResponseFormatPlan.compile(config, reasoning_format=reasoning_format)
    plan.apply_to_config(config)
    if _uses_reasoning_envelope(config):
        config._reasoning_envelope_applied = True
        config._reasoning_final_constraint = plan.final_constraint
        config._reasoning_format = reasoning_format
    return plan.final_constraint


def recompile_reasoning_envelope(config: Any) -> None:
    """Rebuild an installed reasoning envelope after its budget changes.

    Prompt length is only known at the backend boundary, after the original
    request-level grammar has been compiled. Keep the saved final constraint
    and reasoning syntax as the source for a fresh envelope so the scalar
    budget and the grammar consumed by the engine cannot diverge.
    """

    if not config._reasoning_envelope_applied:
        validate_engine_ready(config)
        return

    reasoning_format = config._reasoning_format
    if reasoning_format is None:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            "installed reasoning grammar is missing its reasoning format",
        )
    final_constraint = config._reasoning_final_constraint
    restore_final_constraint(config, final_constraint)
    prepare_response_format(config, reasoning_format=reasoning_format)


def validate_engine_ready(config: Any) -> None:
    """Read-only grammar assertion used at the RPC boundary."""

    if config.response_format is not None:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            "response_format must be prepared before engine serialization",
        )

    constraint = GrammarConstraint.resolve_from_config(config)
    if constraint is not None and constraint.normalized() != constraint:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{constraint.name} must be normalized before engine serialization",
        )


def restore_final_constraint(
    config: Any,
    constraint: Optional[GrammarConstraint],
) -> None:
    """Install the saved post-reasoning constraint for Dash SC phase 2."""

    config.response_format = None
    config._reasoning_envelope_applied = False
    config._reasoning_final_constraint = None
    config._reasoning_format = None
    if constraint is None:
        GrammarConstraint.clear_from_config(config)
    else:
        constraint.apply_to_config(config)
    validate_engine_ready(config)
