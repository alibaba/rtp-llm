import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import GRAMMAR_FIELD_NAMES, GrammarConstraint
from rtp_llm.config.response_format import ResponseFormat, normalize_think_tag

if TYPE_CHECKING:
    from rtp_llm.config.generate_config import ThinkingMode

GrammarScope = Literal["response_content", "response_tail", "full_turn"]


@dataclass(frozen=True)
class ReasoningFormat:
    """Resolved model/frontend syntax for the reasoning section."""

    tag_begin: Union[str, List[str], Dict[str, Any]]
    tag_end: Union[str, List[str], Dict[str, Any]]
    suffix: str = ""
    no_think_excludes: Tuple[str, ...] = ()
    content_excludes: Tuple[str, ...] = ()

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

    def with_boundary_excludes(
        self, tokenizer: Any, think_start_tag: str = ""
    ) -> "ReasoningFormat":
        """Reserve boundary prefixes for the tag, rather than thinking text.

        K3's first boundary tokens are <|open|> and <|close|>. Excluding
        those from AnyText leaves the enclosing tag as the only path after
        <|close|>, so the matcher must finish the complete configured end tag.
        Derive the prefixes from the tokenizer to also cover other models.
        """
        if tokenizer is None:
            return self
        excludes = list(self.content_excludes)
        for boundary in (think_start_tag, self.tag_begin, self.tag_end):
            boundaries = boundary if isinstance(boundary, list) else [boundary]
            for value in boundaries:
                prefix = _boundary_token_prefix(tokenizer, value)
                if prefix and prefix not in excludes:
                    excludes.append(prefix)
        return replace(self, content_excludes=tuple(excludes))

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
        if self.content_excludes:
            think_tag["content"]["excludes"] = list(self.content_excludes)
        if not self.suffix:
            return think_tag
        return {
            "type": "sequence",
            "elements": [
                think_tag,
                {"type": "const_string", "value": self.suffix},
            ],
        }


def _boundary_token_prefix(tokenizer: Any, boundary: Any) -> str:
    if not boundary:
        return ""
    if isinstance(boundary, dict):
        if boundary.get("type") != "token":
            return ""
        token = boundary["token"]
        if isinstance(token, str):
            return token
        token_ids = [token]
    else:
        token_ids = tokenizer.encode(boundary, add_special_tokens=False)
    if not token_ids:
        return ""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return boundary if isinstance(boundary, str) else ""
    prefix = decode(token_ids[:1], skip_special_tokens=False)
    if isinstance(boundary, str) and (not prefix or not boundary.startswith(prefix)):
        # SentencePiece/byte decoders may not round-trip an isolated token.
        # In that case retain the literal boundary instead of excluding other text.
        return boundary
    return prefix


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

        final_constraint = resolve_final_constraint(config)
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

        return cls(final_constraint, engine_constraint)

    def apply_to_config(self, config: Any) -> None:
        if self.engine_constraint is not None and (
            config.has_num_beams() or config.num_return_sequences > 1
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "grammar-constrained decoding does not support beam search or "
                "num_return_sequences > 1",
            )
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


def resolve_final_constraint(config: Any) -> Optional[GrammarConstraint]:
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
    prepared_plan: Optional[ResponseFormatPlan] = None,
) -> Optional[GrammarConstraint]:
    """Compile and install the engine constraint at request entry."""

    if prepared_plan is not None:
        prepared_plan.apply_to_config(config)
        config._model_response_format_plan = prepared_plan
        config._reasoning_envelope_applied = False
        config._reasoning_final_constraint = None
        return prepared_plan.final_constraint
    if config._model_response_format_plan is not None:
        validate_engine_ready(config)
        return config._model_response_format_plan.final_constraint

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
    return plan.final_constraint


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
    config._model_response_format_plan = None
    if constraint is None:
        GrammarConstraint.clear_from_config(config)
    else:
        constraint.apply_to_config(config)
    validate_engine_ready(config)
