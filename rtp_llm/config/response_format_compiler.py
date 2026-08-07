from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import GRAMMAR_FIELD_NAMES, GrammarConstraint
from rtp_llm.config.response_format import ResponseFormat, normalize_think_tag


@dataclass(frozen=True)
class ReasoningFormat:
    """Resolved model/frontend syntax for the reasoning section."""

    tag_begin: Union[str, List[str], Dict[str, Any]]
    tag_end: Union[str, List[str], Dict[str, Any]]
    suffix: str = ""

    @classmethod
    def from_generate_env_config(cls, generate_env_config: Any) -> "ReasoningFormat":
        raw_token_id = generate_env_config.think_end_token_id
        token_id = -1 if raw_token_id is None else int(raw_token_id)
        if token_id != -1:
            return cls(
                tag_begin="",
                tag_end={"type": "token", "token": token_id},
            )

        raw_tag = generate_env_config.think_end_tag
        if raw_tag is None:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "think_end_tag is required when think_end_token_id is not set",
            )
        return cls(tag_begin="", tag_end=normalize_think_tag(str(raw_tag)))

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
        final_constraint = _resolve_final_constraint(config)

        if config.in_think_mode:
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
                    "reasoning_format is required when in_think_mode is enabled",
                )
            final_format = (
                final_constraint.final_format_node()
                if final_constraint is not None
                else {"type": "any_text"}
            )
            engine_constraint = GrammarConstraint(
                "structural_tag",
                _reasoning_envelope(
                    reasoning_format,
                    config.max_thinking_tokens,
                    final_format,
                ),
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
        config.response_format = None
        if self.engine_constraint is None:
            GrammarConstraint.clear_from_config(config)
        else:
            self.engine_constraint.apply_to_config(config)
        validate_engine_ready(config)


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


def _reasoning_envelope(
    reasoning_format: ReasoningFormat,
    max_thinking_tokens: int,
    final_format: Dict[str, Any],
) -> Dict[str, Any]:
    reasoning_prefix = reasoning_format.prefix_format(max_thinking_tokens)
    if reasoning_prefix.get("type") == "sequence":
        elements = list(reasoning_prefix["elements"]) + [final_format]
    else:
        elements = [reasoning_prefix, final_format]
    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": elements,
        },
    }


def prepare_response_format(
    config: Any,
    reasoning_format: Optional[ReasoningFormat] = None,
) -> Optional[GrammarConstraint]:
    """Compile and install the engine constraint at request entry."""

    if config._reasoning_envelope_applied:
        final_constraint = config._reasoning_final_constraint
        if config.in_think_mode:
            validate_engine_ready(config)
            return final_constraint
        restore_final_constraint(config, final_constraint)
        return final_constraint

    plan = ResponseFormatPlan.compile(config, reasoning_format=reasoning_format)
    plan.apply_to_config(config)
    if config.in_think_mode:
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
    if constraint is None:
        GrammarConstraint.clear_from_config(config)
    else:
        constraint.apply_to_config(config)
    validate_engine_ready(config)
