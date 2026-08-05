from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import (
    GrammarConstraint,
    dump_compact_json,
    load_json_field,
    normalize_grammar_value,
)
from rtp_llm.config.response_format import normalize_think_tag, parse_response_format


@dataclass(frozen=True)
class ReasoningFormat:
    """Server/model resolved reasoning envelope format used for grammar wrapping."""

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
                tag_end={"type": "token", "token": int(token_id)},
            )
        raw_tag = generate_env_config.think_end_tag
        if raw_tag is None:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "think_end_tag is required when think_end_token_id is not set",
            )
        tag = normalize_think_tag(str(raw_tag))
        return cls(tag_begin="", tag_end=tag)

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


class ResponseFormatBuilder:
    """Normalize response_format and typed grammar fields in-place on GenerateConfig."""

    def __init__(self, config: Any, reasoning_format: Optional[ReasoningFormat] = None):
        self.config = config
        self.reasoning_format = reasoning_format

    def apply(self) -> Optional[GrammarConstraint]:
        if self.config._reasoning_envelope_applied:
            saved_constraint = self.config._reasoning_final_constraint
            if self.config.in_think_mode:
                return saved_constraint
            self.restore_final_constraint(self.config, saved_constraint)
            return saved_constraint

        if self.config.in_think_mode and (
            self.config.has_num_beams() or self.config.num_return_sequences > 1
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "thinking mode does not support beam search or "
                "num_return_sequences > 1 because it uses grammar-constrained decoding",
            )

        self.config._reasoning_final_constraint = None
        constraint = self._resolve_grammar_constraint()

        if not self.config.in_think_mode:
            return constraint

        if self.reasoning_format is None:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "reasoning_format is required when in_think_mode is enabled",
            )

        if constraint is not None:
            self._wrap_grammar_with_reasoning_envelope(constraint)
        else:
            self._wrap_final_format_with_reasoning_envelope({"type": "any_text"})
        self.config._reasoning_envelope_applied = True
        self.config._reasoning_final_constraint = constraint
        return constraint

    def finalize(self) -> Optional[GrammarConstraint]:
        """Apply response-format projection and verify the engine-ready state."""
        constraint = self.apply()
        self.validate_finalized(self.config)
        return constraint

    @classmethod
    def validate_finalized(cls, config: Any) -> None:
        if config.json_format:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "json_format must be finalized before engine serialization",
            )
        if config.response_format is not None:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "response_format must be finalized before engine serialization",
            )

        constraints = GrammarConstraint.collect_from_config(config)
        cls(config)._validate_grammar_constraints(constraints)
        for constraint in constraints:
            if not isinstance(constraint.value, str):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    f"{constraint.name} must be finalized to a string before engine serialization",
                )

    @classmethod
    def restore_final_constraint(
        cls,
        config: Any,
        constraint: Optional[GrammarConstraint],
    ) -> None:
        """Restore an explicitly saved post-reasoning output constraint.

        Dash SC uses this for its exceptional phase-2 retry. This method does
        not inspect an arbitrary structural-tag AST and therefore cannot
        mistake a caller-provided format for a server-built reasoning
        envelope.
        """
        config.response_format = None
        config.json_format = False
        config.json_schema = None
        config.regex = None
        config.ebnf = None
        config.structural_tag = None
        config._reasoning_envelope_applied = False
        config._reasoning_final_constraint = None

        if constraint is None:
            cls.validate_finalized(config)
            return

        normalized = constraint.normalized()
        if normalized.name == "json_schema":
            config.json_schema = normalized.value
        elif normalized.name == "regex":
            config.regex = normalized.value
        elif normalized.name == "ebnf":
            config.ebnf = normalized.value
        elif normalized.name == "structural_tag":
            config.structural_tag = normalized.value
        else:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"unsupported grammar field {normalized.name}",
            )
        cls.validate_finalized(config)

    def _project_response_format_to_grammar_fields(self) -> None:
        """Project response_format onto typed fields and clear it; rf wins over stale extra_configs grammar."""
        raw_response_format = self.config.response_format
        if raw_response_format is None:
            return

        try:
            rf = parse_response_format(raw_response_format)
        except RecursionError as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "response_format exceeds the supported JSON nesting depth",
            ) from e
        except JSONDecodeError as e:
            if isinstance(raw_response_format, str) and raw_response_format.lstrip().startswith(
                ("{", "[")
            ):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    f"response_format invalid: {str(e)}",
                ) from e
            # Legacy GenerateConfig accepts arbitrary strings as plain text.
            rf = parse_response_format("text")
        except (ValidationError, TypeError) as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"response_format invalid: {str(e)}",
            )

        self.config.response_format = rf
        if rf is None:
            return

        constraint = GrammarConstraint.from_response_format(rf)
        self.config.response_format = None
        self.config.json_format = False
        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
        self.config.structural_tag = None

        if constraint is None:
            return

        normalized = constraint.normalized()
        if normalized.name == "json_schema":
            self.config.json_schema = normalized.value
        elif normalized.name == "regex":
            self.config.regex = normalized.value
        elif normalized.name == "ebnf":
            self.config.ebnf = normalized.value
        elif normalized.name == "structural_tag":
            self.config.structural_tag = normalized.value
        else:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"unsupported grammar field {normalized.name}",
            )

    def _project_legacy_json_format(self) -> None:
        if not self.config.json_format or self.config.response_format is not None:
            return
        if (
            self.config.json_schema is None
            and self.config.regex is None
            and self.config.ebnf is None
            and self.config.structural_tag is None
        ):
            self.config.json_schema = {"type": "object"}
        self.config.json_format = False

    def _resolve_grammar_constraint(self) -> Optional[GrammarConstraint]:
        self._project_response_format_to_grammar_fields()
        self._project_legacy_json_format()
        if self.config.json_schema is not None:
            self.config.json_schema = normalize_grammar_value(
                "json_schema", self.config.json_schema
            )
        if self.config.regex is not None:
            self.config.regex = normalize_grammar_value("regex", self.config.regex)
        if self.config.ebnf is not None:
            self.config.ebnf = normalize_grammar_value("ebnf", self.config.ebnf)
        if self.config.structural_tag is not None:
            structural_tag = load_json_field(
                "structural_tag", self.config.structural_tag
            )
            if (
                isinstance(structural_tag, dict)
                and "type" not in structural_tag
                and (
                    "format" in structural_tag
                    or ("structures" in structural_tag and "triggers" in structural_tag)
                )
            ):
                structural_tag = {"type": "structural_tag", **structural_tag}
            self.config.structural_tag = normalize_grammar_value(
                "structural_tag", structural_tag
            )
        constraints = GrammarConstraint.collect_from_config(self.config)
        self._validate_grammar_constraints(constraints)
        if not constraints:
            return None
        return constraints[0]

    def _validate_grammar_constraints(
        self, constraints: List[GrammarConstraint]
    ) -> None:
        for constraint in constraints:
            constraint.validate_not_empty()

        if len(constraints) > 1:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "only one grammar constraint (json_schema / regex / ebnf / "
                "structural_tag) may be set per request",
            )

    def _wrap_grammar_with_reasoning_envelope(
        self, constraint: GrammarConstraint
    ) -> None:
        final_format = constraint.final_format_node()
        self._wrap_final_format_with_reasoning_envelope(final_format)

    def _wrap_final_format_with_reasoning_envelope(
        self, final_format: Dict[str, Any]
    ) -> None:
        assert self.reasoning_format is not None
        reasoning_prefix = self.reasoning_format.prefix_format(
            self.config.max_thinking_tokens
        )
        if reasoning_prefix.get("type") == "sequence":
            elements = list(reasoning_prefix["elements"]) + [final_format]
        else:
            elements = [reasoning_prefix, final_format]
        envelope = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": elements,
            },
        }
        self.config.structural_tag = dump_compact_json(envelope)
        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
