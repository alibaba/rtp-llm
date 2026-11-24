import base64
import binascii
import math
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

INT32_MAX = 2_147_483_647
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
MAX_MESSAGES = 10_000
MAX_TOOLS = 1024
MAX_STOP_SEQUENCES = 16
MAX_STOP_SEQUENCE_LEN = 1024
MAX_IMAGE_URL_LEN = 8192
MAX_BASE64_IMAGE_CHARS = 28_000_000
_IMAGE_MEDIA_TYPE_RE = re.compile(r"^image/[a-zA-Z0-9.+_-]+$")


class AnthropicTextBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicSystemTextBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicImageSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None

    @model_validator(mode="after")
    def _validate_source(self) -> "AnthropicImageSource":
        if self.type == "base64":
            if (
                not self.media_type
                or not isinstance(self.media_type, str)
                or not _IMAGE_MEDIA_TYPE_RE.match(self.media_type)
            ):
                raise ValueError(
                    "image base64 source requires a valid image media_type (e.g. 'image/png')"
                )
            if not self.data or not isinstance(self.data, str):
                raise ValueError("image base64 source requires non-empty 'data'")
            cleaned = re.sub(r"\s+", "", self.data)
            if not cleaned:
                raise ValueError("image base64 source requires non-empty 'data'")
            if len(cleaned) > MAX_BASE64_IMAGE_CHARS:
                raise ValueError(
                    f"image base64 data exceeds max length of {MAX_BASE64_IMAGE_CHARS} chars"
                )
            try:
                base64.b64decode(cleaned.encode("ascii"), validate=True)
            except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
                raise ValueError(
                    "image base64 source contains invalid base64 data"
                ) from exc
            self.data = cleaned
        elif self.type == "url":
            if not self.url or not isinstance(self.url, str):
                raise ValueError("image url source requires non-empty 'url'")
            if len(self.url) > MAX_IMAGE_URL_LEN:
                raise ValueError(
                    f"image url exceeds max length of {MAX_IMAGE_URL_LEN} chars"
                )
            parsed = urlparse(self.url)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
                raise ValueError(
                    "image url source requires an http or https URL with a valid host"
                )
        return self


class AnthropicImageBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["image"]
    source: AnthropicImageSource
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)
    cache_control: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_tool_use(self) -> "AnthropicToolUseBlock":
        if not self.id:
            raise ValueError("tool_use.id must be a non-empty string")
        if not self.name:
            raise ValueError("tool_use.name must be a non-empty string")
        return self


class AnthropicToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[
        str,
        List[
            Annotated[
                Union[AnthropicTextBlock, AnthropicImageBlock],
                Field(discriminator="type"),
            ]
        ],
        None,
    ] = None
    is_error: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None

    @field_validator("tool_use_id")
    @classmethod
    def _validate_tool_use_id(cls, value: str) -> str:
        if not value:
            raise ValueError("tool_result.tool_use_id must be a non-empty string")
        return value


class AnthropicThinkingBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None


class AnthropicRedactedThinkingBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["redacted_thinking"]
    data: str = ""


AnthropicInputContentBlock = Annotated[
    Union[
        AnthropicTextBlock,
        AnthropicImageBlock,
        AnthropicToolUseBlock,
        AnthropicToolResultBlock,
        AnthropicThinkingBlock,
        AnthropicRedactedThinkingBlock,
    ],
    Field(discriminator="type"),
]


class AnthropicMessageParam(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicInputContentBlock]]

    @model_validator(mode="after")
    def _validate_message_blocks(self) -> "AnthropicMessageParam":
        if isinstance(self.content, list):
            if len(self.content) == 0:
                raise ValueError("message content list cannot be empty")
            for block in self.content:
                if self.role == "user" and isinstance(
                    block,
                    (
                        AnthropicToolUseBlock,
                        AnthropicThinkingBlock,
                        AnthropicRedactedThinkingBlock,
                    ),
                ):
                    raise ValueError(
                        f"block type '{block.type}' is not allowed in 'user' messages"
                    )
                if self.role == "assistant" and isinstance(
                    block, (AnthropicToolResultBlock, AnthropicImageBlock)
                ):
                    raise ValueError(
                        f"block type '{block.type}' is not allowed in 'assistant' messages"
                    )
        return self


class AnthropicTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]
    cache_control: Optional[Dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("tool.name must be a non-empty string")
        return value


class AnthropicToolChoiceAuto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["auto"]
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicToolChoiceAny(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["any"]
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicToolChoiceTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("tool_choice.name must be a non-empty string")
        return value


class AnthropicToolChoiceNone(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["none"]


AnthropicToolChoice = Annotated[
    Union[
        AnthropicToolChoiceAuto,
        AnthropicToolChoiceAny,
        AnthropicToolChoiceTool,
        AnthropicToolChoiceNone,
    ],
    Field(discriminator="type"),
]


class AnthropicThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[StrictInt] = None

    @model_validator(mode="after")
    def _validate_thinking_config(self) -> "AnthropicThinkingConfig":
        if self.type == "enabled":
            if self.budget_tokens is None:
                raise ValueError(
                    "thinking.budget_tokens is required when thinking.type is 'enabled'"
                )
            if self.budget_tokens <= 0 or self.budget_tokens > INT32_MAX:
                raise ValueError(f"thinking.budget_tokens must be in [1, {INT32_MAX}]")
        elif self.type == "disabled":
            if self.budget_tokens is not None:
                raise ValueError(
                    "thinking.budget_tokens cannot be set when thinking.type is 'disabled'"
                )
        elif self.type == "adaptive":
            if self.budget_tokens is not None and (
                self.budget_tokens <= 0 or self.budget_tokens > INT32_MAX
            ):
                raise ValueError(f"thinking.budget_tokens must be in [1, {INT32_MAX}]")
        return self


class AnthropicOutputFormat(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: Literal["json_schema", "text"]
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    name: Optional[str] = None
    strict: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_format(self) -> "AnthropicOutputFormat":
        if self.type == "json_schema" and (
            not isinstance(self.schema_, dict) or len(self.schema_) == 0
        ):
            raise ValueError(
                "output_config.format.schema must be a non-empty object for type 'json_schema'"
            )
        return self


class AnthropicOutputConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: Optional[AnthropicOutputFormat] = None
    effort: Optional[Literal["low", "medium", "high", "max"]] = None


class AnthropicMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: Optional[str] = None


class _AnthropicRequestBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: List[AnthropicMessageParam]
    system: Optional[Union[str, List[AnthropicSystemTextBlock]]] = None
    max_tokens: Optional[StrictInt] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[StrictInt] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Optional[AnthropicMetadata] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    thinking: Optional[AnthropicThinkingConfig] = None
    output_config: Optional[AnthropicOutputConfig] = None
    seed: Optional[StrictInt] = None

    @field_validator("messages")
    @classmethod
    def _validate_messages(
        cls, value: List[AnthropicMessageParam]
    ) -> List[AnthropicMessageParam]:
        if not value:
            raise ValueError("messages must contain at least one message")
        if len(value) > MAX_MESSAGES:
            raise ValueError(
                f"messages cannot contain more than {MAX_MESSAGES} entries"
            )
        if value[0].role != "user":
            raise ValueError("first message must have role 'user'")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("temperature must be a finite number in [0.0, 1.0]")
        return value

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("top_p must be a finite number in [0.0, 1.0]")
        return value

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 0 or value > INT32_MAX):
            raise ValueError(f"top_k must be in [0, {INT32_MAX}]")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < INT64_MIN or value > INT64_MAX):
            raise ValueError(f"seed must be in [{INT64_MIN}, {INT64_MAX}]")
        return value

    @model_validator(mode="after")
    def _validate_request_constraints(self) -> "_AnthropicRequestBase":
        if self.max_tokens is not None and (
            self.max_tokens <= 0 or self.max_tokens > INT32_MAX
        ):
            raise ValueError(f"max_tokens must be in [1, {INT32_MAX}]")

        if self.thinking is not None and self.thinking.budget_tokens is not None:
            if (
                self.max_tokens is not None
                and self.thinking.budget_tokens >= self.max_tokens
            ):
                raise ValueError(
                    "thinking.budget_tokens must be strictly less than max_tokens"
                )

        if self.tools is not None and len(self.tools) > MAX_TOOLS:
            raise ValueError(f"tools cannot contain more than {MAX_TOOLS} entries")

        if self.tool_choice is not None:
            if isinstance(self.tool_choice, AnthropicToolChoiceAny) and not self.tools:
                raise ValueError("tool_choice.type='any' requires non-empty tools")
            if isinstance(self.tool_choice, AnthropicToolChoiceTool):
                if not self.tools:
                    raise ValueError("tool_choice.type='tool' requires non-empty tools")
                tool_names = {tool.name for tool in self.tools}
                if self.tool_choice.name not in tool_names:
                    raise ValueError(
                        f"tool_choice tool '{self.tool_choice.name}' is not declared in tools"
                    )

        if self.stop_sequences is not None:
            if len(self.stop_sequences) > MAX_STOP_SEQUENCES:
                raise ValueError(
                    f"stop_sequences cannot contain more than {MAX_STOP_SEQUENCES} entries"
                )
            for seq in self.stop_sequences:
                if (
                    not isinstance(seq, str)
                    or len(seq) == 0
                    or len(seq) > MAX_STOP_SEQUENCE_LEN
                ):
                    raise ValueError(
                        f"stop_sequences entries must be non-empty strings of at most {MAX_STOP_SEQUENCE_LEN} chars"
                    )

        return self


class AnthropicMessagesRequest(_AnthropicRequestBase):
    max_tokens: StrictInt


class AnthropicCountTokensRequest(_AnthropicRequestBase):
    max_tokens: Optional[StrictInt] = None


class AnthropicOutputThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class AnthropicOutputRedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class AnthropicOutputTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicOutputToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


AnthropicContentBlock = Annotated[
    Union[
        AnthropicOutputThinkingBlock,
        AnthropicOutputRedactedThinkingBlock,
        AnthropicOutputTextBlock,
        AnthropicOutputToolUseBlock,
    ],
    Field(discriminator="type"),
]


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str = ""
    content: List[AnthropicContentBlock]
    stop_reason: Optional[
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

    def to_wire_dict(self) -> Dict[str, Any]:
        """Serialize preserving nullable wire fields like stop_reason and stop_sequence."""
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "model": self.model,
            "content": [block.model_dump(exclude_none=True) for block in self.content],
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": self.usage.model_dump(exclude_none=True),
        }


class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int


class AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
