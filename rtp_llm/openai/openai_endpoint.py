import hashlib
import itertools
import json
import logging
import threading
from collections import OrderedDict
from functools import partial
from typing import Any, AsyncGenerator, List, Optional

from fastapi import Request

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import (
    GenerateConfig,
    ReturnAllProbsMode,
    ThinkingMode,
)
from rtp_llm.config.grammar_constraint import GrammarConstraint
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.config.response_format import (
    ResponseFormat,
    normalize_think_tag,
    prompt_ends_with_think_anchor,
)
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.frontend.recommendation_parser import parse_and_fill_banned_combo
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DebugInfo,
    FunctionCall,
    ModelCard,
    ModelList,
    RoleEnum,
    ToolCall,
    UsageInfo,
)
from rtp_llm.openai.renderer_factory import ChatRendererFactory
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
    StreamResponseObject,
)
from rtp_llm.ops import SpecialTokens
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)

_INT32_MAX = 2_147_483_647

# check-then-set 的去重状态在共享 renderer 上可能被并发线程同时读写；
# 这不是热路径（每个配置组合至多走一次），加锁的代价可以忽略。
_THINK_WARN_LOCK = threading.Lock()
_THINK_WARN_CACHE_SIZE = 128

# Decoding this narrow on a distribution peaked at one token has no way out of a
# repetition loop unless the caller asked for anti-repetition.
_REPETITION_RISK_MAX_TEMPERATURE = 0.1
_REPETITION_RISK_MAX_TOP_P = 0.1


def _warn_once_per_renderer(
    renderer: CustomChatRenderer, warn_key: tuple, message: str, *args: Any
) -> bool:
    """按有界去重键在 renderer 上告警一次，返回本次是否真的输出了告警。

    同一个 renderer 实例会按请求切换模板（user_template / template_key /
    tool-use 变体），只按 renderer 去重会把后续模板的告警一起抑制掉；而
    逐请求告警又会被固定携带参数的客户端按 QPS 放大成日志噪声。故键由调用方
    按“同一类可复现的配置组合”构造，缓存有界以免模板体被无限保留。
    """
    with _THINK_WARN_LOCK:
        warned_keys = getattr(renderer, "_think_warned_keys", None)
        if not isinstance(warned_keys, OrderedDict):
            warned_keys = OrderedDict()
            renderer._think_warned_keys = warned_keys
        should_warn = warn_key not in warned_keys
        if should_warn:
            warned_keys[warn_key] = None
            if len(warned_keys) > _THINK_WARN_CACHE_SIZE:
                warned_keys.popitem(last=False)
        else:
            warned_keys.move_to_end(warn_key)
    if should_warn:
        logging.warning(message, *args)
    return should_warn


def _request_value_digest(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).digest()


# Sampling knobs accept a scalar or one value per returned sequence.
def _as_values(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _enabled_without_anchor_warn_key(
    request: Optional[ChatCompletionRequest], think_start_tag: str
) -> tuple:
    """告警去重键：(tag, 模板标识)。

    同一个 renderer 实例会按请求切换模板（user_template / template_key /
    tool-use 变体），只按 renderer+tag 去重会把后续模板的告警一起抑制掉。
    """
    return (
        think_start_tag,
        _request_value_digest(getattr(request, "user_template", None)),
        _request_value_digest(getattr(request, "template_key", None)),
        bool(getattr(request, "functions", None)),
        bool(getattr(request, "tools", None)),
        _request_value_digest(getattr(request, "tool_choice", None)),
    )


def _positive_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


class OpenaiEndpoint(object):
    def __init__(
        self,
        model_config: ModelConfig,
        misc_config: PyMiscellaneousConfig,
        vit_config: VitConfig,
        tokenizer: BaseTokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
    ):
        # Get values from model_config
        self.generate_env_config = model_config.generate_env_config
        self.max_seq_len = model_config.max_seq_len
        self.model_name = model_config.model_name
        self.special_tokens = model_config.special_tokens
        template_type = model_config.template_type
        ckpt_path = model_config.ckpt_path
        render_config = model_config.render_config

        if tokenizer == None:
            raise AttributeError(f"tokenizer is none!")
        self.tokenizer: BaseTokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id == None:
            self.eos_token_id = self.special_tokens.eos_token_id

        self.stop_words_id_list = self.special_tokens.stop_words_id_list

        render_params = RendererParams(
            model_type=model_config.model_type,
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            template_type=template_type,
            ckpt_path=ckpt_path,
        )

        self.chat_renderer: CustomChatRenderer = ChatRendererFactory.get_renderer(
            self.tokenizer,
            render_params,
            self.generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        logging.info(f"Finally openai endpoint uses renderer: {self.chat_renderer} ")
        self.template_renderer: CustomChatRenderer = (
            self.chat_renderer
            if isinstance(self.chat_renderer, BasicRenderer)
            else BasicRenderer(
                self.tokenizer,
                render_params,
                self.generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )
        )
        logging.info(f"chat_renderer [{self.chat_renderer}] is created.")
        extra_stop_word_ids_list = self.chat_renderer.get_all_extra_stop_word_ids_list()
        self.stop_words_id_list.extend(extra_stop_word_ids_list)
        self.stop_words_str_list = self.special_tokens.stop_words_str_list

        env_stop_words_str = self.generate_env_config.stop_words_str
        env_stop_words_id = self.generate_env_config.stop_words_list
        env_stop_words_str_list = (
            json.loads(env_stop_words_str) if env_stop_words_str else []
        )
        env_stop_words_id_list = (
            json.loads(env_stop_words_id) if env_stop_words_id else []
        )
        env_force_stop = self.generate_env_config.force_stop_words
        if env_force_stop:
            self.stop_words_str_list = env_stop_words_str_list
            self.stop_words_id_list = env_stop_words_id_list
        else:
            self.stop_words_str_list = (
                self.stop_words_str_list + env_stop_words_str_list
            )
            self.stop_words_id_list = self.stop_words_id_list + env_stop_words_id_list

        # sync between stop word id str and stop words id list
        stop_words_str_list_from_id = []
        for stop_word_ids in self.stop_words_id_list:
            word = self.tokenizer.decode(stop_word_ids)
            if len(word):
                stop_words_str_list_from_id.append(word)

        stop_words_id_list_from_str = []
        for stop_word_str in self.stop_words_str_list:
            ids = self.tokenizer.encode(stop_word_str)
            if len(ids):
                stop_words_id_list_from_str.append(ids)

        self.stop_words_str_list += stop_words_str_list_from_id
        self.stop_words_id_list += stop_words_id_list_from_str

        # dedup stop words
        self.stop_words_str_list = list(set(self.stop_words_str_list))
        self.stop_words_id_list = self._dedup_stop_words_list(self.stop_words_id_list)

        logging.info(
            f"use stop_words_str_list [{self.stop_words_str_list}], "
            f"stop_words_id_list [{self.stop_words_id_list}]"
        )

    async def list_models(self):
        model_card = ModelCard(id=self.model_name)
        return ModelList(data=[model_card])

    def _dedup_stop_words_list(
        self, stop_words_list: List[List[int]]
    ) -> List[List[int]]:
        return [i for i, _ in itertools.groupby(sorted(stop_words_list))]

    def _request_prompt_has_think_anchor(
        self,
        config: GenerateConfig,
        input_ids: Optional[List[int]],
        request: Optional[ChatCompletionRequest] = None,
    ) -> bool:
        """Whether the rendered prompt ends with an open think start tag.

        Prefers the flag recorded while rendering; falls back to a token-level
        comparison against the prompt tail so callers that pass input_ids but no
        recorded flag still resolve correctly.

        The fallback compares every token form the anchor can take -- the
        configured ``begin_think_token_ids`` first, then both the raw tag and the
        tag stripped of trailing newlines -- because the text predicate
        ``prompt_ends_with_think_anchor`` tolerates trailing newlines while the
        tag holds a single value (Qwen templates end with ``<think>\\n`` while
        DeepSeek appends a bare ``<think>``). Comparing only one form would let
        the endpoint and the renderers disagree about the same prompt.
        """
        if request is not None:
            anchor_state = request.prompt_has_think_anchor()
            if anchor_state is not None:
                return anchor_state
        if input_ids is None:
            return False
        return any(
            input_ids[-len(begin_ids) :] == begin_ids
            for begin_ids in self._think_anchor_id_variants(config)
        )

    def _think_anchor_id_variants(self, config: GenerateConfig) -> List[List[int]]:
        """Token forms of the open think anchor, longest first."""
        think_start_tag = normalize_think_tag(self.generate_env_config.think_start_tag)
        variants: List[List[int]] = []
        if config.begin_think_token_ids:
            variants.append(list(config.begin_think_token_ids))
        for text in (think_start_tag, think_start_tag.rstrip("\n")):
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in variants:
                variants.append(ids)
        return sorted(variants, key=len, reverse=True)

    def _think_end_id_variants(self) -> List[List[int]]:
        """Token forms of the think end tag, longest first.

        Derived from the same ``THINK_END_TAG`` the no-think excludes ban, so
        "what the grammar forbids" and "what counts as an already-closed block"
        cannot drift apart. Unlike ``_think_anchor_id_variants`` this takes no
        ``config``: the end tag is a deployment constant, and ``end_think_token_ids``
        is deliberately not used -- on DISABLED requests it is empty, and a
        deployment may point it at a generic terminator token.
        """
        think_end_tag = normalize_think_tag(self.generate_env_config.think_end_tag)
        variants: List[List[int]] = []
        for text in (think_end_tag, think_end_tag.rstrip("\n")):
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in variants:
                variants.append(ids)
        return sorted(variants, key=len, reverse=True)

    def _prompt_inside_think_block(
        self, config: GenerateConfig, input_ids: Optional[List[int]]
    ) -> bool:
        """Whether the prompt's last think boundary is an opener.

        ``_request_prompt_has_think_anchor`` only looks at the tail, so a prompt
        that already opened a block and continued it (a caller prefill behind the
        template's anchor) reads as unanchored there. Masking the end tag in that
        state would leave the block unclosable and the answer unreachable, so an
        unterminated block is its own exemption. The scan walks backwards and
        stops at the last boundary, so a prompt that does end near one is cheap.
        A prompt with no think markup at all -- the main DISABLED shape -- has no
        boundary to stop at and scans the whole prompt, but each token is a single
        set-membership test against the boundary first-tokens, not a substring
        match, so the pass stays linear and allocation-free.
        """
        if input_ids is None:
            return False
        start_variants = self._think_anchor_id_variants(config)
        end_variants = self._think_end_id_variants()
        start_first = {variant[0] for variant in start_variants}
        end_first = {variant[0] for variant in end_variants}
        boundary_first = start_first | end_first

        def matches(position: int, variants: List[List[int]]) -> bool:
            return any(
                input_ids[position : position + len(variant)] == variant
                for variant in variants
            )

        for position in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[position]
            if token_id not in boundary_first:
                continue
            if token_id in end_first and matches(position, end_variants):
                return False
            if token_id in start_first and matches(position, start_variants):
                return True
        return False

    def _reasoning_format_for_prompt(
        self,
        config: GenerateConfig,
        renderer: CustomChatRenderer,
        input_ids: Optional[List[int]],
        request: Optional[ChatCompletionRequest] = None,
    ) -> Optional[ReasoningFormat]:
        if config.thinking_mode not in (
            ThinkingMode.ENABLED,
            ThinkingMode.ADAPTIVE,
        ):
            return self._disabled_no_think_format(config, renderer, input_ids, request)

        base_format = renderer.get_reasoning_format()
        think_start_tag = normalize_think_tag(self.generate_env_config.think_start_tag)
        if config.thinking_mode == ThinkingMode.ENABLED:
            anchored = self._request_prompt_has_think_anchor(config, input_ids, request)

            if anchored:
                return base_format

            # R1-style models may legitimately use fixed thinking without an
            # anchor. Warn once per bounded template identity, without retaining
            # request-controlled template bodies for the renderer lifetime.
            _warn_once_per_renderer(
                renderer,
                _enabled_without_anchor_warn_key(request, think_start_tag),
                "thinking_mode=ENABLED but the rendered prompt does not end with "
                "the think start tag %r, so the model may never emit the think end "
                "tag. Pass enable_thinking=false in chat_template_kwargs, or use a "
                "template that injects the anchor.",
                think_start_tag,
            )
            return base_format

        begin_ids = config.begin_think_token_ids or self.tokenizer.encode(
            think_start_tag, add_special_tokens=False
        )
        # ADAPTIVE keeps the token-level comparison: it decides from the first
        # generated token, so it has to agree with what the decoder sees.
        prompt_has_begin = bool(
            begin_ids
            and input_ids is not None
            and input_ids[-len(begin_ids) :] == begin_ids
        )
        if config.thinking_mode == ThinkingMode.ADAPTIVE and prompt_has_begin:
            config.thinking_mode = ThinkingMode.ENABLED
            config.in_think_mode = True
        return ReasoningFormat(
            tag_begin="" if prompt_has_begin else think_start_tag,
            tag_end=base_format.tag_end,
            suffix=base_format.suffix,
            no_think_excludes=base_format.no_think_excludes,
        )

    def _disabled_no_think_format(
        self,
        config: GenerateConfig,
        renderer: CustomChatRenderer,
        input_ids: Optional[List[int]],
        request: Optional[ChatCompletionRequest],
    ) -> Optional[ReasoningFormat]:
        """Enforce no-think on a DISABLED request whose prompt is not mid-think.

        ``thinking_mode=disabled`` is the deployer's statement that the request
        must not reason. A hybrid reasoning checkpoint can still emit a think
        block on its own -- after the template's closed empty block, or with no
        think markup in the prompt at all -- and that text spends the caller's
        whole ``max_new_tokens`` before the answer starts; the response path can
        only re-route text that already exists. Keep the boundary tags out of
        the grammar instead.

        The exemptions are the shapes where the model already is (or was) asked
        to think, and masking ``</think>`` there would leave the block unclosable
        and the reply stuck in reasoning:

        - an OPEN anchor at the end of the prompt (the template just injected
          it), and
        - a prompt inside an unterminated think block (a caller prefill behind
          that anchor).

        Two request shapes keep the previous behavior as well: a renderer that
        installs its own grammar for this request (the engine accepts one grammar
        field per request), and a caller structural_tag that already bounds an
        any_text/any_tokens region (the envelope cannot wrap it).

        This runs on the OpenAI endpoint path only; raw ``prompt`` callers and
        the C++ api_server never reach it. Correct exclusion also depends on
        ``THINK_START_TAG``/``THINK_END_TAG`` matching this deployment's template.
        """
        if not self.generate_env_config.enforce_no_think_on_disabled:
            return None
        if not renderer.emits_reasoning_stream:
            return None
        # ``request`` is Optional on this helper's signature (the tail-anchor and
        # think-block probes accept a bare prompt); the renderer hook is declared
        # for a concrete request, so only ask it when there is one. A None request
        # carries no tool_choice/response_format, so the renderer cannot be about
        # to install a grammar and the envelope is safe to apply.
        if request is not None and renderer.installs_request_grammar(request):
            return None
        if self._request_prompt_has_think_anchor(config, input_ids, request):
            return None
        if self._prompt_inside_think_block(config, input_ids):
            return None
        base_format = renderer.get_reasoning_format()
        # Exclude the bare tags, not the template's newline-suffixed forms:
        # banning "<think>\n" alone would still admit a bare "<think>".
        return ReasoningFormat(
            tag_begin=normalize_think_tag(
                self.generate_env_config.think_start_tag
            ).rstrip("\n"),
            tag_end=normalize_think_tag(self.generate_env_config.think_end_tag).rstrip(
                "\n"
            ),
            suffix=base_format.suffix,
            no_think_excludes=base_format.no_think_excludes,
            enforce_no_think=True,
        )

    def _tokenize_request_stop_words(self, stop_words: List[str]) -> List[List[int]]:
        stop_word_ids = []
        for stop_word in stop_words:
            # Byte-level tokenizers encode a word differently after a space.
            variants = [stop_word]
            if stop_word and not stop_word[0].isspace():
                variants.append(" " + stop_word)
            for variant in variants:
                try:
                    token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                except TypeError:
                    if not getattr(self, "_legacy_tokenizer_warned", False):
                        self._legacy_tokenizer_warned = True
                        logging.warning(
                            "tokenizer %s does not accept add_special_tokens; "
                            "stop words may pick up special tokens and never "
                            "match in the engine",
                            type(self.tokenizer).__name__,
                        )
                    token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    stop_word_ids.append(list(token_ids))
        return stop_word_ids

    def _warn_on_repetition_collapse_risk(
        self, config: GenerateConfig, renderer: CustomChatRenderer
    ) -> None:
        """Flag the sampling configuration that leaves a repetition loop unrecoverable.

        A single-token loop is a property of the model's distribution on its
        prompt, so no sampler setting prevents one from starting; the
        anti-repetition knobs only decide whether the model can leave it. With all
        of them neutral -- which is the default -- and greedy-ish decoding there is
        no way out at all, and ``no_repeat_ngram`` is the one that breaks the loop
        in practice. Warned once per deployment rather than per request: this is a
        configuration shape, not a property of an individual prompt.
        """
        if any(size and size > 0 for size in _as_values(config.no_repeat_ngram_size)):
            return
        if any(penalty != 1 for penalty in _as_values(config.repetition_penalty)):
            return
        if any(penalty != 0 for penalty in _as_values(config.presence_penalty)):
            return
        if any(penalty != 0 for penalty in _as_values(config.frequency_penalty)):
            return
        greedy_ish = (
            any(
                temperature <= _REPETITION_RISK_MAX_TEMPERATURE
                for temperature in _as_values(config.temperature)
            )
            or any(
                top_p <= _REPETITION_RISK_MAX_TOP_P
                for top_p in _as_values(config.top_p)
            )
            or any(top_k == 1 for top_k in _as_values(config.top_k))
        )
        if not greedy_ish:
            return
        _warn_once_per_renderer(
            renderer,
            ("repetition_collapse_risk",),
            "anti-repetition is fully neutral (repetition_penalty=1, "
            "presence_penalty=0, frequency_penalty=0, no_repeat_ngram_size unset) "
            "while decoding is greedy-ish (temperature<=%s, top_p<=%s or top_k=1): "
            "if the model's distribution peaks on a single token, nothing in this "
            "configuration can break the repetition loop. Set no_repeat_ngram_size "
            "or one of the penalties on this service.",
            _REPETITION_RISK_MAX_TEMPERATURE,
            _REPETITION_RISK_MAX_TOP_P,
        )

    def _extract_generation_config(
        self,
        request: ChatCompletionRequest,
        input_ids: Optional[List[int]] = None,
        renderer: Optional[CustomChatRenderer] = None,
    ) -> GenerateConfig:
        # TODO(wangyin): implement this
        renderer = renderer or self.chat_renderer
        config = request.extra_configs or GenerateConfig()
        if request.extra_configs is not None and (
            config.response_format is not None
            or GrammarConstraint.collect_from_config(config)
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "structured output must use top-level response_format, not extra_configs",
            )
        if request.response_format is not None:
            config.response_format = request.response_format
        elif request.json_format:
            config.response_format = ResponseFormat(type="json_object")
        if request.trace_id != None:
            config.trace_id = request.trace_id
        if request.stream == True:
            config.is_streaming = True
        if request.temperature != None:
            config.temperature = request.temperature
        if request.top_p != None:
            config.top_p = request.top_p
        if request.top_k != None:
            config.top_k = request.top_k
        if request.presence_penalty is not None:
            config.presence_penalty = request.presence_penalty
        if request.frequency_penalty is not None:
            config.frequency_penalty = request.frequency_penalty
        if request.repetition_penalty is not None:
            config.repetition_penalty = request.repetition_penalty
        self._warn_on_repetition_collapse_risk(config, renderer)
        if request.n != None:
            config.num_return_sequences = request.n
        request_stop_words_list = request.stop if request.stop != None else []
        if isinstance(request_stop_words_list, str):
            request_stop_words_list = [request_stop_words_list]
        else:
            request_stop_words_list = list(request_stop_words_list)
        request_stop_words_list.extend(config.stop_words_str)
        config.stop_words_str = list(
            set(self.stop_words_str_list + request_stop_words_list)
        )
        config.stop_words_list = self._dedup_stop_words_list(
            self.stop_words_id_list
            + self.chat_renderer.tokenize_words(self.stop_words_str_list)
            + self._tokenize_request_stop_words(request_stop_words_list)
            + config.stop_words_list
        )
        if request.chat_id != None:
            config.chat_id = request.chat_id
        if request.seed != None:
            config.random_seed = request.seed
        if request.logprobs != None:
            if not request.logprobs:
                config.return_all_probs = ReturnAllProbsMode.NONE
            # Priority: if extra_configs.return_all_probs is already set to
            # something non-NONE (typically ORIGINAL), honor that — caller has
            # explicitly opted into a specific mode. Only fall through to
            # logprobs_mode when no extra_configs override is present.
            elif config.return_all_probs == ReturnAllProbsMode.NONE:
                if request.logprobs_mode == "original":
                    config.return_all_probs = ReturnAllProbsMode.ORIGINAL
                else:
                    config.return_all_probs = ReturnAllProbsMode.DEFAULT
        if request.logprobs or request.functions:
            config.is_streaming = True
        if request.prompt_logprobs is not None:
            if request.prompt_logprobs <= 0 or request.prompt_logprobs > 1024:
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    f"prompt_logprobs must be in [1, 1024], got {request.prompt_logprobs}",
                )
            config.return_prompt_logits = True
            config.prompt_logits_top_k = request.prompt_logprobs
            config.enforce_prompt_scoring_constraints()
            request.stream = False
        if config.return_prompt_logits and request.prompt_logprobs is None:
            config.validate()
            request.stream = False
        config.convert_select_tokens(len(self.tokenizer), self.tokenizer)

        if (
            request.extra_configs
            and request.extra_configs.max_thinking_tokens is not None
            and isinstance(request.extra_configs.max_thinking_tokens, int)
        ):
            config.max_thinking_tokens = request.extra_configs.max_thinking_tokens
        if request.thinking_budget is not None:
            budget = int(request.thinking_budget)
            config.max_thinking_tokens = _INT32_MAX if budget < 0 else budget
        config.thinking_mode = renderer.resolve_thinking_mode(request)
        if (
            config.thinking_mode == ThinkingMode.ENABLED
            and renderer.default_thinking_mode != ThinkingMode.ENABLED
            and not renderer.emits_reasoning_stream
            and not self._request_prompt_has_think_anchor(config, input_ids, request)
        ):
            # A fixed-ENABLED think envelope compiles with begin="" and masks EOS
            # until the model emits the think end tag. That is only reachable for
            # models that actually produce a reasoning stream (they emit
            # </think>) or when the template opened a <think> anchor. When the
            # service did not declare thinking on and a request force-enables it
            # on a non-reasoning renderer whose prompt has no anchor (e.g. passing
            # enable_thinking=true to a Qwen2 template that ignores it), the model
            # cannot emit the end tag, so the grammar masks EOS forever and the
            # reply repeats until the length cap. Clamp back to DISABLED and let
            # the model answer normally.
            #
            # ``default_thinking_mode != ENABLED`` deliberately also covers an
            # ADAPTIVE service: adaptive compiles an ``or(think, no_think)``
            # envelope, but once a request forces ENABLED the envelope is the
            # fixed one, so ADAPTIVE carries the same hazard here. A service-level
            # ENABLED is trusted as the deployer declaring the model can think,
            # so it is never clamped.
            #
            # The warning is deduped per (renderer type, model type, declared
            # mode): the condition depends on nothing request-specific, so a
            # client that always sends enable_thinking=true would otherwise log
            # one line per request.
            _warn_once_per_renderer(
                renderer,
                (
                    "clamped_forced_enabled",
                    type(renderer).__name__,
                    getattr(renderer, "model_type", None),
                    renderer.default_thinking_mode,
                ),
                "thinking_mode=ENABLED was force-enabled by the request on a "
                "non-reasoning renderer (%s, model_type=%s) whose rendered prompt "
                "has no <think> anchor; the model cannot emit the think end tag, so "
                "clamping to DISABLED to avoid masking EOS.",
                type(renderer).__name__,
                getattr(renderer, "model_type", None),
            )
            config.thinking_mode = ThinkingMode.DISABLED
        config.in_think_mode = config.thinking_mode == ThinkingMode.ENABLED
        if config.thinking_mode == ThinkingMode.DISABLED:
            config.max_thinking_tokens = 0
        max_completion_tokens = _positive_int_or_none(request.max_completion_tokens)
        max_tokens_cap = _positive_int_or_none(request.max_tokens)
        if max_completion_tokens is not None:
            backend_max_new_tokens = max_completion_tokens
            if max_tokens_cap is not None:
                backend_max_new_tokens = min(backend_max_new_tokens, max_tokens_cap)
            config.max_new_tokens = backend_max_new_tokens
        elif request.max_tokens != None:
            config.max_new_tokens = request.max_tokens
        config.add_thinking_params(
            self.tokenizer,
            self.generate_env_config,
            enable_thinking=(
                None
                if config.thinking_mode == ThinkingMode.ADAPTIVE
                else config.thinking_mode == ThinkingMode.ENABLED
            ),
            reasoning_format=self._reasoning_format_for_prompt(
                config, renderer, input_ids, request
            ),
        )
        if request.debug_info:
            config.return_output_ids = True
        return config

    @staticmethod
    def _apply_renderer_chat_constraints(
        renderer,
        request: ChatCompletionRequest,
        config: GenerateConfig,
    ) -> None:
        apply_constraints = getattr(renderer, "apply_chat_completion_constraints", None)
        if apply_constraints is not None:
            apply_constraints(request, config)

    @staticmethod
    def _merge_tool_calls(
        existing_tool_calls: Optional[List[ToolCall]],
        delta_tool_calls: Optional[List[ToolCall]],
    ) -> Optional[List[ToolCall]]:
        """
        合并增量的 tool_calls 到现有的 tool_calls 中
        Args:
            existing_tool_calls: 现有的 tool_calls 列表
            delta_tool_calls: 增量的 tool_calls 列表
        Returns:
            合并后的 tool_calls 列表
        """
        if delta_tool_calls is None:
            return existing_tool_calls
        if existing_tool_calls is None:
            existing_tool_calls = []
        for delta_tool_call in delta_tool_calls:
            # 查找是否已存在相同 index 的 tool_call
            existing_tool_call = None
            if delta_tool_call.index is not None:
                for existing in existing_tool_calls:
                    if existing.index == delta_tool_call.index:
                        existing_tool_call = existing
                        break
            if existing_tool_call is None:
                # 创建新的 tool_call
                new_tool_call = ToolCall(
                    index=delta_tool_call.index,
                    id=delta_tool_call.id,
                    type=delta_tool_call.type,
                    function=FunctionCall(
                        name=(
                            delta_tool_call.function.name
                            if delta_tool_call.function
                            else None
                        ),
                        arguments=(
                            delta_tool_call.function.arguments
                            if delta_tool_call.function
                            else None
                        ),
                    ),
                )
                existing_tool_calls.append(new_tool_call)
            else:
                # 增量更新现有的 tool_call
                if delta_tool_call.id:
                    existing_tool_call.id = delta_tool_call.id
                if delta_tool_call.type:
                    existing_tool_call.type = delta_tool_call.type
                if delta_tool_call.function:
                    if existing_tool_call.function is None:
                        existing_tool_call.function = FunctionCall(
                            name=delta_tool_call.function.name,
                            arguments=delta_tool_call.function.arguments,
                        )
                    else:
                        if delta_tool_call.function.name:
                            existing_tool_call.function.name = (
                                delta_tool_call.function.name
                            )
                        if delta_tool_call.function.arguments:
                            if existing_tool_call.function.arguments is None:
                                existing_tool_call.function.arguments = (
                                    delta_tool_call.function.arguments
                                )
                            else:
                                existing_tool_call.function.arguments += (
                                    delta_tool_call.function.arguments
                                )
        return existing_tool_calls

    @staticmethod
    async def _collect_complete_response(
        choice_generator: Optional[AsyncGenerator[StreamResponseObject, None]],
        debug_info: Optional[DebugInfo],
        tokenizer: Optional[Any] = None,
    ) -> ChatCompletionResponse:
        all_choices = []
        usage = None
        aux_info = None
        extra_outputs = None
        async for response in choice_generator:
            if len(response.choices) != len(all_choices):
                if all_choices == []:
                    all_choices = [
                        ChatCompletionResponseChoice(
                            index=i,
                            message=ChatMessage(
                                role=choice.delta.role or RoleEnum.assistant,
                                content=choice.delta.content or None,
                                function_call=choice.delta.function_call or None,
                                tool_calls=choice.delta.tool_calls or None,
                            ),
                            finish_reason=choice.finish_reason,
                            logprobs=choice.logprobs,
                        )
                        for i, choice in enumerate(response.choices)
                    ]
                else:
                    raise ValueError(
                        f"response.choices has different length! "
                        f"[{response.choices}] vs [{all_choices}]."
                    )
            else:
                for i in range(len(all_choices)):
                    if all_choices[i].message.content == None:
                        all_choices[i].message.content = (
                            response.choices[i].delta.content or None
                        )
                    else:
                        all_choices[i].message.content += (
                            response.choices[i].delta.content or ""
                        )
                    if all_choices[i].message.reasoning_content == None:
                        all_choices[i].message.reasoning_content = (
                            response.choices[i].delta.reasoning_content or None
                        )
                    else:
                        all_choices[i].message.reasoning_content += (
                            response.choices[i].delta.reasoning_content or ""
                        )
                    all_choices[i].message.role = (
                        response.choices[i].delta.role or all_choices[i].message.role
                    )
                    all_choices[i].message.function_call = (
                        response.choices[i].delta.function_call
                        or all_choices[i].message.function_call
                    )
                    all_choices[i].message.tool_calls = (
                        OpenaiEndpoint._merge_tool_calls(
                            all_choices[i].message.tool_calls,
                            response.choices[i].delta.tool_calls,
                        )
                    )
                    all_choices[i].finish_reason = (
                        response.choices[i].finish_reason
                        or all_choices[i].finish_reason
                    )
                    if all_choices[i].logprobs != None:
                        if response.choices[i].logprobs != None:
                            all_choices[i].logprobs.content += response.choices[
                                i
                            ].logprobs.content
                    else:
                        all_choices[i].logprobs = response.choices[i].logprobs
            usage = response.usage or usage
            aux_info = response.aux_info or aux_info
            extra_outputs = response.extra_outputs or extra_outputs

        if usage == None:
            logging.warning(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)

        if (
            debug_info is not None
            and extra_outputs is not None
            and extra_outputs.output_ids is not None
        ):
            debug_info.output_ids = extra_outputs.output_ids
            if tokenizer:
                debug_info.raw_output = [
                    tokenizer.decode(output_ids)
                    for output_ids in extra_outputs.output_ids
                ]

        return ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model="",
            debug_info=debug_info,
            extra_outputs=extra_outputs,
        )

    @staticmethod
    def _complete_stream_response(
        choice_generator: AsyncGenerator[StreamResponseObject, None],
        debug_info: Optional[DebugInfo],
        tokenizer: Optional[Any] = None,
    ) -> CompleteResponseAsyncGenerator:
        # prompt_logits is attached by renderer.generate_choice on the last StreamResponseObject;
        # capture it here so collect_with_prompt_logits can attach it to the final ChatCompletionResponse.
        captured_prompt_logits = {}

        async def response_generator():
            debug_info_responded = False

            async for response in choice_generator:
                output = None
                if (
                    debug_info is not None
                    and response.extra_outputs is not None
                    and response.extra_outputs.output_ids is not None
                ):
                    output = DebugInfo()
                    output.output_ids = response.extra_outputs.output_ids
                    output.raw_output = [
                        tokenizer.decode(output_ids)
                        for output_ids in response.extra_outputs.output_ids
                    ]

                if response.prompt_logits is not None:
                    captured_prompt_logits["data"] = response.prompt_logits

                yield ChatCompletionStreamResponse(
                    choices=response.choices,
                    usage=response.usage,
                    aux_info=response.aux_info,
                    debug_info=debug_info if not debug_info_responded else output,
                    extra_outputs=response.extra_outputs,
                )
                debug_info_responded = True

        async def collect_with_prompt_logits(generator):
            resp = await OpenaiEndpoint._collect_complete_response(
                generator, debug_info=debug_info, tokenizer=tokenizer
            )
            if "data" in captured_prompt_logits:
                resp.prompt_logprobs = captured_prompt_logits["data"]
            return resp

        return CompleteResponseAsyncGenerator(
            response_generator(), collect_with_prompt_logits
        )

    def _get_debug_info(
        self,
        renderer: CustomChatRenderer,
        renderered_input: RenderedInputs,
        gen_config: GenerateConfig,
    ) -> DebugInfo:
        if renderered_input.rendered_prompt != "":
            prompt = renderered_input.rendered_prompt
        else:
            prompt = self.tokenizer.decode(renderered_input.input_ids)
        return DebugInfo(
            input_prompt=prompt,
            input_ids=renderered_input.input_ids,
            input_urls=[
                mm_input.url for mm_input in renderered_input.multimodal_inputs
            ],
            tokenizer_info=str(self.tokenizer),
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            stop_words_list=self.stop_words_str_list,
            renderer_info=renderer.get_renderer_info(),
            generate_config=gen_config,
        )

    def _align_template_thinking_switch(
        self, chat_request: ChatCompletionRequest, renderer: CustomChatRenderer
    ) -> None:
        """Make the template's think anchor agree with the resolved mode.

        DISABLED is resolved on this side only: renderers hand the request's own
        template kwargs to ``apply_chat_template``, so a template that branches on
        ``enable_thinking`` (the Qwen3.5 family) still injects an open ``<think>``
        anchor unless the caller happened to pass the flag. The prompt then asks
        for the very reasoning the deployment forbids, and the open anchor also
        makes the request exempt from the no-think envelope -- so the model spends
        the caller's whole budget thinking. Injecting ``false`` before rendering
        closes that window. Templates without an ``enable_thinking`` branch ignore
        the kwarg.

        Gated on the same deployment switch as the envelope: with the hardening
        off, removing the anchor would also remove the one signal that routes the
        model's reasoning into ``reasoning_content``, turning invisible thinking
        into visible prose.
        """
        if not self.generate_env_config.enforce_no_think_on_disabled:
            return
        if renderer.resolve_thinking_mode(chat_request) != ThinkingMode.DISABLED:
            return
        chat_request.set_chat_template_kwarg("enable_thinking", False)

    def render_chat(self, chat_request: ChatCompletionRequest):
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        self._align_template_thinking_switch(chat_request, renderer)
        prepopulate_str = ""
        if len(chat_request.messages) > 0 and chat_request.messages[-1].partial:
            prepopulate_str = str(chat_request.messages[-1].content)
            chat_request.messages.pop()
        rendered_input = renderer.render_chat(chat_request)
        if prepopulate_str != "":
            rendered_input.rendered_prompt += prepopulate_str
            rendered_input.input_ids += self.tokenizer.encode(prepopulate_str)
        # Record the anchor once, after prepopulation: a prefill appended behind
        # the anchor means the model is no longer starting from a think block.
        # The response path reads this instead of rendering the prompt again.
        chat_request.set_prompt_has_think_anchor(
            prompt_ends_with_think_anchor(
                rendered_input.rendered_prompt,
                normalize_think_tag(self.generate_env_config.think_start_tag),
            )
        )
        return rendered_input

    def chat_completion(
        self, request_id: int, chat_request: ChatCompletionRequest, raw_request: Request
    ) -> CompleteResponseAsyncGenerator:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(
            chat_request, rendered_input.input_ids, renderer
        )

        # 生成式推荐：chat 链路同样需要从 rendered_prompt 解析已曝光商品并填充
        # banned_combo_token_ids。函数内部做了开关与空值短路，对非推荐场景零侵入。
        parse_and_fill_banned_combo(
            rendered_input.rendered_prompt, generate_config, self.tokenizer
        )
        self._apply_renderer_chat_constraints(renderer, chat_request, generate_config)

        mm_inputs = rendered_input.multimodal_inputs

        if generate_config.return_prompt_logits and mm_inputs:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "prompt scoring does not support multimodal inputs",
            )

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        debug_info = (
            self._get_debug_info(renderer, rendered_input, generate_config)
            if chat_request.debug_info
            else None
        )

        # Extract QoS priority from HTTP headers and store on generate_config
        # so it survives IPC to the dash_sc enqueue loop, where
        # GenerateInput.headers may be absent.
        request_headers = extract_request_headers(raw_request.headers)
        qos_level = request_headers.get("x-dashscope-inner-qos-level")
        if qos_level is not None:
            try:
                generate_config.qos_priority = int(str(qos_level).strip())
            except (TypeError, ValueError):
                pass

        choice_generator = renderer.generate_choice(
            request_id,
            rendered_input.input_ids,
            mm_inputs,
            generate_config,
            self.backend_rpc_server_visitor,
            chat_request,
            headers=request_headers,
        )

        return self._complete_stream_response(
            choice_generator, debug_info, self.tokenizer
        )

    def _prepare_chat_input(self, request_id: int, chat_request):
        import torch

        from rtp_llm.utils.base_model_datatypes import GenerateInput

        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(
            chat_request, rendered_input.input_ids, renderer
        )
        # 与单请求入口共享同一契约：tool_choice 强制的结构化约束必须落到批量链路。
        self._apply_renderer_chat_constraints(renderer, chat_request, generate_config)

        if generate_config.return_prompt_logits and rendered_input.multimodal_inputs:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "prompt scoring does not support multimodal inputs",
            )

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        input_id_tensor = torch.Tensor(rendered_input.input_ids).int().unsqueeze(0)
        gen_input = GenerateInput(
            request_id=request_id,
            token_ids=input_id_tensor,
            mm_inputs=rendered_input.multimodal_inputs,
            generate_config=generate_config,
            tokenizer=self.tokenizer,
        )
        return gen_input, generate_config

    async def _render_single_output(self, outputs, chat_request, generate_config):
        """Render a single GenerateOutputs into a ChatCompletionResponse.
        Only non-streaming mode is supported for batch inference."""
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )

        async def _single_output_gen(out):
            yield out

        output_generator = _single_output_gen(outputs)

        prompt_logits_data = None
        if generate_config.return_prompt_logits:
            (
                output_generator,
                prompt_logits_data,
            ) = await renderer._extract_prompt_logits(output_generator)

        merged_gen = await renderer._merge_non_streaming_outputs(output_generator)
        choice_generator = renderer.render_response_stream(
            merged_gen, chat_request, generate_config
        )
        resp = await self._collect_complete_response(
            choice_generator, None, self.tokenizer
        )
        if prompt_logits_data is not None:
            resp.prompt_logprobs = prompt_logits_data
        return resp

    async def batch_chat_completion(self, base_request_id: int, batch_request) -> list:
        inputs = []
        all_configs = []
        for i, chat_request in enumerate(batch_request.requests):
            if chat_request.stream:
                raise ValueError(
                    f"batch chat completion does not support streaming (request index {i})"
                )
            chat_request.stream = False
            gen_input, generate_config = self._prepare_chat_input(
                base_request_id + i, chat_request
            )
            generate_config.is_streaming = False
            inputs.append(gen_input)
            all_configs.append(generate_config)

        batch_outputs = await self.backend_rpc_server_visitor.batch_enqueue(inputs)

        responses = []
        for i, outputs in enumerate(batch_outputs):
            complete_response = await self._render_single_output(
                outputs, batch_request.requests[i], all_configs[i]
            )
            responses.append(complete_response)

        return responses

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(
            chat_request, rendered_input.input_ids, renderer
        )
        self._apply_renderer_chat_constraints(renderer, chat_request, generate_config)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info
