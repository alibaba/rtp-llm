import copy
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.ops import RoleType
from rtp_llm.utils.check_util import *
from rtp_llm.utils.util import check_with_info


class RequestFormat:
    RAW = "raw"
    CHAT_API = "chatapi"


class ReturnAllProbsMode:
    NONE = 0
    DEFAULT = 1
    ORIGINAL = 2


class RoleAddr(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: RoleType
    ip: str
    http_port: int
    grpc_port: int

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v):
        """Convert string to RoleType enum for deserialization."""
        if isinstance(v, str):
            return getattr(RoleType, v)
        elif isinstance(v, RoleType):
            return v
        else:
            raise ValueError(
                f"RoleType must be a string or RoleType enum, got {type(v)}"
            )

    @field_serializer("role")
    def serialize_role(self, role: RoleType, _info) -> str:
        """Serialize RoleType enum to its name string for JSON serialization."""
        return role.name


# === 跨序列去重共享常量（SYNC: 必须与 RecommendationLogitsProcessor.cc 保持一致） ===
# 对应 C++ kDivergeStartComboWarnThreshold，更改时两侧同步 + test_generate_config_validators.py 会断言一致。
_DIVERGE_START_COMBO_WARN_THRESHOLD = 100
# 对应 C++ kMaxDivergeDepth，超过此值的 num_return_sequences 可能导致采样质量退化。
_MAX_DIVERGE_DEPTH = 8
# 警告限流状态：与 C++ INTERVAL_LOG(300) 对齐，每 300 秒最多输出一次。
# NOTE: 模块级全局无锁读改写，多线程下可能偶发多输出/少输出一次告警，
# 仅影响告警频率不影响正确性，可接受。
_SANITIZE_WARN_INTERVAL = 300  # seconds
_last_sanitize_warn_time: float = 0.0
_last_downgrade_warn_time: float = 0.0


def _reset_sanitize_warn_state():
    """Reset rate-limiting state for testing. NOT for production use."""
    global _last_sanitize_warn_time, _last_downgrade_warn_time
    _last_sanitize_warn_time = 0.0
    _last_downgrade_warn_time = 0.0


class GenerateConfig(BaseModel):
    # --- private attrs（不参与序列化/schema，生命周期与实例绑定） ---
    _diverge_depth_warned: bool = PrivateAttr(default=False)
    _ban_auto_downgraded: bool = PrivateAttr(default=False)

    max_new_tokens: int = 32000
    # only for qwen agent fncall check max input tokens
    max_input_tokens: int = 32000
    max_thinking_tokens: int = 32000
    in_think_mode: bool = (
        False  # same as `enable_thinking` in chat_template_kwargs, discard one in the future
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    end_think_token_ids: List[int] = []
    num_beams: int = 1
    variable_num_beams: List[int] = []
    do_sample: bool = True
    # 0 mean not use num_return_sequences,
    # whether to enable num_return_sequences, the output format of the results is inconsistent.
    num_return_sequences: int = 0
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 1.0
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    presence_penalty: Union[List[float], float] = 0.0
    frequency_penalty: Union[List[float], float] = 0.0
    min_new_tokens: Union[List[int], int] = 0
    no_repeat_ngram_size: Optional[Union[List[int], int]] = None
    # 生成式推荐：组合 token 粒度去重与曝光过滤。
    # combo_token_size 表示一个商品由多少个连续 token 组成（例如三层语义 ID = 3），0 表示关闭该功能。
    # banned_combo_token_ids 是禁止生成的商品 token 组合列表，每项长度必须等于 combo_token_size。
    # auto_parse_banned_combo 为 True 时，frontend 会自动从 prompt 中解析「已推荐曝光的商品序列」并
    # 填充到 banned_combo_token_ids；默认关闭，避免影响非推荐场景。
    combo_token_size: int = 0
    banned_combo_token_ids: List[List[int]] = []
    auto_parse_banned_combo: bool = False
    # 跨序列 combo 去重：当 num_return_sequences > 1 时，任一序列生成完整 combo 后自动广播到
    # 其他序列的 banned_combos，确保多条序列输出互不重复。默认关闭。
    # 跨序列去重开关（primary-protected 非对称模式）：开启后，主序列（序列 0）仅保留自身 ban、
    # 不受其他序列影响；补充序列接收所有序列 banned_combos 并集，保证彼此不重复。
    # 注意：enable_cross_sequence_ban 必须与 num_return_sequences(>1)、combo_token_size(>=2)
    # 在同一次 GenerateConfig 构造中给出。若条件不满足会被自动降级；
    # 后续 update()/update_and_pop() 补齐条件后会自动重新启用（仅限曾被自动降级的情况）。
    enable_cross_sequence_ban: bool = False
    # 跨序列分叉起始商品位置：前 N 个商品所有序列保持 greedy 一致，
    # 从第 N+1 个商品开始对非主序列施加 top-K 遮蔽制造分叉。默认 0（立即分叉）。
    cross_seq_diverge_start_combo: int = 0

    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int], int]] = None
    stop_words_str: List[str] = []
    stop_words_list: List[List[int]] = []
    bad_words_list: Optional[Union[List[List[List[int]]], List[List[int]]]] = None
    eos_token_id: Optional[Union[List[int], int]] = None
    pad_token_id: Optional[Union[List[int], int]] = None
    bos_token_id: Optional[Union[List[int], int]] = None
    using_hf_sampling: bool = False
    print_stop_words: bool = False
    timeout_ms: Optional[int] = -1
    ttft_timeout_ms: Optional[int] = -1
    traffic_reject_priority: Optional[int] = 100
    chat_id: Optional[str] = None
    task_id: Optional[Union[str, int]] = None
    request_format: str = RequestFormat.RAW
    # calculate_loss style: 0 for not calculate; 1 for sum; 2 for each token
    calculate_loss: int = 0
    return_logits: bool = False
    logits_index: Optional[int] = None
    # prompt scoring: return logits for all positions in the input sequence
    return_prompt_logits: bool = False
    prompt_logits_top_k: int = 64
    prompt_logits_start: int = -1
    prompt_logits_end: int = -1
    return_target_logprob: bool = True
    return_incremental: bool = False
    return_hidden_states: bool = False
    return_all_hidden_states: bool = False
    hidden_states_cut_dim: int = 0
    normalized_hidden_states: bool = False
    select_tokens_str: List[str] = []
    select_tokens_id: List[int] = []
    return_input_ids: bool = False
    return_output_ids: bool = False
    md5_value: str = ""
    custom_prop: str = "{}"
    sp_advice_prompt: str = ""
    sp_advice_prompt_token_ids: List[int] = []
    sp_edit: bool = False
    force_disable_sp_run: bool = False
    force_sp_accept: bool = False
    return_cum_log_probs: bool = False
    return_all_probs: int = ReturnAllProbsMode.NONE
    return_softmax_probs: bool = False
    aux_info: bool = True
    can_use_pd_separation: bool = True
    gen_timeline: bool = False
    profile_step: int = 3
    profile_trace_name: str = ""
    out_prefix: str = ""
    # for load balance
    role_addrs: List[RoleAddr] = []
    trace_id: str = ""

    ignore_eos: bool = False
    skip_special_tokens: bool = False
    # lora
    adapter_name: Optional[Union[str, List[str]]] = None
    is_streaming: bool = False

    # multimodal preprocess
    resized_shape: Optional[List[int]] = None
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    fps: Optional[int] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    crop_positions: Optional[List[float]] = None
    mm_timeout_ms: Optional[int] = None

    # whether add vision id in chat template; only use in frontend
    add_vision_id: bool = True

    # 是否允许tool_call专用的标签如<tool_call>作为content传出, 优化tool_call失败时的用户体验
    tool_call_message_extract_strategy: str = "default"  # default/skip_on_failure

    global_request_id: int = -1

    # 只有开启环境变量 REUSE_CACHE 时才生效
    reuse_cache: bool = True

    enable_device_cache: bool = True

    enable_memory_cache: bool = True

    enable_remote_cache: bool = True
    # 是否强制相同 request_id 的 stream 在一批中调度
    force_batch: bool = False
    batch_group_timeout: Optional[int] = None  # ms

    unique_key: str = ""

    # --- validators（统一放在所有 field 声明之后） ---

    @staticmethod
    def _sanitize_diverge_start_combo(v: "int | str | None") -> int:
        """将 cross_seq_diverge_start_combo 规范化为非负 int32 范围内的整数。

        供 field_validator（构造路径）和 update()/update_and_pop()（请求路径）共同复用，
        确保任何入口的值都经过相同的 clamp/类型兜底。

        上下界钳制：
          - 下界：负值 clamp 到 0
          - 上界：超过 INT32_MAX 的值 clamp 到 INT32_MAX，避免 trans_input protobuf 序列化时
            触发 ValueError（C++ 侧 int32_t 对应）。超大值的实际语义等价于“永不分叉”。

        NOTE on rate-limiting: 与 C++ INTERVAL_LOG(300) 对齐，非法值告警每 300 秒最多输出一次，
        避免畸形客户端高 QPS 下无界日志刷屏。
        """
        global _last_sanitize_warn_time
        _INT32_MAX = 2**31 - 1
        if v is None:
            return 0
        try:
            val = int(v)
        except (TypeError, ValueError) as e:
            now = time.monotonic()
            if now - _last_sanitize_warn_time >= _SANITIZE_WARN_INTERVAL:
                logging.getLogger(__name__).warning(
                    "cross_seq_diverge_start_combo received non-integer value %r, defaulting to 0: %s",
                    v,
                    e,
                )
                _last_sanitize_warn_time = now
            return 0
        if val < 0:
            now = time.monotonic()
            if now - _last_sanitize_warn_time >= _SANITIZE_WARN_INTERVAL:
                logging.getLogger(__name__).warning(
                    "cross_seq_diverge_start_combo is negative (%d), clamped to 0", val
                )
                _last_sanitize_warn_time = now
            return 0
        if val > _INT32_MAX:
            now = time.monotonic()
            if now - _last_sanitize_warn_time >= _SANITIZE_WARN_INTERVAL:
                logging.getLogger(__name__).warning(
                    "cross_seq_diverge_start_combo exceeds int32 max (%d), clamped to %d",
                    val,
                    _INT32_MAX,
                )
                _last_sanitize_warn_time = now
            return _INT32_MAX
        # "过大" 告警已移至 _check_cross_seq_ban_compatibility，仅在特性启用时触发，
        # 与 C++ 侧 (enable_cross_seq_ban && diverge_start_combo > threshold) 行为一致。
        return val

    @field_validator("cross_seq_diverge_start_combo", mode="before")
    @classmethod
    def _clamp_diverge_start_combo(cls, v):
        """构造路径入口：委托给 _sanitize_diverge_start_combo。"""
        return cls._sanitize_diverge_start_combo(v)

    @model_validator(mode="after")
    def _check_cross_seq_ban_compatibility(self):
        """cross_sequence_ban 与多项配置不兼容时直接禁用，一次性报告所有不兼容原因。

        NOTE on rate-limiting:
          - 不兼容降级 WARNING：每次调用无条件触发（保证即时性）。
          - 采样质量软告警（超 depth）：通过 _diverge_depth_warned (PrivateAttr) 标志去重，
            同一 GenerateConfig 实例生命周期内最多输出一次。若 update() 修改了
            num_return_sequences 使其重新超 depth，标志会被重置以允许再次告警。
          - C++ 侧使用 INTERVAL_LOG(300) 是因为 fromGenerateInput 在高 QPS 下可能
            对同一配置反复调用，属不同场景。
        """
        if not self.enable_cross_sequence_ban:
            # “先建后补”场景补救：若特性曾被自动降级且当前条件已全部满足，重新启用并继续校验。
            # 解决 request_extractor 两次 update_and_pop 分步合并导致的误降级问题。
            if (
                self._ban_auto_downgraded
                and not self.has_num_beams()
                and self.combo_token_size >= 2
                and self.num_return_sequences > 1
            ):
                self.enable_cross_sequence_ban = True
                self._ban_auto_downgraded = False
                logging.getLogger(__name__).info(
                    "enable_cross_sequence_ban re-enabled: conditions now satisfied after incremental update"
                )
                # 不 return，继续执行下方 depth/过大 告警校验
            else:
                return self
        # SYNC: 以下判定条件必须与 C++ RecommendationLogitsProcessor.cc::fromGenerateInput
        # 中 enable_cross_seq_ban 的计算逻辑保持一致（取反关系）。
        # 同步保障机制：
        #   - 常量：static_assert 钉住 kMaxDivergeDepth / kDivergeStartComboWarnThreshold
        #   - 启用条件逻辑：人工维护双份真值表测试，无运行期交叉校验
        #   - 生产映射：C++ 测试中断言 batchSize(0)==max(num,1) 等价性
        # ━━ 新增/修改启用条件时的 CHECKLIST ━━
        #   1. 同步修改另一侧的判定逻辑
        #   2. 更新双侧真值表测试：
        #      Python: TestCrossLanguageConstantSync::test_enable_conditions_sync
        #      C++:    RecommendationLogitsProcessorTest::testEnableConditionsTruthTable
        #   3. 确认新条件在双侧真值表中均有覆盖（正反例）
        #   未来演进：若条件进一步增多，应将真值表落为共享 JSON 数据文件（单一真源）。
        reasons: list = []
        if self.has_num_beams():
            reasons.append(
                f"incompatible with beam search (max_num_beams={self.max_num_beams()})"
            )
        if self.combo_token_size < 2:
            reasons.append(f"combo_token_size must be >=2, got {self.combo_token_size}")
        if self.num_return_sequences <= 1:
            reasons.append(
                f"num_return_sequences must be >1, got {self.num_return_sequences}"
            )
        if reasons:
            global _last_downgrade_warn_time
            now = time.monotonic()
            if now - _last_downgrade_warn_time >= _SANITIZE_WARN_INTERVAL:
                logging.getLogger(__name__).warning(
                    "enable_cross_sequence_ban disabled: %s", "; ".join(reasons)
                )
                _last_downgrade_warn_time = now
            self.enable_cross_sequence_ban = False
            self._ban_auto_downgraded = True
        elif self.num_return_sequences - 1 > _MAX_DIVERGE_DEPTH:
            # 软告警去重：同一实例生命周期内最多输出一次，避免 update() 多次调用时形成日志噪声。
            if not self._diverge_depth_warned:
                logging.getLogger(__name__).warning(
                    "num_return_sequences=%d exceeds recommended max diverge depth %d, "
                    "sampling quality may degrade for higher-indexed sequences",
                    self.num_return_sequences,
                    _MAX_DIVERGE_DEPTH,
                )
                self._diverge_depth_warned = True
        # 「过大」告警：仅在特性最终仍然启用时触发（reasons 为空），与 C++ 侧行为一致
        # （SYNC: C++ else if (enable_cross_seq_ban && diverge_start_combo > kDivergeStartComboWarnThreshold)）
        if (
            not reasons
            and self.cross_seq_diverge_start_combo > _DIVERGE_START_COMBO_WARN_THRESHOLD
        ):
            logging.getLogger(__name__).warning(
                "cross_seq_diverge_start_combo=%d is very large, top-K diverge masking may never activate",
                self.cross_seq_diverge_start_combo,
            )
        return self

    @field_validator("return_all_probs", mode="before")
    @classmethod
    def _coerce_return_all_probs(cls, v):
        """Legacy bool → ReturnAllProbsMode int. True → DEFAULT, False → NONE.

        return_all_probs was a `bool` field before the ORIGINAL/DEFAULT/NONE three-state
        refactor. Old callers may still pass `True/False`; pydantic strict mode would
        otherwise reject the bool→int implicit conversion.
        """
        if isinstance(v, bool):
            return ReturnAllProbsMode.DEFAULT if v else ReturnAllProbsMode.NONE
        return v

    def gen_hash_value(self):
        cp = copy.copy(self)
        cp.max_new_tokens = 0
        cp.chat_id = None
        cp.random_seed = None
        cp.md5_value = ""
        cp.timeout_ms = -1
        self.md5_value = hashlib.md5(cp.__str__().encode()).hexdigest()

    def max_num_beams(self):
        return (
            self.num_beams
            if len(self.variable_num_beams) == 0
            else max(self.variable_num_beams)
        )

    def has_num_beams(self):
        return self.max_num_beams() > 1

    def is_same(self, config: "GenerateConfig") -> bool:
        return self.md5_value == config.md5_value

    def update(self, new: Dict[str, Any]):
        """批量更新字段。

        降级/重启用语义：
          - 当条件不满足时，enable_cross_sequence_ban 会被自动降级为 False。
          - 若后续 update 补齐了条件，且开关是因自动降级而关闭的（而非用户从未开启），
            则会自动重新启用。这确保 request_extractor 两阶段合并不会导致误降级。
          - 若用户在同一次 update 中显式传入 enable_cross_sequence_ban，视为用户重新表态，
            自动重启用启发式不会覆盖其显式意图。
        """
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # setattr 不会触发 field_validator / model_validator，手动补偿：
        # 1) cross_seq_diverge_start_combo 的 clamp/类型兜底
        if "cross_seq_diverge_start_combo" in new:
            self.cross_seq_diverge_start_combo = self._sanitize_diverge_start_combo(
                self.cross_seq_diverge_start_combo
            )
        # 2) 若 num_return_sequences 变化，重置深度告警标志以允许重新检测
        if "num_return_sequences" in new:
            self._diverge_depth_warned = False
        # 3) 用户显式传入 enable_cross_sequence_ban 时，视为重新表态，清除自动降级标志
        if "enable_cross_sequence_ban" in new:
            self._ban_auto_downgraded = False
        # 4) 跨序列去重兼容性
        self._check_cross_seq_ban_compatibility()

    def update_and_pop(self, new: Dict[str, Any]):
        """批量更新字段并返回未被消费的 key。校验策略同 update()。"""
        to_remove: List[str] = []
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        # setattr 不会触发 field_validator / model_validator，手动补偿：
        if "cross_seq_diverge_start_combo" in new:
            self.cross_seq_diverge_start_combo = self._sanitize_diverge_start_combo(
                self.cross_seq_diverge_start_combo
            )
        if "num_return_sequences" in new:
            self._diverge_depth_warned = False
        if "enable_cross_sequence_ban" in new:
            self._ban_auto_downgraded = False
        self._check_cross_seq_ban_compatibility()
        return {k: v for k, v in new.items() if k not in to_remove}

    @staticmethod
    def create_generate_config(
        generate_config: Dict[str, Any], **kwargs: Any
    ) -> "GenerateConfig":
        generate_config.update(kwargs)
        try:
            config = GenerateConfig(**generate_config)
        except Exception as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_GENERATE_CONFIG_FORMAT,
                f"generate_config validate failed: {str(e)}",
            )
        config.validate()
        return config

    def convert_select_tokens(self, vocab_size, tokenizer):
        # 去重合并 select_tokens_str 派生的 id(而非 +=):保留与 select_tokens_id
        # 的并集语义(str 不会被丢弃),同时保证幂等——batch 共享同一 config 对象
        # 重复调用时不累积。
        for token_str in self.select_tokens_str:
            for token_id in tokenizer.encode(token_str):
                if token_id not in self.select_tokens_id:
                    self.select_tokens_id.append(token_id)
        if not all(
            token_id < vocab_size and token_id >= 0
            for token_id in self.select_tokens_id
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"token_id in select_tokens_id {self.select_tokens_id} should be less than vocab_size {vocab_size}, and shoud not be negative",
            )

    def add_special_tokens(self, special_tokens: Any):
        # 去重 append(而非 +=)保证幂等:batch 共享同一 config 对象时不重复追加,
        # 同时保留用户已传入的 stop words。假设传入的 stop_word_* 不含 batch 维度。
        for ids in special_tokens.stop_words_id_list:
            if ids not in self.stop_words_list:
                self.stop_words_list.append(ids)
        for word in special_tokens.stop_words_str_list:
            if word not in self.stop_words_str:
                self.stop_words_str.append(word)

    def add_thinking_params(self, tokenizer, generate_env_config):
        """Add thinking parameters from generate_env_config.

        Args:
            tokenizer: Tokenizer instance.
            generate_env_config: GenerateEnvConfig object.
        """

        end_think_token_id = generate_env_config.think_end_token_id
        self.end_think_token_ids = (
            [end_think_token_id] if end_think_token_id != -1 else []
        )
        if (
            bool(generate_env_config.think_mode)
            and tokenizer
            and end_think_token_id == -1
        ):
            think_end_tag: str = generate_env_config.think_end_tag.encode(
                "utf-8"
            ).decode("unicode_escape")
            tokenized_result: List[int] = tokenizer.encode(
                think_end_tag, add_special_tokens=False
            )
            self.end_think_token_ids = tokenized_result
        self.in_think_mode = (
            bool(generate_env_config.think_mode) and len(self.end_think_token_ids) >= 0
        )

    def add_stop_ids_from_str(self, tokenizer):
        ids_list = []
        for word in self.stop_words_str:
            if tokenizer is None:
                return
            else:
                token_id = tokenizer.convert_tokens_to_ids(word)
                if isinstance(token_id, int):
                    ids_list.append([token_id])
                elif isinstance(token_id, list):
                    ids_list.append(token_id)
                else:
                    ids_list.append(tokenizer.encode(word, add_special_tokens=True))

        # remove duplicate element
        for item in ids_list:
            if item not in self.stop_words_list:
                self.stop_words_list.append(item)

    def validate(self):
        try:
            check_with_info(
                is_union_positive_integer(self.top_k),
                f"top_k {self.top_k} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.top_p),
                f"top_p {self.top_p} is wrong data type",
            )
            check_with_info(
                is_union_positive_integer(self.min_new_tokens),
                f"min_new_tokens {self.min_new_tokens} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.repetition_penalty),
                f"repetition_penalty {self.repetition_penalty} is wrong data type",
            )
            check_with_info(
                is_union_number(self.presence_penalty),
                f"presence_penalty {self.presence_penalty} is wrong data type",
            )
            check_with_info(
                is_union_number(self.frequency_penalty),
                f"frequency_penalty {self.frequency_penalty} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.max_new_tokens),
                f"max_new_tokens {self.max_new_tokens} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.num_beams),
                f"num_beams {self.num_beams} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer(self.variable_num_beams),
                f"variable_num_beams {self.variable_num_beams} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.num_return_sequences),
                f"num_return_sequences {self.num_return_sequences} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.temperature),
                f"temperature {self.temperature} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.no_repeat_ngram_size),
                f"no_repeat_ngram_size {self.no_repeat_ngram_size} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.random_seed),
                f"random_seed {self.random_seed} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_number, self.top_p_decay),
                f"top_p_decay {self.top_p_decay} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_number, self.top_p_min),
                f"top_p_min {self.top_p_min} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.top_p_reset_ids),
                f"top_p_reset_ids {self.top_p_reset_ids} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.eos_token_id),
                f"eos_token_id {self.eos_token_id} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.pad_token_id),
                f"pad_token_id {self.pad_token_id} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.bos_token_id),
                f"bos_token_id {self.bos_token_id} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer_list(self.stop_words_list),
                f"stop_words_list {self.stop_words_list} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.combo_token_size),
                f"combo_token_size {self.combo_token_size} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer_list(self.banned_combo_token_ids),
                f"banned_combo_token_ids {self.banned_combo_token_ids} is wrong data type",
            )
            if self.combo_token_size > 0:
                check_with_info(
                    all(
                        len(combo) == self.combo_token_size
                        for combo in self.banned_combo_token_ids
                    ),
                    f"every item in banned_combo_token_ids must have length == combo_token_size {self.combo_token_size}",
                )
            check_with_info(
                is_union_positive_integer(self.sp_advice_prompt_token_ids),
                f"sp_advice_prompt_token_ids {self.sp_advice_prompt_token_ids} is wrong data type",
            )
            if self.in_think_mode:
                check_with_info(
                    is_positive_integer(self.max_thinking_tokens),
                    f"max_thinking_tokens {self.max_thinking_tokens} is wrong data type",
                )
                check_with_info(
                    is_list_positive_integer(self.end_think_token_ids),
                    f"end_think_token_ids {self.end_think_token_ids} is wrong data type",
                )
            calculate_loss_list = [0, 1, 2]
            check_with_info(
                self.calculate_loss in calculate_loss_list,
                f"calculate_loss {self.top_k} in generate_config can only be in {calculate_loss_list},"
                " but it's {self.calculate_loss}",
            )
            if self.return_prompt_logits:
                self.enforce_prompt_scoring_constraints()
                check_with_info(
                    0 < self.prompt_logits_top_k <= 1024,
                    f"prompt_logits_top_k must be in [1, 1024], got {self.prompt_logits_top_k}",
                )
                check_with_info(
                    self.num_return_sequences == 0 and self.num_beams <= 1,
                    "prompt scoring does not support num_return_sequences > 0 or beam search",
                )
                if self.prompt_logits_start >= 0 and self.prompt_logits_end >= 0:
                    check_with_info(
                        self.prompt_logits_start <= self.prompt_logits_end,
                        f"prompt_logits_start ({self.prompt_logits_start}) must <= prompt_logits_end ({self.prompt_logits_end})",
                    )
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))

    def enforce_prompt_scoring_constraints(self):
        """Clamp config fields for prompt scoring mode. Call after setting return_prompt_logits=True."""
        self.max_new_tokens = 1
        self.is_streaming = False
        self.reuse_cache = False
        self.can_use_pd_separation = False
