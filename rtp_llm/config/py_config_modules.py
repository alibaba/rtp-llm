import logging
import os
from typing import Optional

from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.kv_cache_config import KVCacheConfig

from rtp_llm.ops import (
    ArpcConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DeviceResourceConfig,
    EPLBConfig,
    FfnDisAggregateConfig,
    FMHAConfig,
    GrpcConfig,
    HWKernelConfig,
    MiscellaneousConfig,
    ModelSpecificConfig,
    MoeConfig,
    PDSepConfig,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
    RoleType,
    RuntimeConfig,
    SamplerConfig,
    SpeculativeExecutionConfig,
    VitSeparation,
)

DEFAULT_START_PORT = 8088
MASTER_INFO_PORT_NUM = 11
MIN_WORKER_INFO_PORT_NUM = 8
WORKER_INFO_PORT_NUM = MIN_WORKER_INFO_PORT_NUM


class ServerConfig:
    def __init__(self):
        self.frontend_server_count = 4
        self.start_port = DEFAULT_START_PORT
        self.timeout_keep_alive = 5
        self.frontend_server_id = 0
        self.rank_id = 0
        self.worker_info_port_num: int = MIN_WORKER_INFO_PORT_NUM
        self.shutdown_timeout: int = 50  # Default timeout in seconds, -1 means wait indefinitely
        self.monitor_interval: int = 1   # Monitor interval in seconds

    # update_from_args 方法已不再需要
    # 配置绑定现在通过声明式 bind_to 参数在 add_argument 时自动处理

    def to_string(self):
        return (
            f"frontend_server_count: {self.frontend_server_count}\n"
            f"start_port: {self.start_port}\n"
            f"timeout_keep_alive: {self.timeout_keep_alive}\n"
            f"frontend_server_id: {self.frontend_server_id}\n"
            f"rank_id: {self.rank_id}\n"
            f"worker_info_port_num: {self.worker_info_port_num}\n"
            f"shutdown_timeout: {self.shutdown_timeout}\n"
            f"monitor_interval: {self.monitor_interval}"
        )


class PyMiscellaneousConfig:
    """Python wrapper for C++ MiscellaneousConfig with additional Python-only fields."""
    def __init__(self):
        self.misc_config = MiscellaneousConfig()
        # Additional Python-only fields
        self.oss_endpoint: str = ""
        self.openai_api_key: str = "EMPTY"
        self.dashscope_api_key: str = "EMPTY"
        self.dashscope_http_url: Optional[str] = None
        self.dashscope_websocket_url: Optional[str] = None

    def to_string(self):
        return (
            self.misc_config.to_string() + "\n"
            f"oss_endpoint: {self.oss_endpoint}\n"
            f"openai_api_key: {self.openai_api_key}\n"
            f"dashscope_api_key: {self.dashscope_api_key}\n"
            f"dashscope_http_url: {self.dashscope_http_url}\n"
            f"dashscope_websocket_url: {self.dashscope_websocket_url}"
        )

class LoraConfig:
    def __init__(self):
        self.lora_info: str = "{}"
        self.merge_lora: bool = True

    def to_string(self):
        return f"lora_info: {self.lora_info}\n" f"merge_lora: {self.merge_lora}\n"


class LoadConfig:
    def __init__(self):
        self.load_method: str = "auto"

    def to_string(self):
        return (
            f"load_method: {self.load_method}"
        )


class RenderConfig:
    def __init__(self):
        self.model_template_type: Optional[str] = None
        self.default_chat_template_key: str = "default"
        self.default_tool_use_template_key: str = "tool_use"
        self.llava_chat_template: str = ""

    def to_string(self):
        return (
            f"model_template_type: {self.model_template_type}\n"
            f"default_chat_template_key: {self.default_chat_template_key}\n"
            f"default_tool_use_template_key: {self.default_tool_use_template_key}\n"
            f"llava_chat_template: {self.llava_chat_template}"
        )

class GangConfig:
    def __init__(self):
        self.fake_gang_env: bool = False
        self.gang_annocation_path: str = "/etc/podinfo/annotations"
        self.gang_config_string: Optional[str] = None
        self.zone_name: str = ""
        self.distribute_config_file: str = ""
        self.dist_barrier_timeout: Optional[int] = None
        self.gang_sleep_time: int = 10
        self.gang_timeout_min: int = 30
        self.json_gang_parts: Optional[str] = None
        self.leader_address: Optional[str] = None

    def to_string(self):
        return (
            f"fake_gang_env: {self.fake_gang_env}\n"
            f"gang_annocation_path: {self.gang_annocation_path}\n"
            f"gang_config_string: {self.gang_config_string}\n"
            f"zone_name: {self.zone_name}\n"
            f"distribute_config_file: {self.distribute_config_file}\n"
            f"dist_barrier_timeout: {self.dist_barrier_timeout}\n"
            f"gang_sleep_time: {self.gang_sleep_time}\n"
            f"gang_timeout_min: {self.gang_timeout_min}\n"
            f"json_gang_parts: {self.json_gang_parts}\n"
            f"lead_address: {self.leader_address}\n"
        )

class VitConfig:
    def __init__(self):
        self.vit_separation: VitSeparation = VitSeparation.VIT_SEPARATION_LOCAL
        self.vit_trt: int = 0
        self.trt_cache_enabled: int = 0
        self.trt_cache_path: Optional[str] = None
        self.download_headers: str = ""
        self.mm_cache_item_num: int = 10
        self.url_cache_item_num: int = 100
        self.use_igraph_cache: bool = True
        self.igraph_search_dom: str = "com.taobao.search.igraph.common"
        self.igraph_vipserver: int = 0
        self.igraph_table_name: str = ""
        self.default_key: Optional[str] = None

    def to_string(self):
        return (
            f"vit_separation: {self.vit_separation}\n"
            f"vit_trt: {self.vit_trt}\n"
            f"trt_cache_enabled: {self.trt_cache_enabled}\n"
            f"trt_cache_path: {self.trt_cache_path}\n"
            f"download_headers: {self.download_headers}\n"
            f"mm_cache_item_num: {self.mm_cache_item_num}\n"
            f"url_cache_item_num: {self.url_cache_item_num}\n"
            f"use_igraph_cache: {self.use_igraph_cache}\n"
            f"igraph_search_dom: {self.igraph_search_dom}\n"
            f"igraph_vipserver: {self.igraph_vipserver}\n"
            f"igraph_table_name: {self.igraph_table_name}\n"
            f"igraph_default_key: {self.default_key}"
        )


class GenerateEnvConfig:
    def __init__(self):
        self.think_end_tag: str = "</think>\n\n"
        self.think_end_token_id: int = -1
        self.think_mode: int = 0
        self.force_stop_words: bool = False
        self.stop_words_list: Optional[str] = None
        self.stop_words_str: Optional[str] = None
        self.think_start_tag: str = "<think>\n"
        self.generation_config_path: Optional[str] = None

    def to_string(self):
        return (
            f"think_end_tag: {self.think_end_tag}\n"
            f"think_end_token_id: {self.think_end_token_id}\n"
            f"think_mode: {self.think_mode}\n"
            f"force_stop_words: {self.force_stop_words}\n"
            f"stop_words_list: {self.stop_words_list}\n"
            f"stop_words_str: {self.stop_words_str}\n"
            f"think_start_tag: {self.think_start_tag}\n"
            f"generation_config_path: {self.generation_config_path}"
        )

class QuantizationConfig:
    def __init__(self):
        self.int8_mode: int = 0
        self.quantization: str = ""

    def to_string(self):
        return f"int8_mode: {self.int8_mode}\n" f"quantization: {self.quantization}"

    def get_quantization(self):
        """Get quantization string with compatibility logic.
        
        Returns quantization from self.quantization, or "INT8" if int8_mode == 1
        or weight_type is INT8 (from environment variable).
        """
        if self.quantization:
            return self.quantization
        if self.int8_mode == 1:
            return "INT8"
        # Check weight_type from environment variable (compatibility logic)
        import os
        weight_type = os.environ.get("WEIGHT_TYPE", "").upper()
        if weight_type == "INT8":
            return "INT8"
        return ""


class EmbeddingConfig:
    def __init__(self):
        self.embedding_model: int = 0
        self.extra_input_in_mm_embedding = ""

    def to_string(self):
        return (
            f"embedding_model: {self.embedding_model}\n"
            f"extra_input_in_mm_embedding: {self.extra_input_in_mm_embedding}"
        )


class RoleConfig:
    def __init__(self):
        self._role_type: RoleType = RoleType.PDFUSION

    @property
    def role_type(self) -> RoleType:
        """Get role_type as RoleType enum."""
        return self._role_type

    @role_type.setter
    def role_type(self, value):
        """Set role_type, accepting either RoleType enum or string."""
        if isinstance(value, str):
            self._role_type = RoleConfig._trans_role_type(value)
        elif isinstance(value, RoleType):
            self._role_type = value
        else:
            raise TypeError(f"role_type must be RoleType enum or str, got {type(value)}")

    def to_string(self):
        return f"role_type: {self._role_type.name}"

    @staticmethod
    def _trans_role_type(role_type: str) -> RoleType:
        role_type = role_type.upper()
        if role_type == "PDFUSION":
            return RoleType.PDFUSION
        elif role_type == "PREFILL":
            return RoleType.PREFILL
        elif role_type == "DECODE":
            return RoleType.DECODE
        elif role_type == "VIT":
            return RoleType.VIT
        elif role_type == "FRONTEND":
            return RoleType.FRONTEND
        else:
            return RoleType.PDFUSION

class JITConfig:
    def __init__(self):
        self.remote_jit_dir: str = ""

    def to_string(self):
        return f"remote_jit_dir: {self.remote_jit_dir}"


class DeepEPConfig:
    """
    Configuration for DeepEP settings.
    Used to track whether user has explicitly set these values.
    If all are None, auto_configure_deepep will be called.
    Otherwise, these values will be copied to moe_config.
    """
    def __init__(self):
        self.use_deepep_moe: Optional[bool] = None
        self.use_deepep_internode: Optional[bool] = None
        self.use_deepep_low_latency: Optional[bool] = None

    def to_string(self):
        return (
            f"use_deepep_moe: {self.use_deepep_moe}\n"
            f"use_deepep_internode: {self.use_deepep_internode}\n"
            f"use_deepep_low_latency: {self.use_deepep_low_latency}"
        )


class PyEnvConfigs:
    def __init__(self):
        self.server_config: ServerConfig = ServerConfig()
        self.profiling_debug_logging_config: ProfilingDebugLoggingConfig = (
            ProfilingDebugLoggingConfig()
        )
        self.model_args: ModelArgs = ModelArgs()
        self.lora_config: LoraConfig = LoraConfig()
        self.load_config: LoadConfig = LoadConfig()
        self.render_config: RenderConfig = RenderConfig()
        self.gang_config: GangConfig = GangConfig()
        self.vit_config: VitConfig = VitConfig()
        self.generate_env_config: GenerateEnvConfig = GenerateEnvConfig()
        self.quantization_config: QuantizationConfig = QuantizationConfig()
        self.eplb_config: EPLBConfig = EPLBConfig()
        self.kv_cache_config: KVCacheConfig = KVCacheConfig()
        self.device_resource_config: DeviceResourceConfig = DeviceResourceConfig()
        self.runtime_config: RuntimeConfig = RuntimeConfig()
        # EngineConfig has been merged into RuntimeConfig and ModelConfig
        # warm_up and warm_up_with_loss are in RuntimeConfig
        # max_seq_len is in ModelConfig
        self.embedding_config: EmbeddingConfig = EmbeddingConfig()
        self.role_config: RoleConfig = RoleConfig()
        self.pd_separation_config: PDSepConfig = PDSepConfig()
        self.parallelism_config: ParallelismConfig = (
            ParallelismConfig()
        )
        self.ffn_disaggregate_config: FfnDisAggregateConfig = FfnDisAggregateConfig()
        self.model_specific_config = ModelSpecificConfig()
        self.fmha_config = FMHAConfig()
        self.misc_config = PyMiscellaneousConfig()
        self.concurrency_config = ConcurrencyConfig()
        self.moe_config = MoeConfig()
        self.jit_config = JITConfig()
        self.py_hw_kernel_config: HWKernelConfig = HWKernelConfig()
        self.sp_config = SpeculativeExecutionConfig()
        self.cache_store_config = CacheStoreConfig()
        self.arpc_config = ArpcConfig()
        self.sampler_config = SamplerConfig()
        self.grpc_config = GrpcConfig()
        self.deep_ep_config = DeepEPConfig()
