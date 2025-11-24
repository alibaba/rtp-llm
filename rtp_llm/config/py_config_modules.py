import logging
import os
from typing import Optional

from rtp_llm.ops import (
    ConcurrencyConfig,
    FfnDisAggregateConfig,
    FMHAConfig,
    MiscellaneousConfig,
    ModelSpecificConfig,
    ParallelismDistributedConfig,
    ProfilingDebugLoggingConfig,
    RoleType,
)
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.weight_type import WEIGHT_TYPE

DEFAULT_START_PORT = 8088
MASTER_INFO_PORT_NUM = 11
MIN_WORKER_INFO_PORT_NUM = 7
WORKER_INFO_PORT_NUM = MIN_WORKER_INFO_PORT_NUM


def get_env_int(name: str, default: int = -1):
    v = os.environ.get(name, None)
    return int(v) if v is not None and v != "" else default


def get_env_str(name: str, default: str = ""):
    v = os.environ.get(name, None)
    return v if v is not None else default


def get_env_bool(name: str, default: bool = False):
    # in fact, we can always get value from env, if that's not specified, we return default value
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    return v.lower() == "1" or v.lower() == "on" or v.lower() == "true"


class ServerConfig:
    def __init__(self):
        self.frontend_server_count = 4
        self.start_port = DEFAULT_START_PORT
        self.timeout_keep_alive = 5
        self.frontend_server_id = 0
        self.rank_id = 0

    def update_from_env(self):
        self.frontend_server_count = int(
            os.environ.get("FRONTEND_SERVER_COUNT", self.frontend_server_count)
        )
        self.start_port = int(os.environ.get("START_PORT", self.start_port))
        self.timeout_keep_alive = int(
            os.environ.get("TIMEOUT_KEEP_ALIVE", self.timeout_keep_alive)
        )
        self.frontend_server_id = int(
            os.environ.get("FRONTEND_SERVER_ID", self.frontend_server_id)
        )

    def to_string(self):
        return (
            f"frontend_server_count: {self.frontend_server_count}\n"
            f"start_port: {self.start_port}\n"
            f"timeout_keep_alive: {self.timeout_keep_alive}\n"
            f"frontend_server_id: {self.frontend_server_id}\n"
            f"rank_id: {self.rank_id}"
        )


class ModelConfig:
    def __init__(self):
        self.extra_data_path: str = ""
        self.local_extra_data_path: Optional[str] = None
        self.tokenizer_path: str = ""
        self.act_type: str = "FP16"
        self.use_float32: bool = False
        self.original_checkpoint_path: Optional[str] = None
        self.mla_ops_type: str = "AUTO"
        self.ft_plugin_path: Optional[str] = None
        self.weight_type: Optional[str] = None

        self.task_type: Optional[str] = None
        self.model_type: str = ""
        self.checkpoint_path: str = ""
        self.oss_endpoint: str = ""
        self.ptuning_path: Optional[str] = None
        self.openai_api_key = "EMPTY"

        self.openai_api_key: str = "EMPTY"
        self.dashscope_api_key: str = "EMPTY"
        self.dashscope_http_url: Optional[str] = None
        self.dashscope_websocket_url: Optional[str] = None
        self.json_model_override_args: str = "{}"

    def update_from_env(self):
        self.extra_data_path = os.environ.get("EXTRA_DATA_PATH", self.extra_data_path)
        self.local_extra_data_path = os.environ.get(
            "LOCAL_EXTRA_DATA_PATH", self.local_extra_data_path
        )
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", self.tokenizer_path)
        if int(os.environ.get("USE_FLOAT32", 0) == "1"):
            self.act_type = WEIGHT_TYPE.FP32.to_str()
            logging.info(f"set data_type = WEIGHT_TYPE.FP32 by USE_FLOAT32 == 1")
        else:
            self.act_type = os.environ.get("ACT_TYPE", self.act_type)
        self.use_float32 = get_env_bool("USE_FLOAT32", self.use_float32)
        self.original_checkpoint_path = os.environ.get(
            "ORIGINAL_CHECKPOINT_PATH", self.original_checkpoint_path
        )
        self.mla_ops_type = os.environ.get("MLA_OPS_TYPE", self.mla_ops_type)
        self.ft_plugin_path = os.environ.get("FT_PLUGIN_PATH", self.ft_plugin_path)
        self.weight_type = os.environ.get("WEIGHT_TYPE", self.weight_type)
        self.task_type = os.environ.get("TASK_TYPE", self.task_type)
        self.model_type = os.environ.get("MODEL_TYPE", self.model_type)
        self.checkpoint_path = os.environ.get("CHECKPOINT_PATH", self.checkpoint_path)
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", self.checkpoint_path)
        self.oss_endpoint = os.environ.get("OSS_ENDPOINT", self.oss_endpoint)
        self.ptuning_path = os.environ.get("PTUNING_PATH", self.ptuning_path)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)
        self.dashscope_api_key = os.environ.get(
            "DASHSCOPE_API_KEY", self.dashscope_api_key
        )
        self.dashscope_http_url = os.environ.get(
            "DASHSCOPE_HTTP_URL", self.dashscope_http_url
        )
        self.dashscope_websocket_url = os.environ.get(
            "DASHSCOPE_WEBSOCKET_URL", self.dashscope_websocket_url
        )
        self.json_model_override_args = os.environ.get(
            "JSON_MODEL_OVERRIDE_ARGS", self.json_model_override_args
        )

    def to_string(self):
        return (
            f"extra_data_path: {self.extra_data_path}\n"
            f"local_extra_data_path: {self.local_extra_data_path}\n"
            f"tokenizer_path: {self.tokenizer_path}\n"
            f"act_type: {self.act_type}\n"
            f"use_float32: {self.use_float32}\n"
            f"original_checkpoint_path: {self.original_checkpoint_path}\n"
            f"mla_ops_type: {self.mla_ops_type}\n"
            f"ft_plugin_path: {self.ft_plugin_path}\n"
            f"weight_type: {self.weight_type}\n"
            f"task_type: {self.task_type}\n"
            f"model_type: {self.model_type}\n"
            f"checkpoint_path: {self.checkpoint_path}\n"
            f"oss_endpoint: {self.oss_endpoint}\n"
            f"ptuning_path: {self.ptuning_path}\n"
            f"openai_api_key: {self.openai_api_key}\n"
            f"dashscope_api_key: {self.dashscope_api_key}\n"
            f"dashscope_http_url: {self.dashscope_http_url}\n"
            f"dashscope_websocket_url: {self.dashscope_websocket_url}\n"
            f"json_model_override_args: {self.json_model_override_args}"
        )


# Todo: 合并到c++的SpeculativeExecutionConfig
class PySpeculativeExecutionConfig:
    def __init__(self):
        self.gen_num_per_circle: int = 5
        self.sp_quantization: Optional[str] = None
        self.sp_checkpoint_path: Optional[str] = None
        self.sp_type: Optional[str] = None
        self.sp_model_type: Optional[str] = None
        self.sp_kv_cache_dtype: Optional[str] = None

    def update_from_env(self):
        self.gen_num_per_circle = int(
            os.environ.get("GEN_NUM_PER_CIRCLE", self.gen_num_per_circle)
        )
        sp_int8_mode = int(os.environ.get("SP_INT8_MODE", 0))
        self.sp_quantization = os.environ.get("SP_QUANTIZATION", self.sp_quantization)
        if sp_int8_mode and not self.sp_quantization:
            self.sp_quantization = WEIGHT_TYPE.INT8.to_str()

        self.sp_kv_cache_dtype = os.environ.get("SP_KV_CACHE_DTYPE", None)
        self.sp_checkpoint_path = os.environ.get(
            "SP_CHECKPOINT_PATH", self.sp_checkpoint_path
        )
        self.sp_type = os.environ.get("SP_TYPE", self.sp_type)
        self.sp_model_type = os.environ.get("SP_MODEL_TYPE", self.sp_model_type)

    def to_string(self):
        return (
            f"gen_num_per_circle: {self.gen_num_per_circle}\n"
            f"sp_quantization: {self.sp_quantization}\n"
            f"sp_checkpoint_path: {self.sp_checkpoint_path}\n"
            f"sp_type: {self.sp_type}\n"
            f"sp_model_type: {self.sp_model_type}"
            f"sp_kv_cache_dtype: {self.sp_kv_cache_dtype}\n"
            f"sp_checkpoint_path: {self.sp_checkpoint_path}"
        )


class LoraConfig:
    def __init__(self):
        self.lora_info: str = "{}"
        self.merge_lora: bool = True

    def update_from_env(self):
        self.lora_info = os.environ.get("LORA_INFO", self.lora_info)
        self.merge_lora = get_env_bool("MERGE_LORA", self.merge_lora)

    def to_string(self):
        return f"lora_info: {self.lora_info}\n" f"merge_lora: {self.merge_lora}\n"


class LoadConfig:
    def __init__(self):
        self.phy2log_path: str = ""
        self.converter_num_per_gpu: int = 4
        self.tokenizers_parallelism: bool = False
        # seem like it's a third-party pkg environment, but we reserve it temporar
        self.load_ckpt_num_process: int = 0
        self.load_method: str = "auto"
        self.use_fast_tokenizer: bool = True

    def update_from_env(self):
        self.phy2log_path = os.environ.get("PHY2LOG_PATH", self.phy2log_path)
        self.converter_num_per_gpu = int(
            os.environ.get("CONVERTER_NUM_PER_GPU", self.converter_num_per_gpu)
        )
        self.tokenizers_parallelism = get_env_bool(
            "TOKENIZERS_PARALLELISM", self.tokenizers_parallelism
        )
        self.load_ckpt_num_process = int(
            os.environ.get("LOAD_CKPT_NUM_PROCESS", self.load_ckpt_num_process)
        )
        self.load_method = str(os.environ.get("LOAD_METHOD", self.load_method)).lower()
        self.use_fast_tokenizer = get_env_bool(
            "USE_FAST_TOKENIZER", self.use_fast_tokenizer
        )

    def to_string(self):
        return (
            f"phy2log_path: {self.phy2log_path}\n"
            f"converter_num_per_gpu: {self.converter_num_per_gpu}\n"
            f"tokenizers_parallelism: {self.tokenizers_parallelism}\n"
            f"load_ckpt_num_process: {self.load_ckpt_num_process}\n"
            f"load_method: {self.load_method}"
        )


class RenderConfig:
    def __init__(self):
        self.model_template_type: Optional[str] = None
        self.default_chat_template_key: str = "default"
        self.default_tool_use_template_key: str = "tool_use"
        self.llava_chat_template: str = ""

    def update_from_env(self):
        self.model_template_type = os.environ.get(
            "MODEL_TEMPLATE_TYPE", self.model_template_type
        )
        self.default_chat_template_key = os.environ.get(
            "DEFAULT_CHAT_TEMPLATE_KEY", self.default_chat_template_key
        )
        self.default_tool_use_template_key = os.environ.get(
            "DEFAULT_TOOL_USE_TEMPLATE_KEY", self.default_tool_use_template_key
        )
        self.llava_chat_template = os.environ.get(
            "LLAVA_CHAT_TEMPLATE", self.llava_chat_template
        )

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

    def update_from_env(self):
        self.fake_gang_env = get_env_bool("FAKE_GANG_ENV", self.fake_gang_env)
        self.gang_annocation_path = os.environ.get(
            "GANG_ANNOCATION_PATH", self.gang_annocation_path
        )
        self.gang_config_string = os.environ.get(
            "GANG_CONFIG_STRING", self.gang_config_string
        )
        self.zone_name = os.environ.get("ZONE_NAME", self.zone_name)
        self.distribute_config_file = os.environ.get(
            "DISTRIBUTE_CONFIG_FILE", self.distribute_config_file
        )
        dist_barrier_timeout_env = os.environ.get("DIST_BARRIER_TIMEOUT")
        self.dist_barrier_timeout = (
            int(dist_barrier_timeout_env) if dist_barrier_timeout_env else None
        )
        self.gang_sleep_time = int(
            os.environ.get("GANG_SLEEP_TIME", self.gang_sleep_time)
        )
        self.gang_timeout_min = int(
            os.environ.get("GANG_TIMEOUT_MIN", self.gang_timeout_min)
        )
        self.json_gang_parts = os.environ.get("JSON_GANG_PARTS", self.json_gang_parts)
        self.leader_address = os.environ.get("LEADER_ADDRESS", self.leader_address)

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
        self.vit_separation: int = 0
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

    def update_from_env(self):
        self.vit_separation = int(os.environ.get("VIT_SEPARATION", self.vit_separation))
        self.vit_trt = int(os.environ.get("VIT_TRT", self.vit_trt))
        self.trt_cache_enabled = int(
            os.environ.get("TRT_CACHE_ENABLED", self.trt_cache_enabled)
        )
        self.trt_cache_path = os.environ.get("TRT_CACHE_PATH", self.trt_cache_path)
        self.download_headers = os.environ.get(
            "DOWNLOAD_HEADERS", self.download_headers
        )
        self.mm_cache_item_num = int(
            os.environ.get("MM_CACHE_ITEM_NUM", self.mm_cache_item_num)
        )
        self.url_cache_item_num = int(
            os.environ.get("url_cache_item_num", self.url_cache_item_num)
        )
        self.use_igraph_cache = get_env_bool("USE_IGRAPH_CACHE", self.use_igraph_cache)
        self.igraph_search_dom = get_env_str(
            "IGRAPH_SEARCH_DOM", self.igraph_search_dom
        )
        self.igraph_vipserver = get_env_int("IGRAPH_VIPSERVER", self.igraph_vipserver)
        self.igraph_table_name = get_env_str(
            "IGRAPH_TABLE_NAME", self.igraph_table_name
        )
        self.default_key = os.environ.get("IGRAPH_DEFAULT_KEY", self.default_key)

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

    def update_from_env(self):
        self.think_end_tag = os.environ.get("THINK_END_TAG", self.think_end_tag)
        self.think_end_token_id = int(
            os.environ.get("THINK_END_TOKEN_ID", self.think_end_token_id)
        )
        self.think_mode = int(os.environ.get("THINK_MODE", self.think_mode))
        self.force_stop_words = get_env_bool("FORCE_STOP_WORDS", self.force_stop_words)
        self.stop_words_list = os.environ.get("STOP_WORDS_LIST", self.stop_words_list)
        self.stop_words_str = os.environ.get("STOP_WORDS_STR", self.stop_words_str)
        self.think_start_tag = os.environ.get("THINK_START_TAG", self.think_start_tag)
        self.generation_config_path = os.environ.get(
            "GENERATION_CONFIG_PATH", self.generation_config_path
        )

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

    def update_from_env(self):
        self.int8_mode = int(os.environ.get("INT8_MODE", self.int8_mode))
        self.quantization = os.environ.get("QUANTIZATION", self.quantization)
        if self.int8_mode and not self.quantization:
            self.quantization = WEIGHT_TYPE.INT8.to_str()

    def to_string(self):
        return f"int8_mode: {self.int8_mode}\n" f"quantization: {self.quantization}"


class PyEplbConfig:
    def __init__(self):
        self.eplb_mode: str = "NONE"
        self.eplb_update_time: int = 5000
        self.redundant_expert: int = 0
        self.hack_ep_single_entry: int = 0
        self.balance_method: str = "mix"
        self.eplb_force_repack: int = 0
        self.eplb_stats_window_size: int = 10

    def update_from_env(self):
        self.eplb_mode = os.environ.get("EPLB_MODE", self.eplb_mode)
        self.eplb_update_time = int(
            os.environ.get("EPLB_UPDATE_TIME", self.eplb_update_time)
        )
        self.redundant_expert = int(
            os.environ.get("REDUNDANT_EXPERT", self.redundant_expert)
        )
        self.hack_ep_single_entry = int(
            os.environ.get("HACK_EP_SINGLE_ENTRY", self.hack_ep_single_entry)
        )
        self.balance_method = os.environ.get("BALANCE_METHOD", self.balance_method)
        self.eplb_force_repack = int(
            os.environ.get("EPLB_FORCE_REPACK", self.eplb_force_repack)
        )
        self.eplb_stats_window_size = int(
            os.environ.get("EPLB_STATS_WINDOW_SIZE", self.eplb_stats_window_size)
        )

    def to_string(self):
        return (
            f"eplb_mode: {self.eplb_mode}\n"
            f"eplb_update_time: {self.eplb_update_time}\n"
            f"redundant_expert: {self.redundant_expert}\n"
            f"hack_ep_single_entry: {self.hack_ep_single_entry}\n"
            f"balance_method: {self.balance_method}\n"
            f"eplb_force_repack: {self.eplb_force_repack}\n"
            f"eplb_stats_window_size: {self.eplb_stats_window_size}"
        )


class PyKvCacheConfig:
    def __init__(self):
        self.int8_kv_cache: int = 0
        self.fp8_kv_cache: int = 0
        self.kv_cache_mem_mb: int = -1
        self.seq_size_per_block: int = -1
        self.test_block_num: int = 0
        self.use_block_cache: Optional[int] = None
        self.blockwise_use_fp8_kv_cache: int = 0
        self.kv_cache_dtype: Optional[str] = None
        self.reuse_cache: bool = False

    def update_from_env(self):
        self.int8_kv_cache = int(os.environ.get("INT8_KV_CACHE", self.int8_kv_cache))
        self.fp8_kv_cache = int(os.environ.get("FP8_KV_CACHE", self.fp8_kv_cache))
        self.kv_cache_mem_mb = int(
            os.environ.get("KV_CACHE_MEM_MB", self.kv_cache_mem_mb)
        )
        self.seq_size_per_block = int(
            os.environ.get("SEQ_SIZE_PER_BLOCK", self.seq_size_per_block)
        )
        self.test_block_num = int(os.environ.get("TEST_BLOCK_NUM", self.test_block_num))
        use_block_cache = os.environ.get("USE_BLOCK_CACHE")
        if use_block_cache is not None:
            self.use_block_cache = int(use_block_cache)
        self.blockwise_use_fp8_kv_cache = int(
            os.environ.get(
                "BLOCKWISE_USE_FP8_KV_CACHE", self.blockwise_use_fp8_kv_cache
            )
        )
        self.kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", None)
        if self.int8_kv_cache:
            self.kv_cache_dtype = WEIGHT_TYPE.INT8.to_str()
        elif self.blockwise_use_fp8_kv_cache or self.fp8_kv_cache:
            self.kv_cache_dtype = WEIGHT_TYPE.FP8.to_str()
        elif int(os.environ.get("USE_FLOAT32", 0)):
            self.kv_cache_dtype = WEIGHT_TYPE.FP32.to_str()
        if not self.kv_cache_dtype:
            self.kv_cache_dtype = WEIGHT_TYPE.AUTO.to_str()
        self.reuse_cache = get_env_bool("REUSE_CACHE", self.reuse_cache)

    def to_string(self):
        return (
            f"int8_kv_cache: {self.int8_kv_cache}\n"
            f"fp8_kv_cache: {self.fp8_kv_cache}\n"
            f"kv_cache_dtype: {self.kv_cache_dtype}\n"
            f"kv_cache_mem_mb: {self.kv_cache_mem_mb}\n"
            f"seq_size_per_block: {self.seq_size_per_block}\n"
            f"test_block_num: {self.test_block_num}\n"
            f"use_block_cache: {self.use_block_cache}\n"
            f"blockwise_use_fp8_kv_cache: {self.blockwise_use_fp8_kv_cache}\n"
            f"reuse_cache: {self.reuse_cache}"
        )


class PyDeviceResourceConfig:
    def __init__(self):
        self.reserver_runtime_mem_mb: int = 1024
        self.specify_gpu_arch: str = ""
        self.acext_gemm_config_dir: Optional[str] = None
        self.device_reserve_memory_bytes: int = -1073741824

    def update_from_env(self):
        self.reserver_runtime_mem_mb = int(
            os.environ.get("RESERVER_RUNTIME_MEM_MB", self.reserver_runtime_mem_mb)
        )
        self.specify_gpu_arch = os.environ.get(
            "SPECIFY_GPU_ARCH", self.specify_gpu_arch
        )
        self.acext_gemm_config_dir = os.environ.get(
            "ACEXT_GEMM_CONFIG_DIR", self.acext_gemm_config_dir
        )
        self.device_reserve_memory_bytes = int(
            os.environ.get(
                "DEVICE_RESERVE_MEMORY_BYTES", self.device_reserve_memory_bytes
            )
        )

    def to_string(self):
        return (
            f"reserver_runtime_mem_mb: {self.reserver_runtime_mem_mb}\n"
            f"specify_gpu_arch: {self.specify_gpu_arch}\n"
            f"acext_gemm_config_dir: {self.acext_gemm_config_dir}\n"
            f"device_reserve_memory_bytes: {self.device_reserve_memory_bytes}"
        )


class SparseConfig:
    def __init__(self):
        self.sparse_config_file: Optional[str] = None

    def update_from_env(self):
        self.sparse_config_file = os.environ.get(
            "SPARSE_CONFIG_FILE", self.sparse_config_file
        )

    def to_string(self):
        return f"sparse_config_file: {self.sparse_config_file}"


class EngineConfig:
    def __init__(self):
        self.warm_up: int = 1
        self.warm_up_with_loss: int = 0
        self.max_seq_len: int = 0

    def update_from_env(self):
        self.warm_up = int(os.environ.get("WARM_UP", self.warm_up))
        self.warm_up_with_loss = int(
            os.environ.get("WARM_UP_WITH_LOSS", self.warm_up_with_loss)
        )
        self.max_seq_len = int(os.environ.get("MAX_SEQ_LEN", self.max_seq_len))

    def to_string(self):
        return (
            f"warm_up: {self.warm_up}\n"
            f"warm_up_with_loss: {self.warm_up_with_loss}\n"
            f"max_seq_len: {self.max_seq_len}"
        )


class EmbeddingConfig:
    def __init__(self):
        self.embedding_model: int = 0
        self.extra_input_in_mm_embedding = ""

    def update_from_env(self):
        self.embedding_model = int(
            os.environ.get("EMBEDDING_MODEL", self.embedding_model)
        )
        self.extra_input_in_mm_embedding = os.environ.get(
            "EXTRA_INPUT_IN_MM_EMBEDDING", self.extra_input_in_mm_embedding
        )

    def to_string(self):
        return (
            f"embedding_model: {self.embedding_model}\n"
            f"extra_input_in_mm_embedding: {self.extra_input_in_mm_embedding}"
        )


class RoleConfig:
    def __init__(self):
        self.role_type: RoleType = RoleType.PDFUSION

    def update_from_env(self):
        self.role_type = self._trans_role_type(os.environ.get("ROLE_TYPE", ""))

    def to_string(self):
        return f"role_type: {self.role_type.name}"

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


class PdSeparationConfig:
    def __init__(self):
        # Prefill related configuration
        self.prefill_retry_times: int = 0
        self.prefill_retry_timeout_ms: int = 20
        self.prefill_max_wait_timeout_ms: int = 600 * 1000

        # Decode related configuration
        self.decode_retry_times: int = 100
        self.decode_retry_timeout_ms: int = 100
        self.decode_retry_interval_ms: int = 1
        self.decode_polling_kv_cache_step_ms: int = 30
        self.decode_entrance: int = 0

        # RDMA related configuration
        self.rdma_connect_retry_times: int = 0
        self.load_cache_timeout_ms: int = 5000

    def update_from_env(self):
        # Prefill related configuration
        self.prefill_retry_times = int(
            os.environ.get("PREFILL_RETRY_TIMES", self.prefill_retry_times)
        )
        self.prefill_retry_timeout_ms = int(
            os.environ.get("PREFILL_RETRY_TIMEOUT_MS", self.prefill_retry_timeout_ms)
        )
        self.prefill_max_wait_timeout_ms = int(
            os.environ.get(
                "PREFILL_MAX_WAIT_TIMEOUT_MS", self.prefill_max_wait_timeout_ms
            )
        )

        # Decode related configuration
        self.decode_retry_times = int(
            os.environ.get("DECODE_RETRY_TIMES", self.decode_retry_times)
        )
        self.decode_retry_timeout_ms = int(
            os.environ.get("DECODE_RETRY_TIMEOUT_MS", self.decode_retry_timeout_ms)
        )
        self.decode_retry_interval_ms = int(
            os.environ.get("DECODE_RETRY_INTERVAL_MS", self.decode_retry_interval_ms)
        )
        self.decode_polling_kv_cache_step_ms = int(
            os.environ.get(
                "DECODE_POLLING_KV_CACHE_STEP_MS", self.decode_polling_kv_cache_step_ms
            )
        )
        self.decode_entrance = int(
            os.environ.get("DECODE_ENTRANCE", self.decode_entrance)
        )

        # RDMA related configuration
        self.rdma_connect_retry_times = int(
            os.environ.get("RDMA_CONNECT_RETRY_TIMES", self.rdma_connect_retry_times)
        )
        self.load_cache_timeout_ms = int(
            os.environ.get("LOAD_CACHE_TIMEOUT_MS", self.load_cache_timeout_ms)
        )

    def to_string(self):
        return (
            f"prefill_retry_times: {self.prefill_retry_times}\n"
            f"prefill_retry_timeout_ms: {self.prefill_retry_timeout_ms}\n"
            f"prefill_max_wait_timeout_ms: {self.prefill_max_wait_timeout_ms}\n"
            f"decode_retry_times: {self.decode_retry_times}\n"
            f"decode_retry_timeout_ms: {self.decode_retry_timeout_ms}\n"
            f"decode_retry_interval_ms: {self.decode_retry_interval_ms}\n"
            f"decode_polling_kv_cache_step_ms: {self.decode_polling_kv_cache_step_ms}\n"
            f"decode_entrance: {self.decode_entrance}\n"
            f"rdma_connect_retry_times: {self.rdma_connect_retry_times}\n"
            f"load_cache_timeout_ms: {self.load_cache_timeout_ms}"
        )


class WorkerConfig:
    def __init__(self):
        self.worker_info_port_num: int = MIN_WORKER_INFO_PORT_NUM

    def update_from_env(self):
        self.worker_info_port_num = int(
            os.environ.get("WORKER_INFO_PORT_NUM", self.worker_info_port_num)
        )

    def to_string(self):
        return f"worker_info_port_num: {self.worker_info_port_num}"


class JITConfig:
    def __init__(self):
        self.remote_jit_dir: str = ""

    def update_from_env(self):
        self.remote_jit_dir = os.environ.get("REMOTE_JIT_DIR", self.remote_jit_dir)
        os.environ["REMOTE_JIT_DIR"] = fetch_remote_file_to_local(
            self.remote_jit_dir, MountRwMode.RWMODE_RW
        )

    def to_string(self):
        return f"remote_jit_dir: {self.remote_jit_dir}"


class PyHwKernelConfig:
    def __init__(self):
        self.deep_gemm_num_sm: int = -1
        self.arm_gemm_use_kai: bool = False
        self.enable_stable_scatter_add: bool = False
        self.enable_multi_block_mode: bool = True
        self.ft_disable_custom_ar: bool = True
        self.rocm_hipblaslt_config: str = "gemm_config.csv"
        self.use_swizzleA = False
        self.enable_cuda_graph: bool = False
        self.enable_cuda_graph_debug_mode: bool = False
        self.use_aiter_pa: bool = True
        self.use_asm_pa: bool = True
        self.enable_native_cuda_graph: bool = False
        self.num_native_cuda_graph: int = 200

    def update_from_env(self):
        self.deep_gemm_num_sm = get_env_int("DEEP_GEMM_NUM_SM", self.deep_gemm_num_sm)
        self.arm_gemm_use_kai = get_env_bool("ARM_GEMM_USE_KAI", self.arm_gemm_use_kai)
        self.enable_stable_scatter_add = get_env_bool(
            "ENABLE_STABLE_SCATTER_ADD", self.enable_stable_scatter_add
        )
        self.enable_multi_block_mode = get_env_bool(
            "ENABLE_MULTI_BLOCK_MODE", self.enable_multi_block_mode
        )
        self.ft_disable_custom_ar = get_env_bool(
            "FT_DISABLE_CUSTOM_AR", self.ft_disable_custom_ar
        )
        self.rocm_hipblaslt_config = get_env_str(
            "ROCM_HIPBLASLT_CONFIG", self.rocm_hipblaslt_config
        )
        self.use_swizzleA = get_env_bool("USE_SWIZZLEA", self.use_swizzleA)
        self.enable_cuda_graph = get_env_bool(
            "ENABLE_CUDA_GRAPH", self.enable_cuda_graph
        )
        self.enable_cuda_graph_debug_mode = get_env_bool(
            "ENABLE_CUDA_GRAPH_DEBUG_MODE", self.enable_cuda_graph_debug_mode
        )
        self.use_aiter_pa = get_env_bool("USE_AITER_PA", self.use_aiter_pa)
        self.use_asm_pa = get_env_bool("USE_ASM_PA", self.use_asm_pa)
        self.enable_native_cuda_graph = get_env_bool(
            "ENABLE_NATIVE_CUDA_GRAPH", self.enable_native_cuda_graph
        )
        self.num_native_cuda_graph = get_env_int(
            "NUM_NATIVE_CUDA_GRAPH", self.num_native_cuda_graph
        )

    def to_string(self):
        return (
            f"deep_gemm_num_sm: {self.deep_gemm_num_sm}\n"
            f"arm_gemm_use_kai: {self.arm_gemm_use_kai}\n"
            f"enable_stable_scatter_add: {self.enable_stable_scatter_add}\n"
            f"enable_multi_block_mode: {self.enable_multi_block_mode}\n"
            f"ft_disable_custom_ar: {self.ft_disable_custom_ar}\n"
            f"rocm_hipblaslt_config: {self.rocm_hipblaslt_config}\n"
            f"use_swizzleA: {self.use_swizzleA}\n"
            f"enable_cuda_graph: {self.enable_cuda_graph}\n"
            f"enable_cuda_graph_debug_mode: {self.enable_cuda_graph_debug_mode}\n"
            f"use_aiter_pa: {self.use_aiter_pa}\n"
            f"use_asm_pa: {self.use_asm_pa}\n"
            f"enable_native_cuda_graph: {self.enable_native_cuda_graph}\n"
            f"num_native_cuda_graph: {self.num_native_cuda_graph}"
        )


class PyEnvConfigs:
    def __init__(self):
        self.server_config: ServerConfig = ServerConfig()
        self.profiling_debug_config: ProfilingDebugLoggingConfig = (
            ProfilingDebugLoggingConfig()
        )
        self.model_config: ModelConfig = ModelConfig()
        self.py_speculative_execution_config: PySpeculativeExecutionConfig = (
            PySpeculativeExecutionConfig()
        )
        self.lora_config: LoraConfig = LoraConfig()
        self.load_config: LoadConfig = LoadConfig()
        self.render_config: RenderConfig = RenderConfig()
        self.gang_config: GangConfig = GangConfig()
        self.vit_config: VitConfig = VitConfig()
        self.generate_env_config: GenerateEnvConfig = GenerateEnvConfig()
        self.quantization_config: QuantizationConfig = QuantizationConfig()
        self.py_eplb_config: PyEplbConfig = PyEplbConfig()
        self.py_kv_cache_config: PyKvCacheConfig = PyKvCacheConfig()
        self.py_device_resource_config: PyDeviceResourceConfig = (
            PyDeviceResourceConfig()
        )
        self.sparse_config: SparseConfig = SparseConfig()
        self.engine_config: EngineConfig = EngineConfig()
        self.embedding_config: EmbeddingConfig = EmbeddingConfig()
        self.worker_config: WorkerConfig = WorkerConfig()
        self.role_config: RoleConfig = RoleConfig()
        self.pd_separation_config: PdSeparationConfig = PdSeparationConfig()
        self.parallelism_distributed_config: ParallelismDistributedConfig = (
            ParallelismDistributedConfig()
        )
        self.ffn_disaggregate_config: FfnDisAggregateConfig = FfnDisAggregateConfig()
        self.model_specific_config = ModelSpecificConfig()
        self.fmha_config = FMHAConfig()
        self.misc_config = MiscellaneousConfig()
        self.concurrency_config = ConcurrencyConfig()
        self.jit_config = JITConfig()
        self.py_hw_kernel_config = PyHwKernelConfig()

    def update_from_env(self):
        self.server_config.update_from_env()
        self.profiling_debug_config.update_from_env()
        self.model_config.update_from_env()
        self.py_speculative_execution_config.update_from_env()
        self.lora_config.update_from_env()
        self.load_config.update_from_env()
        self.render_config.update_from_env()
        self.gang_config.update_from_env()
        self.vit_config.update_from_env()
        self.generate_env_config.update_from_env()
        self.quantization_config.update_from_env()
        self.py_eplb_config.update_from_env()
        self.py_kv_cache_config.update_from_env()
        self.py_device_resource_config.update_from_env()
        self.sparse_config.update_from_env()
        self.engine_config.update_from_env()
        self.embedding_config.update_from_env()
        self.worker_config.update_from_env()
        self.role_config.update_from_env()
        self.pd_separation_config.update_from_env()
        # in gpt model parameters, we should update it from g_parallel_info
        self.parallelism_distributed_config.update_from_env()
        self.model_specific_config.update_from_env()
        self.fmha_config.update_from_env()
        self.misc_config.update_from_env()
        self.concurrency_config.update_from_env()
        self.ffn_disaggregate_config.update_from_env()
        self.jit_config.update_from_env()
        self.py_hw_kernel_config.update_from_env()
        logging.info(self.to_string())

    def to_string(self):
        return (
            "[server_config]\n" + self.server_config.to_string() + "\n\n"
            "[profiling_debug_config]\n"
            + self.profiling_debug_config.to_string()
            + "\n\n"
            "[model_config]\n" + self.model_config.to_string() + "\n\n"
            "[py_speculative_execution_config]\n"
            + self.py_speculative_execution_config.to_string()
            + "\n\n"
            "[lora_config]\n" + self.lora_config.to_string() + "\n\n"
            "[load_config]\n" + self.load_config.to_string() + "\n\n"
            "[render_config]\n" + self.render_config.to_string() + "\n\n"
            "[gang_config]\n" + self.gang_config.to_string() + "\n\n"
            "[vit_config]\n" + self.vit_config.to_string() + "\n\n"
            "[generate_env_config]\n" + self.generate_env_config.to_string() + "\n\n"
            "[quantization_config]\n" + self.quantization_config.to_string() + "\n\n"
            "[py_eplb_config]\n" + self.py_eplb_config.to_string() + "\n\n"
            "[py_kv_cache_config]\n" + self.py_kv_cache_config.to_string() + "\n\n"
            "[py_device_resource_config]\n"
            + self.py_device_resource_config.to_string()
            + "\n\n"
            "[sparse_config]\n" + self.sparse_config.to_string() + "\n\n"
            "[engine_config]\n" + self.engine_config.to_string() + "\n\n"
            "[embedding_config]\n" + self.embedding_config.to_string() + "\n\n"
            "[worker_config]\n" + self.worker_config.to_string() + "\n\n"
            "[role_config]\n" + self.role_config.to_string() + "\n\n"
            "[pd_separation_config]\n" + self.pd_separation_config.to_string() + "\n\n"
            "[parallelism_distributed_config]\n"
            + self.parallelism_distributed_config.to_string()
            + "\n\n"
            "[model_specific_config]\n"
            + self.model_specific_config.to_string()
            + "\n\n"
            "[fmha_config]\n" + self.fmha_config.to_string() + "\n\n"
            "[misc_config]\n" + self.misc_config.to_string() + "\n\n"
            "[concurrency_config]\n" + self.concurrency_config.to_string() + "\n\n"
            "[jit_config]\n" + self.jit_config.to_string() + "\n\n"
            "[py_hw_kernel_config]\n" + self.py_hw_kernel_config.to_string() + "\n\n"
        )


# some configs are from static method or global method, etc, we collect them in `StaticConfig`, but in-none-static methods,
# we should use configs alone. This design can make the codes of this project more clear. All configs
# should be retrived from `StaticConfig` or a top-down `PyEnvConfigs`. Notably, we don't modify smoke
# test envs and that's necessary.
StaticConfig = PyEnvConfigs()
StaticConfig.update_from_env()

# The envs we reserve below:
# 1. weights convert: because we don't use it in our project.
# 2. smoke test.
