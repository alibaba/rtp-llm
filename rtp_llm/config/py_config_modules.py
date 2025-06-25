from rtp_llm.ops import ConcurrencyConfig, DeviceResourceConfig, FMHAConfig, HWKernelConfig, KVCacheConfig, MiscellaneousConfig, ModelSpecificConfig, ParallelismDistributedConfig, ProfilingDebugLoggingConfig, ServiceDiscoveryConfig, SchedulerConfig, MoeConfig, SamplerConfig, SpeculativeExecutionConfig, CacheStoreConfig, BatchDecodeSchedulerConfig, FIFOSchedulerConfig
from typing import Optional
import os

DEFAULT_START_PORT = 8088
MASTER_INFO_PORT_NUM = 11
MIN_WORKER_INFO_PORT_NUM = 7
WORKER_INFO_PORT_NUM = MIN_WORKER_INFO_PORT_NUM

def get_env_int(name:str, default:int=-1):
    v = os.environ.get(name, None)
    return int(v) if v is not None and v != "" else default

def get_env_str(name:str, default:str=""):
    v = os.environ.get(name, None)
    return v if v is not None else default

def get_env_bool(name:str, default:bool=False):
    ## in fact, we can always get value from env, if that's not specified, we return default value
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    return v.lower() == "1" or v.lower() == "on" or v.lower() == "true"

class ServerConfig:
    def __init__(self):
        self.fronted_server_count = 4
        self.start_port = 8088
        self.timeout_keep_alive = 5
        self.fronted_server_id = 0

    def update_from_env(self):
        self.fronted_server_count = int(os.environ.get("FRONTED_SERVER_COUNT", self.fronted_server_count))
        self.start_port = int(os.environ.get("START_PORT", self.start_port))
        self.timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", self.timeout_keep_alive))
        self.fronted_server_id = int(os.environ.get("FRONTED_SERVER_ID", self.fronted_server_id))

    def to_string(self):
        return (
            f"fronted_server_count: {self.fronted_server_count}\n"
            f"start_port: {self.start_port}\n"
            f"timeout_keep_alive: {self.timeout_keep_alive}\n"
            f"fronted_server_id: {self.fronted_server_id}"
        )

class ModelConfig:
    def __init__(self):
        self.extra_data_path: Optional[str] = None
        self.local_extra_data_path: Optional[str] = None
        self.tokenizer_path: Optional[str] = None
        self.act_type: str = 'FP16'
        self.use_float32: Optional[bool] = None
        self.original_checkpoint_path: Optional[str] = None
        self.mla_ops_type: str = 'AUTO'
        self.parallel_batch: int = 0
        self.ft_plugin_path: Optional[str] = None
        self.weight_type: Optional[str] = None
        self.task_type: Optional[str] = None
        self.model_type: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.oss_endpoint: Optional[str] = None
        self.ptuning_path: Optional[str] = None

    def update_from_env(self):
        self.extra_data_path = os.environ.get("EXTRA_DATA_PATH", self.extra_data_path)
        self.local_extra_data_path = os.environ.get("LOCAL_EXTRA_DATA_PATH", self.local_extra_data_path)
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", self.tokenizer_path)
        self.act_type = os.environ.get("ACT_TYPE", self.act_type)
        use_float32 = os.environ.get("USE_FLOAT32")
        if use_float32 is not None:
            self.use_float32 = get_env_bool(use_float32)
        self.original_checkpoint_path = os.environ.get("ORIGINAL_CHECKPOINT_PATH", self.original_checkpoint_path)
        self.mla_ops_type = os.environ.get("MLA_OPS_TYPE", self.mla_ops_type)
        self.parallel_batch = int(os.environ.get("PARALLEL_BATCH", self.parallel_batch))
        self.ft_plugin_path = os.environ.get("FT_PLUGIN_PATH", self.ft_plugin_path)
        self.weight_type = os.environ.get("WEIGHT_TYPE", self.weight_type)
        self.task_type = os.environ.get("TASK_TYPE", self.task_type)
        self.model_type = os.environ.get("MODEL_TYPE", self.model_type)
        self.checkpoint_path = os.environ.get("CHECKPOINT_PATH", self.checkpoint_path)
        self.oss_endpoint = os.environ.get("OSS_ENDPOINT", self.oss_endpoint)
        self.ptuning_path = os.environ.get("PTUNING_PATH", self.ptuning_path)

    def to_string(self):
            return (
                f"extra_data_path: {self.extra_data_path}\n"
                f"local_extra_data_path: {self.local_extra_data_path}\n"
                f"tokenizer_path: {self.tokenizer_path}\n"
                f"act_type: {self.act_type}\n"
                f"use_float32: {self.use_float32}\n"
                f"original_checkpoint_path: {self.original_checkpoint_path}\n"
                f"mla_ops_type: {self.mla_ops_type}\n"
                f"parallel_batch: {self.parallel_batch}\n"
                f"ft_plugin_path: {self.ft_plugin_path}\n"
                f"weight_type: {self.weight_type}\n"
                f"task_type: {self.task_type}\n"
                f"model_type: {self.model_type}\n"
                f"checkpoint_path: {self.checkpoint_path}\n"
                f"oss_endpoint: {self.oss_endpoint}\n"
                f"ptuning_path: {self.ptuning_path}"
            )

# Todo: 合并到c++的SpeculativeExecutionConfig
class PySpeculativeExecutionConfig:
    def __init__(self):
        self.gen_num_per_circle: int = 5
        self.sp_quantization: Optional[str] = None
        self.sp_checkpoint_path: Optional[str] = None

    def update_from_env(self):
        self.gen_num_per_circle = int(os.environ.get("GEN_NUM_PER_CIRCLE", self.gen_num_per_circle))
        self.sp_quantization = os.environ.get("SP_QUANTIZATION", self.sp_quantization)
        self.sp_checkpoint_path = os.environ.get("SP_CHECKPOINT_PATH", self.sp_checkpoint_path)

    def to_string(self):
        return (
            f"gen_num_per_circle: {self.gen_num_per_circle}\n"
            f"sp_quantization: {self.sp_quantization}\n"
            f"sp_checkpoint_path: {self.sp_checkpoint_path}"
        )

class LoraConfig:
    def __init__(self):
        self.lora_info: str = '{}'
        self.merge_lora: bool = True

    def update_from_env(self):
        self.lora_info = os.environ.get("LORA_INFO", self.lora_info)
        merge_lora = os.environ.get("MERGE_LORA")
        if merge_lora is not None:
            self.merge_lora = merge_lora.lower() == "true"

    def to_string(self):
        return (
            f"lora_info: {self.lora_info}\n"
            f"merge_lora: {self.merge_lora}"
        )

class LoadConfig:
    def __init__(self):
        self.phy2log_path: str = ""
        self.converter_num_per_gpu: int = 4
        self.tokenizers_parallelism: bool = False
        self.load_ckpt_num_process: int = 0

    def update_from_env(self):
        self.phy2log_path = os.environ.get("PHY2LOG_PATH", self.phy2log_path)
        self.converter_num_per_gpu = int(os.environ.get("CONVERTER_NUM_PER_GPU", self.converter_num_per_gpu))
        tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
        if tokenizers_parallelism is not None:
            self.tokenizers_parallelism = tokenizers_parallelism.lower() == "true"
        self.load_ckpt_num_process = int(os.environ.get("CKPT_NUM_PROCESS", self.load_ckpt_num_process))

    def to_string(self):
        return (
            f"phy2log_path: {self.phy2log_path}\n"
            f"converter_num_per_gpu: {self.converter_num_per_gpu}\n"
            f"tokenizers_parallelism: {self.tokenizers_parallelism}\n"
            f"load_ckpt_num_process: {self.load_ckpt_num_process}"
        )

class RenderConfig:
    def __init__(self):
        self.model_template_type: Optional[str] = None
        self.default_chat_template_key: str = 'default'
        self.default_tool_use_template_key: str = 'tool_use'
        self.llava_chat_template: str = ''

    def update_from_env(self):
        self.model_template_type = os.environ.get("MODEL_TEMPLATE_TYPE", self.model_template_type)
        self.default_chat_template_key = os.environ.get("DEFAULT_CHAT_TEMPLATE_KEY", self.default_chat_template_key)
        self.default_tool_use_template_key = os.environ.get("DEFAULT_TOOL_USE_TEMPLATE_KEY", self.default_tool_use_template_key)
        self.llava_chat_template = os.environ.get("LLAVA_CHAT_TEMPLATE", self.llava_chat_template)

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
        self.distribute_config_file: Optional[str] = None
        self.dist_barrier_timeout: int = 45
        self.gang_sleep_time: int = 10
        self.gang_timeout_min: int = 30

    def update_from_env(self):
        fake_gang_env = os.environ.get("FAKE_GANG_ENV")
        if fake_gang_env is not None:
            self.fake_gang_env = fake_gang_env.lower() == "true"
        self.gang_annocation_path = os.environ.get("ANNOCATION_PATH", self.gang_annocation_path)
        self.gang_config_string = os.environ.get("CONFIG_STRING", self.gang_config_string)
        self.zone_name = os.environ.get("ZONE_NAME", self.zone_name)
        self.distribute_config_file = os.environ.get("DISTRIBUTE_CONFIG_FILE", self.distribute_config_file)
        self.dist_barrier_timeout = int(os.environ.get("DIST_BARRIER_TIMEOUT", self.dist_barrier_timeout))
        self.gang_sleep_time = int(os.environ.get("SLEEP_TIME", self.gang_sleep_time))
        self.gang_timeout_min = int(os.environ.get("TIMEOUT_MIN", self.gang_timeout_min))

    def to_string(self):
        return (
            f"fake_gang_env: {self.fake_gang_env}\n"
            f"gang_annocation_path: {self.gang_annocation_path}\n"
            f"gang_config_string: {self.gang_config_string}\n"
            f"zone_name: {self.zone_name}\n"
            f"distribute_config_file: {self.distribute_config_file}\n"
            f"dist_barrier_timeout: {self.dist_barrier_timeout}\n"
            f"gang_sleep_time: {self.gang_sleep_time}\n"
            f"gang_timeout_min: {self.gang_timeout_min}"
        )

class VitConfig:
    def __init__(self):
        self.vit_separation: int = 0
        self.vit_trt: int = 0
        self.trt_cache_enabled: int = 0
        self.trt_cache_path: Optional[str] = None
        self.download_headers: str = ''
        self.mm_cache_item_num: int = 10

    def update_from_env(self):
        self.vit_separation = int(os.environ.get("VIT_SEPARATION", self.vit_separation))
        self.vit_trt = int(os.environ.get("VIT_TRT", self.vit_trt))
        self.trt_cache_enabled = int(os.environ.get("TRT_CACHE_ENABLED", self.trt_cache_enabled))
        self.trt_cache_path = os.environ.get("TRT_CACHE_PATH", self.trt_cache_path)
        self.download_headers = os.environ.get("DOWNLOAD_HEADERS", self.download_headers)
        self.mm_cache_item_num = int(os.environ.get("MM_CACHE_ITEM_NUM", self.mm_cache_item_num))

    def to_string(self):
        return (
            f"vit_separation: {self.vit_separation}\n"
            f"vit_trt: {self.vit_trt}\n"
            f"trt_cache_enabled: {self.trt_cache_enabled}\n"
            f"trt_cache_path: {self.trt_cache_path}\n"
            f"download_headers: {self.download_headers}\n"
            f"mm_cache_item_num: {self.mm_cache_item_num}"
        )

class GenerateConfig:
    def __init__(self):
        self.think_end_tag: str = "<think>\n\n"
        self.think_end_token_id: int = -1
        self.think_mode: int = 0
        self.force_stop_words: bool = False
        self.stop_words_list: Optional[str] = None
        self.stop_words_str: Optional[str] = None
        self.think_start_tag: str = "<think>\n"
        self.generation_config_path: Optional[str] = None

    def update_from_env(self):
        self.think_end_tag = os.environ.get("THINK_END_TAG", self.think_end_tag)
        self.think_end_token_id = int(os.environ.get("THINK_END_TOKEN_ID", self.think_end_token_id))
        self.think_mode = int(os.environ.get("THINK_MODE", self.think_mode))
        force_stop_words = os.environ.get("FORCE_STOP_WORDS")
        if force_stop_words is not None:
            self.force_stop_words = force_stop_words.lower() == "true"
        self.stop_words_list = os.environ.get("STOP_WORDS_LIST", self.stop_words_list)
        self.stop_words_str = os.environ.get("STOP_WORDS_STR", self.stop_words_str)
        self.think_start_tag = os.environ.get("THINK_START_TAG", self.think_start_tag)
        self.generation_config_path = os.environ.get("GENERATION_CONFIG_PATH", self.generation_config_path)

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

    def update_from_env(self):
        self.int8_mode = int(os.environ.get("INT8_MODE", self.int8_mode))

    def to_string(self):
        return f"int8_mode: {self.int8_mode}"

class PyEplbConfig:
    def __init__(self):
        self.eplb_mode: str = 'NONE'
        self.eplb_update_time: int = 5000
        self.redundant_expert: int = 0
        self.hack_ep_single_entry: int = 0
        self.balance_method: str = 'mix'
        self.eplb_force_repack: int = 0
        self.eplb_stats_window_size: int = 10

    def update_from_env(self):
        self.eplb_mode = os.environ.get("EPLB_MODE", self.eplb_mode)
        self.eplb_update_time = int(os.environ.get("EPLB_UPDATE_TIME", self.eplb_update_time))
        self.redundant_expert = int(os.environ.get("REDUNDANT_EXPERT", self.redundant_expert))
        self.hack_ep_single_entry = int(os.environ.get("HACK_EP_SINGLE_ENTRY", self.hack_ep_single_entry))
        self.balance_method = os.environ.get("BALANCE_METHOD", self.balance_method)
        self.eplb_force_repack = int(os.environ.get("EPLB_FORCE_REPACK", self.eplb_force_repack))
        self.eplb_stats_window_size = int(os.environ.get("EPLB_STATS_WINDOW_SIZE", self.eplb_stats_window_size))

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
        self.kv_cache_mem_mb: int = -1
        self.seq_size_per_block: Optional[str] = None
        self.test_block_num: int = 0
        self.use_block_cache: Optional[int] = None

    def update_from_env(self):
        self.int8_kv_cache = int(os.environ.get("INT8_KV_CACHE", self.int8_kv_cache))
        self.kv_cache_mem_mb = int(os.environ.get("KV_CACHE_MEM_MB", self.kv_cache_mem_mb))
        self.seq_size_per_block = os.environ.get("SEQ_SIZE_PER_BLOCK", self.seq_size_per_block)
        self.test_block_num = int(os.environ.get("TEST_BLOCK_NUM", self.test_block_num))
        use_block_cache = os.environ.get("USE_BLOCK_CACHE")
        if use_block_cache is not None:
            self.use_block_cache = int(use_block_cache)

    def to_string(self):
        return (
            f"int8_kv_cache: {self.int8_kv_cache}\n"
            f"kv_cache_mem_mb: {self.kv_cache_mem_mb}\n"
            f"seq_size_per_block: {self.seq_size_per_block}\n"
            f"test_block_num: {self.test_block_num}\n"
            f"use_block_cache: {self.use_block_cache}"
        )

class PyDeviceResourceConfig:
    def __init__(self):
        self.reserver_runtime_mem_mb: int = 128
        self.specify_gpu_arch: str = ''
        self.acext_gemm_config_dir: Optional[str] = None

    def update_from_env(self):
        self.reserver_runtime_mem_mb = int(os.environ.get("RESERVER_RUNTIME_MEM_MB", self.reserver_runtime_mem_mb))
        self.specify_gpu_arch = os.environ.get("SPECIFY_GPU_ARCH", self.specify_gpu_arch)
        self.acext_gemm_config_dir = os.environ.get("ACEXT_GEMM_CONFIG_DIR", self.acext_gemm_config_dir)

    def to_string(self):
        return (
            f"reserver_runtime_mem_mb: {self.reserver_runtime_mem_mb}\n"
            f"specify_gpu_arch: {self.specify_gpu_arch}\n"
            f"acext_gemm_config_dir: {self.acext_gemm_config_dir}"
        )

class SparseConfig:
    def __init__(self):
        self.sparse_config_file: Optional[str] = None

    def update_from_env(self):
        self.sparse_config_file = os.environ.get("SPARSE_CONFIG_FILE", self.sparse_config_file)

    def to_string(self):
        return f"sparse_config_file: {self.sparse_config_file}"

class EngineConfig:
    def __init__(self):
        self.warm_up: int = 1
        self.warm_up_with_loss: int = 0
        self.max_seq_len: int = 0

    def update_from_env(self):
        self.warm_up = int(os.environ.get("WARM_UP", self.warm_up))
        self.warm_up_with_loss = int(os.environ.get("WARM_UP_WITH_LOSS", self.warm_up_with_loss))
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

    def update_from_env(self):
        self.embedding_model = int(os.environ.get("EMBEDDING_MODEL", self.embedding_model))

    def to_string(self):
        return f"embedding_model: {self.embedding_model}"

class WorkerConfig:
    def __init__(self):
        self.worker_info_port_num: int = 7

    def update_from_env(self):
        self.worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", self.worker_info_port_num))

    def to_string(self):
        return f"worker_info_port_num: {self.worker_info_port_num}"

class PyEnvConfigs:
    def __init__(self):
        self.server_config: ServerConfig = ServerConfig()
        self.profiling_debug_config: ProfilingDebugLoggingConfig = ProfilingDebugLoggingConfig()
        self.model_config: ModelConfig = ModelConfig()
        self.py_speculative_execution_config: PySpeculativeExecutionConfig = PySpeculativeExecutionConfig()
        self.lora_config: LoraConfig = LoraConfig()
        self.load_config: LoadConfig = LoadConfig()
        self.render_config: RenderConfig = RenderConfig()
        self.gang_config: GangConfig = GangConfig()
        self.vit_config: VitConfig = VitConfig()
        self.generate_config: GenerateConfig = GenerateConfig()
        self.quantization_config: QuantizationConfig = QuantizationConfig()
        self.py_eplb_config: PyEplbConfig = PyEplbConfig()
        self.py_kv_cache_config: PyKvCacheConfig = PyKvCacheConfig()
        self.py_device_resource_config: PyDeviceResourceConfig = PyDeviceResourceConfig()
        self.sparse_config: SparseConfig = SparseConfig()
        self.engine_config: EngineConfig = EngineConfig()
        self.embedding_config: EmbeddingConfig = EmbeddingConfig()
        self.worker_config: WorkerConfig = WorkerConfig()

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
        self.generate_config.update_from_env()
        self.quantization_config.update_from_env()
        self.py_eplb_config.update_from_env()
        self.py_kv_cache_config.update_from_env()
        self.py_device_resource_config.update_from_env()
        self.sparse_config.update_from_env()
        self.engine_config.update_from_env()
        self.embedding_config.update_from_env()
        self.worker_config.update_from_env()
