from typing import Dict, Any, List, Optional, Set
import os
import json
import torch
import logging
import typing
# make sure so init
from dataclasses import dataclass, field, fields
from enum import Enum
from maga_transformer.utils.util import str_to_bool, closest_power_of_2
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.config.task_type import TaskType, check_task_type
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info, g_master_info, g_worker_info, WORKER_INFO_PORT_NUM
from maga_transformer.distribute.gang_info import get_gang_info, GangInfo
from maga_transformer.ops import GptInitParameter, QuantAlgo, SpecialTokens, MlaOpsType, EplbMode
from maga_transformer.utils.gemm_utils.cutlass_config import load_cutlass_gemm_config

updated_params: Set[str] = set()

def get_pad_size(size: int , align_size: int):
    return (align_size - (size % align_size)) % align_size

class DataClassBase:
    @classmethod
    def from_dict(cls, kvs: Dict[str, Any]):
        n_kvs = {k: v for k, v in kvs.items() if k in {f.name for f in fields(cls)}}

        # 兼容老的sparse config使用的key 没有加layer
        for k, v in kvs.items():
            if k in ["head_num", "inter_size"] and isinstance(v, list):
                n_kvs.update({"layer_"+k : v})

        data_class = cls(**n_kvs)
        return data_class

mc_sim_7b_63 = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]

@dataclass
class SparseConfig(DataClassBase):
    layer_num: int = 0
    layer_head_num: List[int] = field(default_factory=lambda: [])
    layer_inter_size: List[int] = field(default_factory=lambda: [])

    def check(self) -> bool:
        if self.layer_num == 0:
            logging.info("sparse config layer_num must not be empty")
            return False
        if len(self.layer_head_num) != self.layer_num:
            logging.info(f"sparse config layer_num and head_num must match, layer_num: {self.layer_num}, head_num: {self.layer_head_num}")
            return False
        if len(self.layer_inter_size) != self.layer_num:
            logging.info(f"sparse config layer_num and inter_size must match, layer_num: {self.layer_num}, inter_size: {self.layer_inter_size}")
            return False
        return True

class VitParameters:
    # config includes origin vit config in ckpt/config.json
    config: Dict[str, Any] = {}
    special_token_ids: Dict[str, Any] = {}
    special_tokens: Dict[str, Any] = {}
    vit_weights: Any = None


class TemplateType(Enum):
    chat = "chat"
    vqa = "vqa"
    base = "image"


class ConfigMode(Enum):
    SimpleMode = 1
    ComplexMode = 2


class GptInitModelParameters:
    __slots__ = {
        "gpt_init_params",
        "_model_related_types",
        "has_lm_head_bias",
        "src_quantization_bit",
        "ptuning_path",
        "tp_split_emb_and_lm_head",
        "mm_related_params",
        "lora_infos",
        "multi_task_prompt",
        "normalize_lm_head_weight",
        "ref_module",
        "ref_dict",
        "tie_word_embeddings",
        "need_ffn_act_scale",
        "task_type",
        "add_special_tokens",
        "template_type",
        "build_position_ids",
        "routed_scaling_factor",
        "is_ft_style_weight",
        "vit_run_batch",
        "phy2log",
        "is_mtp",
        "num_nodes",
        "use_qk_norm",
        "enable_merge_w13"
    }

    # copy from maga_transformer/ops/libth_transformer.pyi for python intelligence
    activation_type: str
    add_bias_linear: bool
    block_nums: int
    cache_store_connect_port: int
    cache_store_listen_port: int
    cache_store_rdma_connect_port: int
    cache_store_rdma_listen_port: int
    cache_store_rdma_mode: bool
    ckpt_path: str
    cross_attn_input_len: int
    data_type: str
    decode_polling_kv_cache_step_ms: int
    decode_retry_timeout_ms: int
    decode_retry_times: int
    decode_use_async_load_cache: bool
    deepseek_mscale_all_dim: float
    deepseek_rope_mscale: float
    dp_rank: int
    dp_size: int
    dp_tp_nccl_port: int
    embedding_size: int
    enable_eplb: bool
    enable_fast_gen: bool
    enable_partial_fallback: bool
    enable_sp: bool
    enable_speculative_decoding: bool
    ep_rank: int
    ep_size: int
    eplb_mode: EplbMode
    eplb_update_time: int
    expert_num: int
    fast_gen_max_context_len: int
    ffn_tp_nccl_port: int
    ffn_tp_rank: int
    ffn_tp_size: int
    gen_num_per_circle: int
    has_lm_head: bool
    has_moe_norm: bool
    has_positional_encoding: bool
    has_post_decoder_layernorm: bool
    has_pre_decoder_layernorm: bool
    head_num: int
    head_num_kv: int
    hidden_size: int
    http_port: int
    include_sep_tokens: bool
    input_embedding_scalar: float
    input_vocab_size: int
    inter_padding_size: int
    inter_size: int
    is_causal: bool
    is_multimodal: bool
    is_sparse_head: bool
    kv_cache_data_type: str
    kv_cache_mem_mb: int
    kv_lora_rank: int
    layer_head_num: list[int]
    layer_head_num_kv: list[int]
    layer_inter_padding_size: list[int]
    layer_inter_size: list[int]
    layer_num: int
    layernorm_eps: float
    layernorm_type: str
    load_balance_policy_name: str
    load_cache_timeout_ms: int
    local_rank: int
    logit_scale: float
    max_context_batch_size: int
    max_generate_batch_size: int
    max_rpc_timeout_ms: int
    max_seq_len: int
    mla_ops_type: MlaOpsType
    mm_position_ids_style: int
    mm_sep_tokens: list[list[int]]
    model_name: str
    model_rpc_port: int
    moe_inter_padding_size: int
    moe_k: int
    moe_layer_index: list[int]
    moe_n_group: int
    moe_normalize_expert_scale: bool
    moe_style: int
    moe_topk_group: int
    mrope_section: list[int]
    nccl_ip: str
    nope_head_dim: int
    norm_type: str
    num_layers: int
    num_valid_layer: int
    org_embedding_max_pos: int
    pd_sep_enable_fallback: bool
    pd_separation: bool
    phy_exp_num: int
    position_id_len_factor: int
    position_ids_style: int
    pre_allocate_op_mem: bool
    pre_seq_len: int
    prefill_max_wait_timeout_ms: int
    prefill_retry_timeout_ms: int
    prefill_retry_times: int
    prefix_projection: bool
    py_eplb: typing.Any
    q_lora_rank: int
    q_scaling: float
    qk_norm: bool
    quant_algo: QuantAlgo
    rdma_connect_retry_times: int
    remote_rpc_server_port: int
    reserve_runtime_mem_mb: int
    residual_scalar: float
    reuse_cache: bool
    reverse_e_h_norm: bool
    rope_head_dim: int
    rotary_embedding_base: float
    rotary_embedding_dim: int
    rotary_embedding_mscale: float
    rotary_embedding_offset: int
    rotary_embedding_scale: float
    rotary_embedding_style: int
    rotary_factor1: float
    rotary_factor2: float
    scheduler_reserve_resource_ratio: int
    scoring_func: int
    seq_size_per_block: int
    size_per_head: int
    softmax_extra_scale: float
    special_tokens: SpecialTokens
    tokenizer_path: str
    tp_nccl_port: int
    num_nodes: int
    tp_rank: int
    tp_size: int
    type_vocab_size: int
    use_attention_linear_bias: bool
    use_cache_store: bool
    use_cross_attn: bool
    use_expert_attention: bool
    use_fp32_to_compute_logit: bool
    use_kvcache: bool
    use_logn_attn: bool
    use_mla: bool
    use_norm_attn_out_residual: bool
    use_norm_input_residual: bool
    using_hf_sampling: bool
    v_head_dim: int
    vit_separation: int
    vocab_size: int
    warm_up: bool
    warm_up_with_loss: bool
    worker_addrs: list[str]
    worker_grpc_addrs: list[str]
    worker_port_offset: int
    world_size: int

    def __init__(self,
                 head_num: int,
                 size_per_head: int,
                 layer_num: int,
                 max_seq_len: int,
                 vocab_size: int,
                 **kwargs: Any):
        hidden_size = head_num * size_per_head
        self.gpt_init_params = GptInitParameter(
            head_num, size_per_head, layer_num, max_seq_len, vocab_size, hidden_size
        )
        self._model_related_types: Dict[str, str] = {
            "layernorm_type": "setLayerNormType",
            "norm_type": "setNormType",
            "activation_type": "setActivationType",
            "kv_cache_data_type": "setKvCacheDataType"
        }
        self.has_lm_head_bias = False
        self.normalize_lm_head_weight = False
        self.src_quantization_bit = 0
        self.tp_split_emb_and_lm_head = True

        self.ptuning_path = None
        self.multi_task_prompt = None
        self.pre_seq_len = 0
        self.prefix_projection = False
        self.mm_related_params: VitParameters = VitParameters()
        self.ref_module: Optional[torch.nn.Module] = None
        self.ref_dict: Dict[str, torch.Tensor] = {}
        self.task_type = TaskType.LANGUAGE_MODEL

        self.tie_word_embeddings = False
        self.need_ffn_act_scale = False
        self.nccl_ip = g_master_info.ip
        self.tp_nccl_port = g_master_info.tp_nccl_port
        self.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
        self.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
        self.model_rpc_port = g_worker_info.rpc_server_port
        self.http_port = g_worker_info.http_port
        self.cache_store_listen_port = g_worker_info.cache_store_listen_port
        self.cache_store_connect_port = g_worker_info.cache_store_connect_port
        self.cache_store_rdma_listen_port = g_worker_info.cache_store_rdma_listen_port
        self.cache_store_rdma_connect_port = g_worker_info.cache_store_rdma_connect_port
        self.remote_rpc_server_port = g_worker_info.remote_rpc_server_port
        self.worker_port_offset = WORKER_INFO_PORT_NUM

        self.add_special_tokens = True
        self.template_type = TemplateType.chat
        self.build_position_ids = False
        self.routed_scaling_factor = 1.0
        self.vit_run_batch = False
        self.is_ft_style_weight = False

        self.is_multimodal = False
        self.model_name = ""

        self.world_size = g_parallel_info.world_size
        self.phy2log: List[List[int]] = []

        self.enable_eplb = self.eplb_mode != EplbMode.NONE
        
        self.is_mtp = False
        self.use_qk_norm = False
        self.enable_merge_w13 = False

        for k, v in kwargs.items():
            setattr(self, k, v)


    # read and write directly through GptInitModelParameters.k
    def __getattr__(self, k: str):
        return getattr(self.gpt_init_params, k)

    def __setattr__(self, k: str, v: Any):
        updated_params.add(k)
        if k in self.__slots__:
            object.__setattr__(self, k, v)
        elif v is not None:
            self.gpt_init_params.__setattr__(k, v)
            if k in self._model_related_types:
                getattr(self.gpt_init_params, self._model_related_types[k])()

    def update(self, update_params: Dict[str, Any]):
        for k, v in update_params.items():
            setattr(self, k, v)
        return self

    def update_worker_addrs(self):
        worker_addrs = []
        worker_grpc_addrs = []
        for member in get_gang_info().members:
            logging.info(f"member world rank: {member.world_rank}, member local rank: {member.local_rank}, local rank: {self.local_rank}, " \
                f"tp_size: {self.tp_size}, dp_size: {self.dp_size}, dp_rank: {self.dp_rank}")
            if int((member.world_rank / self.tp_size) % self.dp_size) == self.dp_rank:
                worker_addrs.append(f'{member.ip}:{member.cache_store_listen_port}:{member.cache_store_rdma_listen_port}')
                worker_grpc_addrs.append(f'{member.ip}:{member.rpc_server_port}')
                logging.info(f"append member for pd sep " \
                    f"{member.ip}:{member.rpc_server_port}, {member.cache_store_listen_port}, " \
                    f"{member.cache_store_rdma_listen_port} to local rank {self.local_rank}, world rank {member.world_rank}")
        self.worker_grpc_addrs = worker_grpc_addrs
        self.worker_addrs = worker_addrs

    def update_config_with_sparse_config(self, ckpt_path: str):
        sparse_config_file = None
        sparse_config = None
        if os.path.exists(os.path.join(ckpt_path, "config.json")):
            sparse_config_file = os.path.join(ckpt_path, "config.json")
        if os.environ.get('SPARSE_CONFIG_FILE', None) is not None:
            sparse_config_file = os.environ['SPARSE_CONFIG_FILE']

        if sparse_config_file is not None:
            logging.info(f"read sparse config from: {sparse_config_file}")
            with open(sparse_config_file, 'r') as reader:
                sparse_config_json = json.loads(reader.read())
                sparse_config = SparseConfig.from_dict(sparse_config_json)

        if sparse_config and sparse_config.check():
            self.layer_num = sparse_config.layer_num
            self.layer_head_num = sparse_config.layer_head_num
            self.layer_head_num_kv = sparse_config.layer_head_num
            self.layer_inter_size = sparse_config.layer_inter_size
            self.is_sparse_head = True

    def update_inter_padding_size(self, tp_size: int, ep_size: int, dp_size: int):
        if tp_size * dp_size != ep_size:
            raise ValueError(f"tp_size:{tp_size} * dp_size:{dp_size} != ep_size:{ep_size}")
        # new tp_size just only for moe
        if self.quant_algo.isGroupwise():
            align_size = tp_size * self.quant_algo.getGroupSize()
            moe_align_size = self.quant_algo.getGroupSize()
        else:
            align_size = tp_size * 64
            moe_align_size = 64
        if self.layer_inter_size:
            layer_inter_padding_size = []
            for idx in range(len(self.layer_inter_size)):
                inter_size = self.layer_inter_size[idx]
                layer_inter_padding_size.append(inter_size + (get_pad_size(inter_size, align_size) if self.quant_algo.isQuant() else 0))
            self.layer_inter_padding_size = layer_inter_padding_size
        self.inter_padding_size = \
            self.inter_size + (get_pad_size(self.inter_size, align_size) if self.quant_algo.isQuant() else 0)
        if self.head_num_kv <= 0:
            self.head_num_kv = self.head_num
        if self.inter_padding_size <= 0:
            self.inter_padding_size = self.inter_size

        if self.moe_inter_padding_size <= 0:
            self.moe_inter_padding_size = self.inter_size
        if self.moe_inter_padding_size > 0:
            moe_align_size = moe_align_size if self.quant_algo.isQuant() else  8
            self.moe_inter_padding_size = self.moe_inter_padding_size + (get_pad_size(self.moe_inter_padding_size, moe_align_size))

        logging.info(f"update_inter_padding_size: {self.inter_padding_size}, moe_inter_padding_size: {self.moe_inter_padding_size}, layer_inter_size: {self.layer_inter_size}")

    def update_task_prompt_tokens_id(self, tokenizer):
        if self.multi_task_prompt:
            for info in self.multi_task_prompt:
                task_id: str = str(info['task_id'])
                prompt: str = info['prompt']
                tokens_id = tokenizer.encode(prompt)
                self.insertMultiTaskPromptTokens(task_id, tokens_id)

    def update_task_prompt_config(self):
        prompt_file_path =  os.environ.get('MULTI_TASK_PROMPT', None)
        if not prompt_file_path:
            self.multi_task_prompt = None
        else:
            with open(prompt_file_path, 'r') as reader:
                multi_task_prompt = json.loads(reader.read(), strict=False)
                self.multi_task_prompt = multi_task_prompt
                return

        prompt_str =  os.environ.get('MULTI_TASK_PROMPT_STR', None)
        if not prompt_str:
            self.multi_task_prompt = None
        else:
            self.multi_task_prompt = json.loads(prompt_str, strict=False)
            return

    def update_task_type_use_kvcache(self):
        self.task_type = check_task_type(self.ckpt_path)
        self.setTaskType(self.task_type.value)
        self.use_kvcache = (self.task_type == TaskType.LANGUAGE_MODEL)
        logging.info(f"model task type: {self.task_type}, use_kvcache: {self.use_kvcache}")

    def update_weight_style(self, ckpt_path: str):
        if os.path.exists(os.path.join(ckpt_path, "model.safetensors.index.json")):
            meta_file = os.path.join(ckpt_path, "model.safetensors.index.json")
            logging.info(f"read weight style from: {meta_file}")
            with open(meta_file, 'r') as reader:
                meta_json = json.loads(reader.read())
                self.is_ft_style_weight = meta_json.get("is_ft_style_weight", False)

    def update_common(self,
                      ckpt_path: str,
                      lora_infos: Optional[Dict[str, str]],
                      ptuning_path: Optional[str],
                      tokenizer_path: str,
                      int8_mode: bool,
                      data_type: WEIGHT_TYPE,
                      max_seq_len: int,
                      seq_size_per_block: int,
                      gen_num_per_circle: int,
                      ref_module: Optional[torch.nn.Module] = None,
                      ref_dict: Dict[str, torch.Tensor] = {},
                      parallel_info: ParallelInfo=g_parallel_info,
                      config_mode: ConfigMode = ConfigMode.ComplexMode,
                      gang_info: Optional[GangInfo] = None):
        self.tp_size = parallel_info.tp_size
        self.tp_rank = parallel_info.tp_rank
        self.ep_size = parallel_info.ep_size
        self.ep_rank = parallel_info.ep_rank
        self.dp_size = parallel_info.dp_size
        self.dp_rank = parallel_info.dp_rank
        self.ffn_tp_rank = parallel_info.ffn_tp_rank
        self.ffn_tp_size = parallel_info.ffn_tp_size
        self.enable_sp = parallel_info.ffn_sp_size > 1
        self.local_rank = parallel_info.local_rank

        self.eplb_update_time = int(os.environ.get("EPLB_UPDATE_TIME", 5000))
        self.eplb_mode = EplbMode.__members__[os.environ.get('EPLB_MODE', 'NONE')]

        self.phy_exp_num = int(os.environ.get("REDUNDANT_EXPERT", 0)) + self.expert_num
        self.enable_merge_w13 = os.getenv('ENABLE_MERGE_W13', '0').lower() == '1'
        logging.info(f"phy_exp_num: {self.phy_exp_num}, use merge w13: {self.enable_merge_w13}")
        
        if gang_info is not None:
            self.num_nodes = gang_info.num_nodes
        else:
            try:
                self.num_nodes = get_gang_info().num_nodes
            except:
                self.num_nodes = 1
            

        self.ckpt_path = ckpt_path
        self.lora_infos = lora_infos
        self.tokenizer_path = tokenizer_path
        if not self.quant_algo.isQuant() and int8_mode:
            self.quant_algo.setQuantAlgo("weight_only_per_col", 8, 0)
        self.data_type = data_type.to_str()
        self.gen_num_per_circle = gen_num_per_circle
        self.ptuning_path = ptuning_path
        self.ref_module = ref_module
        self.ref_dict = ref_dict
        if max_seq_len != 0:
            self.max_seq_len = max_seq_len
        if self.max_seq_len < 1:
            self.max_seq_len = 1024
        logging.info(f'max_seq_len: {self.max_seq_len}')

        self.update_task_type_use_kvcache()

        logging.info(f"config_mode = {config_mode}")
        if config_mode == ConfigMode.SimpleMode:
            return

        self.update_worker_addrs()
        self.update_config_with_sparse_config(ckpt_path)
        self.update_inter_padding_size(self.tp_size, self.ep_size, self.dp_size)
        self.update_task_prompt_config()

        self.update_weight_style(ckpt_path)

        load_cutlass_gemm_config(self.quant_algo)

        hack_layer_num = int(os.environ.get('HACK_LAYER_NUM', 0))
        if (hack_layer_num):
            logging.info(f"hack layernum to {hack_layer_num}")
            self.layer_num = hack_layer_num

        self.seq_size_per_block = closest_power_of_2(int(max(seq_size_per_block, self.max_seq_len // 128))) # must be 2^n
        self.seq_size_per_block = int(os.environ.get('SEQ_SIZE_PER_BLOCK', self.seq_size_per_block))
        logging.info(f'seq_size_per_block: {self.seq_size_per_block}')
        self.max_generate_batch_size = int(os.environ.get('CONCURRENCY_LIMIT', 128))
        logging.info(f'max_generate_batch_size: {self.max_generate_batch_size}')
        self.max_context_batch_size = int(os.environ.get('MAX_CONTEXT_BATCH_SIZE', 1))
        logging.info(f'max_context_batch_size: {self.max_context_batch_size}')
        self.reserve_runtime_mem_mb = int(os.environ.get('RESERVER_RUNTIME_MEM_MB', 128))
        logging.info(f'reserve_runtime_mem_mb: {self.reserve_runtime_mem_mb}')
        self.kv_cache_mem_mb = int(os.environ.get('KV_CACHE_MEM_MB', -1))
        logging.info(f'kv_cache_mem_mb: {self.kv_cache_mem_mb}')
        self.block_nums = int(os.environ.get('TEST_BLOCK_NUM', 0))
        logging.info(f'block_nums: {self.block_nums}')
        if os.environ.get('TEST_LAYER_NUM'):
            logging.info(f'replace model layer with TEST_LAYER_NUM: {os.environ.get("TEST_LAYER_NUM")}')
            self.layer_num = int(os.environ.get('TEST_LAYER_NUM', self.layer_num))
        self.enable_partial_fallback = bool(int(os.environ.get('ENABLE_PARTIAL_FALLBACK', 0)))
        logging.info(f'enable_partial_fallback: {self.enable_partial_fallback}')
        self.enable_fast_gen = bool(int(os.environ.get('ENABLE_FAST_GEN', 0)))
        logging.info(f'enable_fast_gen: {self.enable_fast_gen}')
        self.warm_up = bool(int(os.environ.get('WARM_UP', 1)))
        logging.info(f'warm_up: {self.warm_up}')
        self.warm_up_with_loss = bool(int(os.environ.get('WARM_UP_WITH_LOSS', 0)))
        logging.info(f'warm_up_with_loss: {self.warm_up_with_loss}')

        self.vit_separation = int(os.environ.get('VIT_SEPARATION', 0))
        logging.info(f'vit_separation: {self.vit_separation}')

        self.fast_gen_max_context_len = int(os.environ.get('FAST_GEN_MAX_CONTEXT_LEN', 1024))
        logging.info(f'fast_gen_max_context_len: {self.fast_gen_max_context_len}')

        self.max_rpc_timeout_ms = int(os.environ.get('MAX_RPC_TIMEOUT_MS', 0))
        logging.info(f'max_rpc_timeout_ms: {self.max_rpc_timeout_ms}')

        self.pd_separation = bool(int(os.environ.get('PD_SEPARATION', 0)))
        logging.info(f'pd_separation: {self.pd_separation}')
        if self.pd_separation:
            self.prefill_retry_times = int(os.environ.get('PREFILL_RETRY_TIMES', 0))
            logging.info(f'prefill_retry_times: {self.prefill_retry_times}')
            self.prefill_retry_timeout_ms = int(os.environ.get('PREFILL_RETRY_TIMEOUT_MS', 0))
            logging.info(f'prefill_retry_timeout_ms: {self.prefill_retry_timeout_ms}')
            self.prefill_max_wait_timeout_ms = int(os.environ.get('PREFILL_MAX_WAIT_TIMEOUT_US', 600 * 1000 * 1000))
            logging.info(f'prefill_max_wait_timeout_ms: {self.prefill_max_wait_timeout_ms}')
            self.pd_sep_enable_fallback = bool(int(os.environ.get('PD_SEP_ENABLE_FALLBACK', 0)))
            logging.info(f'pd_sep_enable_fallback: {self.pd_sep_enable_fallback}')
            self.load_balance_policy_name = os.environ.get('LOAD_BALANCE_POLICY_NAME', "RR")
            logging.info(f'load_balance_policy_name: {self.load_balance_policy_name}')
            policy_list = ["RR", "WRR"]
            if not self.load_balance_policy_name in policy_list:
                raise Exception(f"load_balance_policy_name {self.load_balance_policy_name} " \
                    f"is not right, it must in {policy_list}")
            self.sync_status_interval_ms = int(os.environ.get('SYNC_STATUS_INTERVAL_MS', 50))
            logging.info(f'sync_status_interval_ms: {self.sync_status_interval_ms}')

        self.use_cache_store = bool(int(os.environ.get('USE_CACHE_STORE', 0)))
        logging.info(f'use_cache_store: {self.use_cache_store}')
        if self.use_cache_store:
            self.cache_store_rdma_mode = bool(int(os.environ.get('CACHE_STORE_RDMA_MODE', 1)))
            logging.info(f'cache_store_rdma_mode: {self.cache_store_rdma_mode}')

            self.load_cache_timeout_ms = int(os.environ.get('LOAD_CACHE_TIMEOUT_MS', 0))
            logging.info(f'load_cache_timeout_ms: {self.load_cache_timeout_ms}')

            self.decode_retry_times = int(os.environ.get('DECODE_RETRY_TIMES', 0))
            logging.info(f'decode_retry_times: {self.prefill_retry_times}')
            self.decode_retry_timeout_ms = int(os.environ.get('DECODE_RETRY_TIMEOUT_MS', 0))
            logging.info(f'decode_retry_timeout_ms: {self.decode_retry_timeout_ms}')

            self.rdma_connect_retry_times = int(os.environ.get('RDMA_CONNECT_RETRY_TIMES', 0))
            logging.info(f'rdma_connect_retry_times: {self.rdma_connect_retry_times}')

            self.decode_polling_kv_cache_step_ms = int(os.environ.get('DECODE_POLLING_KV_CACHE_STEP_MS', 30))
            logging.info(f'decode_polling_kv_cache_step_ms: {self.decode_polling_kv_cache_step_ms}')

            self.decode_use_async_load_cache = bool(int(os.environ.get('DECODE_USE_ASYNC_LOAD_CACHE', 1)))
            logging.info(f'decode_use_async_load_cache: {self.decode_use_async_load_cache}')

        self.scheduler_reserve_resource_ratio = int(os.environ.get('SCHEDUlER_RESERVE_RESOURCE_RATIO', 5))
        logging.info(f'scheduler_reserve_resource_ratio: {self.scheduler_reserve_resource_ratio}')
        self.reuse_cache = os.environ.get('REUSE_CACHE', None) == '1' or os.environ.get('USE_BLOCK_CACHE', None) == '1'
        logging.info(f'reuse_cache: {self.reuse_cache}')
        self.pre_allocate_op_mem = bool(int(os.environ.get('PRE_ALLOCATE_OP_MEM', 1)))
        logging.info(f'pre_allocate_op_mem: {self.pre_allocate_op_mem}')
        if bool(int(os.environ.get('INT8_KV_CACHE', 0))):
            self.kv_cache_data_type = WEIGHT_TYPE.INT8.to_str()
        elif self.quant_algo.isFp8() and not self.quant_algo.isGroupwise():
            self.kv_cache_data_type = WEIGHT_TYPE.FP8.to_str()
        else:
            self.kv_cache_data_type = self.data_type

        logging.info(f'kv_cache_data_type: {self.kv_cache_data_type}')
        logging.info(f'tp_split_emb_and_lm_head: {self.tp_split_emb_and_lm_head}')

        # use environment variables to update stop_words_str and stop_words_id
        env_stop_words_str = os.environ.get('STOP_WORDS_STR', None)
        env_stop_words_id = os.environ.get('STOP_WORDS_LIST', None)
        env_stop_words_str_list = json.loads(env_stop_words_str) if env_stop_words_str else []
        env_stop_words_id_list = json.loads(env_stop_words_id) if env_stop_words_id else []
        env_force_stop = os.environ.get('FORCE_STOP_WORDS', None)
        if env_force_stop and str_to_bool(env_force_stop):
            self.special_tokens.stop_words_str_list = env_stop_words_str_list
            self.special_tokens.stop_words_id_list = env_stop_words_id_list
        else:
            self.special_tokens.stop_words_str_list = self.special_tokens.stop_words_str_list + env_stop_words_str_list
            self.special_tokens.stop_words_id_list = self.special_tokens.stop_words_id_list + env_stop_words_id_list

        logging.info(f"use stop_words_str_list [{self.special_tokens.stop_words_str_list }]," \
                        f" stop_words_id_list [{self.special_tokens.stop_words_id_list}]")

    def get_params_dict(self):
        res: Dict[str, Any] = {}
        for name in updated_params:
            res[name] = eval('self.' + name)
        return res

    def eval_model_size(self):
        layer_param_bytes = 2
        if self.quant_algo.getWeightBits() == 8:
            layer_param_bytes = 1
        elif self.quant_algo.getWeightBits() == 4:
            layer_param_bytes = 0.54

        model_size = self.word_emb_param_count * 2 + \
            self.layer_weight_param_count * layer_param_bytes + \
                self.gpt_init_params.hidden_size * layer_param_bytes + \
                self.word_emb_param_count * 2  # maybe some model donot have lm_head

        kv_cache_mem_size = self._eval_kv_cache_mem_size()
        runtime_buffer = self._eval_runtime_buffer_mem_size()
        total_size = model_size  + kv_cache_mem_size + runtime_buffer
        logging.info(f"total_size(Bytes): {total_size}, model_size:{model_size}, kv_cache_mem_size:{kv_cache_mem_size}, runtime_buffer:{runtime_buffer}")
        return total_size

    def _eval_kv_cache_mem_size(self):
        if self.task_type != TaskType.LANGUAGE_MODEL:
            return 0
        kv_cache_bytes = 1 if self.kv_cache_data_type in [WEIGHT_TYPE.FP8.to_str(), WEIGHT_TYPE.INT8.to_str()] else 2
        kv_cache_size = 2 * self.layer_num * self.head_num_kv * self.size_per_head * kv_cache_bytes * self.max_seq_len
        return kv_cache_size

    def _eval_runtime_buffer_mem_size(self):
        input_buffer = self.max_seq_len * self.gpt_init_params.hidden_size
        qkv_gemm_buffer_size = self.max_seq_len * (self.head_num_kv*2 + self.head_num_kv) * self.size_per_head
        attn_buffer_size = self.max_seq_len * self.gpt_init_params.hidden_size
        ffn_export_num = self.expert_num if self.gpt_init_params.moe_k else 1
        ffn_w_count = 1 if self.activation_type == 'gelu' else 2
        ffn_buffer = (self.max_seq_len * self.gpt_init_params.hidden_size + ffn_w_count* self.max_seq_len * self.inter_size)*ffn_export_num
        return input_buffer + qkv_gemm_buffer_size + attn_buffer_size + ffn_buffer

    @property
    def model_param_count(self):
        return self.word_emb_param_count*2 + self.layer_weight_param_count + self.gpt_init_params.hidden_size

    @property
    def word_emb_param_count(self):
        return self.vocab_size * self.gpt_init_params.hidden_size

    @property
    def layer_weight_param_count(self):
        hidden_size = self.gpt_init_params.hidden_size

        layer_weight_param_count = 0
        # qkv
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = layer_weight_param_count + head_num * self.size_per_head * hidden_size *3
        elif self.head_num_kv != self.head_num:
            layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * hidden_size + \
                self.layer_num * (self.head_num_kv * self.size_per_head) * 2
        else:
            layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * hidden_size *3

        # attn_o_w
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = layer_weight_param_count + head_num * self.size_per_head * hidden_size
        else:
            layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * hidden_size

        # ffn w1, w2, w3
        ffn_export_num = self.expert_num if self.expert_num > 0 else 1
        ffn_w_count = 2 if self.activation_type == 'gelu' else 3
        if self.layer_inter_size and isinstance(self.layer_inter_size, list):
            for layer_inter_size in self.layer_inter_size:
                if self.moe_style == 1:
                    layer_weight_param_count = layer_weight_param_count + layer_inter_size * hidden_size * ffn_w_count * ffn_export_num
                else:
                    layer_weight_param_count = layer_weight_param_count + layer_inter_size * hidden_size * ffn_w_count
                    if self.moe_style == 2:
                        layer_weight_param_count = layer_weight_param_count + self.moe_inter_padding_size * hidden_size * ffn_w_count * ffn_export_num

        else:
            if self.moe_style == 1:
                layer_weight_param_count = layer_weight_param_count + self.layer_num * self.inter_size * hidden_size * ffn_w_count * ffn_export_num
            else:
                layer_weight_param_count = layer_weight_param_count + self.layer_num * self.inter_size * hidden_size * ffn_w_count
                if self.moe_style == 2:
                    layer_weight_param_count = layer_weight_param_count + len(self.moe_layer_index) * self.moe_inter_padding_size * hidden_size * ffn_w_count * ffn_export_num

        if ffn_export_num > 1:
            layer_weight_param_count = layer_weight_param_count + len(self.moe_layer_index) * hidden_size * ffn_export_num
        # other small tensor
        layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * 11
        return layer_weight_param_count
