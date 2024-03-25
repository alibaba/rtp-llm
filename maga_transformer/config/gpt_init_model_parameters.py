from typing import Dict, Any, List, Optional, Set
import os
import json
import torch
import logging
# make sure so init
from dataclasses import dataclass, field, fields
from enum import Enum
from maga_transformer.utils.util import WEIGHT_TYPE

updated_params: Set[str] = set()

def get_pad_size(size: int , align_size: int):
    return (align_size - (size % align_size)) % align_size

def str_to_bool(s: str):
    true_values = ('yes', 'true', '1')
    false_values = ('no', 'false', '0')
    if s.lower() in true_values:
        return True
    elif s.lower() in false_values:
        return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))

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
class MedusaConfig(DataClassBase):
    medusa_num_heads: int = 0
    medusa_num_layers: int = 0
    medusa_choices: List[List[int]] = field(default_factory=lambda: mc_sim_7b_63)
    top_k: int = 10
    posterior_threshold: float = 0.09
    posterior_alpha: float = 0.3

    def check(self) -> bool:
        if self.medusa_num_heads <= 0 or self.medusa_num_layers <= 0 or len(self.medusa_choices) <= 0:
            logging.info(f"medusa config error: {self}")
            return False
        return True

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
    vit_special_token_ids: Dict[str, Any] = {}
    vit_special_tokens: Dict[str, Any] = {}
    image_expand_token: Optional[int] = None
    vit_weights = None

class ModelType(Enum):
    NORMAL = "normal"
    EMBEDDING = "embedding"

class GptInitModelParameters:
    __slots__ = {
        "gpt_init_params",
        "_model_related_types",
        "has_lm_head_bias",
        "reserve_runtime_mem_mb",
        "kv_cache_mem_mb",
        "src_quantization_bit",
        "ptuning_path",
        "tp_split_emb_and_lm_head",
        "vit_related_params",
        "lora_infos",
        "multi_task_prompt",
        "medusa_config",
        "normalize_lm_head_weight",
        "ref_model",
        "is_quant_mode",
        "model_type"
    }

    def __init__(self,
                 head_num: int,
                 size_per_head: int,
                 layer_num: int,
                 max_seq_len: int,
                 vocab_size: int,
                 **kwargs):
        hidden_size = head_num * size_per_head
        self.gpt_init_params = torch.classes.FasterTransformer.GptInitParameter(
            head_num, size_per_head, layer_num, max_seq_len, vocab_size, hidden_size
        )
        self._model_related_types: Dict[str, str] = {
            "layernorm_type": "setLayerNormType",
            "norm_type": "setNormType",
            "activation_type": "setActivationType"
        }
        self.has_lm_head_bias = False
        self.normalize_lm_head_weight = False
        self.kv_cache_mem_mb = -1
        self.src_quantization_bit = 0
        self.tp_split_emb_and_lm_head = True
        self.medusa_config = None

        self.is_quant_mode = False
        self.ptuning_path = None
        self.pre_seq_len = 0
        self.prefix_projection = False
        self.vit_related_params: VitParameters = VitParameters()
        self.ref_model: Optional[torch.nn.Module] = None

        self.model_type = ModelType.NORMAL

        for k, v in kwargs.items():
            setattr(self, k, v)

    # read and write directly through GptInitModelParameters.k
    def __getattr__(self, k: str):
        return self.gpt_init_params.__getattr__(k)

    def __setattr__(self, k: str, v: Any):
        updated_params.add(k)
        if k in self.__slots__:
            object.__setattr__(self, k, v)
        elif v is not None:
            self.gpt_init_params.__setattr__(k, v)
            if k in self._model_related_types:
                self.gpt_init_params.__getattr__(self._model_related_types[k])()

    def update(self, update_params: Dict[str, Any]):
        for k, v in update_params.items():
            setattr(self, k, v)
        return self

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

    def update_medusa_config(self, ckpt_path: str):
        medusa_config_file = None
        medusa_config = None
        if os.path.exists(os.path.join(ckpt_path, "config.json")):
            medusa_config_file = os.path.join(ckpt_path, "config.json")
        if os.environ.get('MEDUSA_CONFIG_FILE', None) is not None:
            medusa_config_file = os.environ['MEDUSA_CONFIG_FILE']

        if medusa_config_file is not None:
            with open(medusa_config_file, 'r') as reader:
                medusa_config_json = json.loads(reader.read())
            if medusa_config_json.get('medusa_config', None) is not None:
                medusa_config = MedusaConfig.from_dict(medusa_config_json['medusa_config'])

        if medusa_config is not None and medusa_config.check():
            logging.info("use medusa config")
            self.medusa_config = medusa_config
            self.gpt_init_params.use_medusa = True

    def update_embedding_config(self, ckpt_path: str):
        def _check_is_sentence_transformer_repo() -> bool:
            if os.path.exists(os.path.join(ckpt_path, "config_sentence_transformers.json")):
                return True
            module_file_path = os.path.join(ckpt_path, "modules.json")
            if os.path.exists(module_file_path):
                with open(module_file_path, 'r') as reader:
                    content = reader.read()
                    if 'sentence_transformers' in content:
                        return True
            return False
        if os.environ.get('EMBEDDING_MODEL', '0') == '1' or _check_is_sentence_transformer_repo():
            self.model_type = ModelType.EMBEDDING

    def update_inter_padding_size(self, tp_size: int):
        align_size = tp_size * 64
        if self.layer_inter_size:
            layer_inter_padding_size = []
            for idx in range(len(self.layer_inter_size)):
                inter_size = self.layer_inter_size[idx]
                layer_inter_padding_size.append(inter_size + (get_pad_size(inter_size, align_size) if self.is_quant_mode else 0))
            self.layer_inter_padding_size = layer_inter_padding_size
        self.inter_padding_size = \
            self.inter_size + (get_pad_size(self.inter_size, align_size) if self.is_quant_mode else 0)
        if self.head_num_kv <= 0:
            self.head_num_kv = self.head_num
        if self.inter_padding_size <= 0:
            self.inter_padding_size = self.inter_size

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

    def update_ptuning_config(self):
        if not self.ptuning_path:
            inner_ptuing_path = os.path.join(self.ckpt_path, 'ptuning')
            if os.path.exists(inner_ptuing_path):
                logging.info(f"ckpt contain ptuning ckpt files, {inner_ptuing_path}")
                self.ptuning_path = inner_ptuing_path
        logging.info(f"use ptuning from model_config set by env, {self.ptuning_path}")
        if self.ptuning_path:
            config_file_path = os.path.join(self.ptuning_path, "config.json")
        else:
            config_file_path = os.path.join(self.ckpt_path, "config.json")
        if not os.path.exists(config_file_path):
            return
        logging.info(f"load ptuing config from {config_file_path}")
        with open(config_file_path, 'r') as reader:
            content = json.load(reader)
            if 'pre_seq_len' in content:
                self.pre_seq_len = content['pre_seq_len']
            if 'prefix_projection' in content:
                self.prefix_projection = content['prefix_projection']
        logging.info(f"read ptuning config, pre_seq_len:{self.pre_seq_len}, prefix_projection:{self.prefix_projection}")

    def update_common(self,
                      ckpt_path: str,
                      lora_infos: Optional[Dict[str, str]],
                      ptuning_path: Optional[str],
                      tokenizer_path: str,
                      int8_mode: bool,
                      data_type: WEIGHT_TYPE,
                      max_seq_len: int,
                      seq_size_per_block: int,
                      tp_size: int,
                      gen_num_per_circle: int,
                      ref_model: Optional[torch.nn.Module]):
        self.ckpt_path = ckpt_path
        self.lora_infos = lora_infos
        self.tokenizer_path = tokenizer_path
        self.quant_algo.int8_mode = int8_mode
        self.is_quant_mode = int8_mode or self.quant_algo.int4_mode
        self.data_type = data_type.to_str()
        self.gen_num_per_circle = gen_num_per_circle
        self.ptuning_path = ptuning_path
        self.ref_model = ref_model
        if max_seq_len != 0:
            self.max_seq_len = max_seq_len
        if self.max_seq_len < 1:
            self.max_seq_len = 1024
        logging.info(f'max_seq_len: {self.max_seq_len}')

        self.update_config_with_sparse_config(ckpt_path)
        self.update_inter_padding_size(tp_size)
        self.update_task_prompt_config()
        self.update_ptuning_config()
        self.update_medusa_config(ckpt_path)
        self.update_embedding_config(ckpt_path)

        self.seq_size_per_block = seq_size_per_block
        logging.info(f'seq_size_per_block: {self.seq_size_per_block}')
        self.max_generate_batch_size = int(os.environ.get('CONCURRENCY_LIMIT', 128))
        logging.info(f'max_generate_batch_size: {self.max_generate_batch_size}')
        self.max_context_batch_size = int(os.environ.get('MAX_CONTEXT_BATCH_SIZE', 1))
        logging.info(f'max_context_batch_size: {self.max_context_batch_size}')
        self.reserve_runtime_mem_mb = int(os.environ.get('RESERVER_RUNTIME_MEM_MB', 1 * 1024))
        logging.info(f'reserve_runtime_mem_mb: {self.reserve_runtime_mem_mb}')
        self.kv_cache_mem_mb = int(os.environ.get('KV_CACHE_MEM_MB', -1))
        logging.info(f'kv_cache_mem_mb: {self.kv_cache_mem_mb}')
        self.pre_allocate_op_mem = bool(int(os.environ.get('PRE_ALLOCATE_OP_MEM', 1)))
        logging.info(f'pre_allocate_op_mem: {self.pre_allocate_op_mem}')
        self.int8_kv_cache = bool(int(os.environ.get('INT8_KV_CACHE', 0)))
        logging.info(f'int8_kv_cache: {self.int8_kv_cache}')
        value = os.environ.get('TP_SPLIT_EMB_AND_LMHEAD')
        if value is not None:
            self.tp_split_emb_and_lm_head = str_to_bool(value)
        logging.info(f'tp_split_emb_and_lm_head: {self.tp_split_emb_and_lm_head}')

        # Update stop_words_str and stop_word_ids from ENV
        if os.environ.get('STOP_WORDS_STR', None) is not None:
            self.special_tokens.stop_words_str = self.special_tokens.stop_words_str + json.loads(os.environ['STOP_WORDS_STR'])
        elif os.environ.get('FORCE_STOP_WORDS_STR', None):
            self.special_tokens.stop_words_str = json.loads(os.environ['FORCE_STOP_WORDS_STR'])

        if os.environ.get('STOP_WORDS_LIST', None) is not None:
            self.special_tokens.stop_words_list = self.special_tokens.stop_words_list + json.loads(os.environ['STOP_WORDS_LIST'])
        elif os.environ.get('FORCE_STOP_WORDS_LIST', None):
            self.special_tokens.stop_words_list = json.loads(os.environ['FORCE_STOP_WORDS_LIST'])

    def get_params_dict(self):
        res: Dict[str, Any] = {}
        for name in updated_params:
            res[name] = eval('self.' + name)
        return res

    def eval_model_size(self):
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
                layer_weight_param_count = layer_weight_param_count + layer_inter_size * hidden_size * ffn_w_count * ffn_export_num

        else:
            layer_weight_param_count = layer_weight_param_count + self.layer_num * self.inter_size * hidden_size * ffn_w_count * ffn_export_num

        if ffn_export_num > 1:
            layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * ffn_export_num

        # other small tensor
        layer_weight_param_count = layer_weight_param_count + self.layer_num * hidden_size * 11

        word_emb_param_count =  self.vocab_size * hidden_size
        layer_param_bytes = 2
        if self.quant_algo.int8_mode:
            layer_param_bytes = 1
        elif self.quant_algo.int4_mode:
            layer_param_bytes = 0.54

        model_size = word_emb_param_count * 2 + \
            layer_weight_param_count * layer_param_bytes + \
                hidden_size * layer_param_bytes + \
                word_emb_param_count * 2  # maybe some model donot have lm_head

        return model_size
