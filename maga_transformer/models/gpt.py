import os
import json
import math
import torch
import logging
from typing import Optional, Union, List, Dict, Any
import torch.nn.functional as F

from maga_transformer.utils.util import get_device, to_torch_dtype
from maga_transformer.models.gpt_util.rms import RMSNorm
from maga_transformer.ops.comm.parallel_op import ParallelEmbedding, ParallelLinear
from maga_transformer.utils.model_weights_loader import get_model_weights_loader, estimate_load_parallel_num, ModelWeightsLoader
from maga_transformer.utils.model_weight import W, ModelDeployWeightInfo, LoRAModelWeightInfo, LoRAMap
from maga_transformer.utils.time_util import Timer
from maga_transformer.utils.model_weight import LoraResource
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.task_type import TaskType
from maga_transformer.models.downstream_modules.utils import create_custom_module
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.utils.database import CkptDatabase, ModuleDatabase, DictDatabase
from maga_transformer.models.gpt_util.prefix_encoder import PrefixEncoder
from maga_transformer.models.gpt_util.medusa_head import MedusaHead
from maga_transformer.models.base_model import BaseModel
from maga_transformer.distribute.worker_info import g_parallel_info
from transformers import AutoTokenizer

def get_slopes(n: int) -> List[float]:
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))

        return get_slopes_power_of_2(closest_power_of_2) + \
            get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

class GPT(BaseModel):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()

        self.config = config
        self.load_tokenizer()
        self.init_misc()
        self.init_linear_bias()
        self.init_prefix_encoder()
        self.init_medusa()
        self.init_pipeline_param()
        self.load()
    
    def init_misc(self):
        self.task_type = self.config.task_type
        self.custom_module = self.load_custom_module()
        self.compute_dtype = to_torch_dtype(self.config.data_type)
                    
    def split_slopes_tp(self, slopes: torch.Tensor):
        local_head_num = 1 if self.config.head_num == 1 else self.config.head_num // g_parallel_info.tp_size
        start_pos = local_head_num * g_parallel_info.tp_rank
        return slopes[start_pos: start_pos + local_head_num]

    def init_linear_bias(self):
        if self.config.use_attention_linear_bias:
            slopes = torch.Tensor(get_slopes(self.config.head_num))
            slopes = self.split_slopes_tp(slopes)
            self.linear_bias_slopes = slopes.to(self.compute_dtype).cuda()

    def init_prefix_encoder(self):
        self.prefix_encoder = None
        if self.config.pre_seq_len is not None and self.config.pre_seq_len > 0:
            self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(self.config)

    def init_medusa(self):
        self.medusa_head: Optional[torch.nn.Module] = None
        if self.config.gpt_init_params.use_medusa:
            self.medusa_head = MedusaHead(self.config)

    def init_pipeline_param(self):
        # Embeddings to encode or decode tokens.
        hidden_dim = self.config.gpt_init_params.hidden_size
        assert(hidden_dim != 0)
        all_gather = self.config.tp_split_emb_and_lm_head and g_parallel_info.tp_size > 1
        
        if g_parallel_info.is_pp_first:
            self.word_embedding = ParallelEmbedding(all_gather)
            self.position_encoding = ParallelEmbedding(all_gather) if self.config.has_positional_encoding else None
            self.token_type_embeddings = ParallelEmbedding(all_gather) if self.config.type_vocab_size else None

            if self.config.has_pre_decoder_layernorm:
                if self.config.norm_type == 'layernorm' or self.config.norm_type == 'alphanorm':
                    self.pre_decoder_layernorm = torch.nn.LayerNorm(hidden_dim, eps=self.config.layernorm_eps, dtype=self.compute_dtype).to('cuda:0')
                elif self.config.norm_type == 'rmsnorm':

                    self.pre_decoder_layernorm = RMSNorm(hidden_dim, eps=self.config.layernorm_eps, use_bias=True).to('cuda:0')
            else:
                self.pre_decoder_layernorm = None

        if g_parallel_info.is_pp_last:
            if self.config.has_post_decoder_layernorm:
                if self.config.norm_type == 'layernorm' or self.config.norm_type == 'alphanorm':
                    self.post_decoder_layernorm = torch.nn.LayerNorm(hidden_dim, eps=self.config.layernorm_eps, dtype=self.compute_dtype).to('cuda:0')
                elif self.config.norm_type == 'rmsnorm':
                    self.post_decoder_layernorm = RMSNorm(hidden_dim, eps=self.config.layernorm_eps, use_bias=True).to('cuda:0')
            else:
                self.post_decoder_layernorm = None
            if self.task_type == TaskType.LANGUAGE_MODEL:
                self.lm_head = ParallelLinear(all_gather)

    @classmethod
    def from_config(cls, config: GptInitModelParameters):
        return cls(config)

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        assert config.tokenizer_path
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    def load_tokenizer(self):
        if self.config.tokenizer_path:
            self.tokenizer = self.get_tokenizer(self.config)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
                self.config.special_tokens.eos_token_id = self.tokenizer.eos_token_id
            self.config.update_task_prompt_tokens_id(self.tokenizer)

    @staticmethod
    def get_weight_cls() -> ModelDeployWeightInfo:
        raise NotImplementedError

    def update(self, lora_infos: Dict[str, str]):
        with Timer() as timer:
            self.weight.lora_resource.update(lora_infos)
        logging.info(f'update lora weights time: {timer.cost_ms() / 1000 :.2f} s')

    def load(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        self._load_weights(device)
        self._initialize_from_weight(device)

    def load_custom_module(self) -> Optional[CustomModule]:        
        return create_custom_module(self.task_type, self.config, self.tokenizer)
    
    def _load_custom_module_weights(self, model_weights_loader: ModelWeightsLoader):
        if self.custom_module is not None:
            tensor_names = self.custom_module.handler.tensor_info()
            tensor_map: Dict[str, torch.Tensor] = {}
            for name in tensor_names:
                tensors = model_weights_loader.load_tensor(name)
                if len(tensors) !=1 :
                    raise Exception(f"load tensor {name} failed, get len=={len(tensors)}")
                loaded_tensor = tensors[0].cuda()
                tensor_map[name] = loaded_tensor
                self.weight.append_global_weight(name, loaded_tensor)
            self.custom_module.handler.init(tensor_map)
            
    def init_database(self):
        if self.config.ref_module is not None:
            self.database = ModuleDatabase(self.config.ref_module)
        elif len(self.config.ref_dict) != 0:
            self.database = DictDatabase(self.config.ref_dict)
        else:
            self.database = CkptDatabase(self.config.ckpt_path, self.config.ptuning_path)

    def load_static_lora(self):
        # static lora load
        self.static_lora: bool = self.config.lora_infos is not None and len(self.config.lora_infos) == 1
        if self.static_lora:
            for name, path in self.config.lora_infos.items():           
                self.database.load_lora(name, path)
            self.database.dump_lora_info()

    def load_model_weight(self):
        load_parallel_num = estimate_load_parallel_num(
            self.config, g_parallel_info.tp_size)
        weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
        model_weights_loader = get_model_weights_loader(weights_info, self.database, compute_dtype=self.compute_dtype)
        self.weight = model_weights_loader.load_weights_from_scratch(num_process=load_parallel_num)
        self._load_custom_module_weights(model_weights_loader)
        if self.static_lora:
            lora_name = list(self.config.lora_infos.keys())[0]
            model_weights_loader.show_warns(lora_name=lora_name)
        else:
            model_weights_loader.show_warns()

        self.weight.lora_resource = LoraResource({}, self.database, weights_info, LoRAMap())
        self.weight.lora_resource.model_weights_loader = model_weights_loader
        if self.config.lora_infos is not None and len(self.config.lora_infos) > 1:
            self.update(self.config.lora_infos)

    def _load_weights(self, 
                      ref_dict: Dict[str, torch.Tensor] = {},
                      device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        device = device or get_device()
        with Timer() as timer:
            self.init_database()
            self.load_static_lora()
            self.load_model_weight()
        logging.info(f'load weights time: {timer.cost_ms() / 1000 :.2f} s')

    def init_word_embedding_weight(self):
        assert self.word_embedding is not None
        self.word_embedding.set_weight(self.weight.steal_pytorch_weight(W.embedding))
        self.weight.append_global_weight(W.embedding, self.word_embedding._emb)        
        if (self.config.input_embedding_scalar - 1 > 1e-6):
            self.word_embedding.set_scalar(self.config.input_embedding_scalar)

    def init_lm_head_weight(self):
        if self.lm_head is not None:
            if self.weight.has_pytorch_weight(W.lm_head):
                lm_head_w = self.weight.steal_pytorch_weight(W.lm_head)
            else:
                lm_head_w = self.word_embedding.weight
            if self.config.normalize_lm_head_weight:
                lm_head_w = F.normalize(lm_head_w)
            if self.config.logit_scale != 1.0:
                lm_head_w = self.config.scale_logit * lm_head_w
            self.lm_head.set_weight(lm_head_w, self.weight.steal_pytorch_weight(W.lm_head_b))
            self.weight.append_global_weight(W.lm_head, self.lm_head._w)            

            if self.config.tp_split_emb_and_lm_head:
                self.vocab_size_padded = self.lm_head.weight.shape[0] * g_parallel_info.tp_size
            else:
                self.vocab_size_padded = self.lm_head.weight.shape[0]

    def _safe_load_from_module(self, param: torch.nn.Parameter, fname: str):
        # np_w is 1-D array since a bin file doesn't have shape info.
        print(f"load {fname} to {param.data.shape}")
        param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to('cuda:0')

    def _safe_load_prefix_encoder_weight_from_module(self, param: torch.nn.Parameter, fname: str, ctype: torch.dtype):
        # np_w is 1-D array since a bin file doesn't have shape info.
        param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to(ctype).to('cuda:0')

    def _safe_load_medusa_head_weight_from_module(self, module: torch.nn.Module, ctype: torch.dtype):
        named_parameters = {k: v for k,v in module.named_parameters()}
        for key in named_parameters.keys():
            named_parameters[key].data = self.weight.steal_pytorch_weight(key).reshape(named_parameters[key].data.shape).to(ctype).to('cuda:0')

    def init_prefix_encoder_weight(self):
        #TODO@miji check tp
        if self.prefix_encoder is not None:
            self._safe_load_prefix_encoder_weight_from_module(
                self.prefix_encoder.embedding.weight,
                W.prefix_w,
                self.compute_dtype)
            if self.prefix_encoder.prefix_projection:
                raise Exception("not implement prefix_projection yet")

    def init_medusa_weight(self):
        if self.medusa_head is not None:
            self._safe_load_medusa_head_weight_from_module(
                self.medusa_head,
                self.compute_dtype)

    def init_pipeline_weight(self):
        # pylint:disable=line-too-long
        if g_parallel_info.is_pp_first:
            if self.is_multimodal():
                self.load_vit_weight(self.compute_dtype)
            if self.position_encoding is not None:
                pos_weight = self.weight.steal_pytorch_weight(W.positional_embedding)
                assert pos_weight is not None, "positional embedding weight not found"
                pos_weight = pos_weight[:self.config.max_seq_len].cuda()
                self.position_encoding.set_weight(pos_weight)
                self.weight.append_global_weight(W.positional_embedding, self.position_encoding._emb)
            if self.token_type_embeddings is not None:
                token_type_weight = self.weight.steal_pytorch_weight(W.token_type_embedding)
                assert token_type_weight is not None, "token_type embedding weight not found"
                self.token_type_embeddings.set_weight(token_type_weight.cuda())
                self.weight.append_global_weight(W.token_type_embedding, self.token_type_embeddings._emb)                
            if self.pre_decoder_layernorm is not None:
                self._safe_load_from_module(self.pre_decoder_layernorm.weight, W.pre_decoder_ln_gamma)
                self._safe_load_from_module(self.pre_decoder_layernorm.bias, W.pre_decoder_ln_beta)
                self.weight.append_global_weight(W.pre_decoder_ln_gamma, self.pre_decoder_layernorm.weight.data)
                self.weight.append_global_weight(W.pre_decoder_ln_beta, self.pre_decoder_layernorm.bias.data)

        if g_parallel_info.is_pp_last:
            if self.post_decoder_layernorm is not None:
                self._safe_load_from_module(self.post_decoder_layernorm.weight, W.final_ln_gamma)
                self._safe_load_from_module(self.post_decoder_layernorm.bias, W.final_ln_beta)
                self.weight.append_global_weight("final_layernorm.gamma", self.post_decoder_layernorm.weight.data)
                self.weight.append_global_weight("final_layernorm.beta", self.post_decoder_layernorm.bias.data)

    def _initialize_from_weight(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        self.init_word_embedding_weight()
        self.init_lm_head_weight()
        self.init_prefix_encoder_weight()
        self.init_medusa_weight()
        self.init_pipeline_weight()
        torch.cuda.empty_cache()

    def update_pre_seq_len(self, config: GptInitModelParameters) -> None:
        config_json_path = os.path.join(config.ckpt_path, "config.json")
        if not os.path.exists(config_json_path):
            return
        with open(config_json_path, 'r') as reader:
            config_json = json.loads(reader.read())
        config.pre_seq_len = config_json.get('pre_seq_len', 0)
        config.prefix_projection = config_json.get('prefix_projection', False)

    @staticmethod
    def _load_quant_config(ckpt_path: str, config_json: Dict[str, Any], config: GptInitModelParameters):
        quant_config_path = os.path.join(ckpt_path, 'smoothquant.ini')
        if os.path.exists(quant_config_path):
            config.quant_algo.setQuantAlgo('smooth_quant', 0, 0)

        quant_config = config_json.get("quantization_config", None)
        if quant_config is not None:
            config.quant_algo.setQuantAlgo(quant_config['quant_method'], quant_config["bits"], quant_config.get("group_size", 0))
