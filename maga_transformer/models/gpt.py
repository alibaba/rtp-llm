import os
import json
import math
import torch
import logging
import gc
from typing import Optional, Union, List, Dict
from torch.nn.utils.init import skip_init
import torch.nn.functional as F

from maga_transformer.utils.util import get_device, to_torch_dtype
from maga_transformer.models.gpt_util.rms import RMSNorm
from maga_transformer.utils.model_weights_loader import get_model_weights_loader, estimate_load_parallel_num
from maga_transformer.utils.model_weight import W, ModelDeployWeightInfo, LoRAModelWeightInfo, LoRAMap
from maga_transformer.utils.time_util import Timer
from maga_transformer.utils.model_weight import LoraResource
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.database import CkptDatabase, ModuleDatabase
from maga_transformer.models.gpt_util.prefix_encoder import PrefixEncoder
from maga_transformer.models.gpt_util.medusa_head import MedusaHead
from maga_transformer.models.base_model import BaseModel
from maga_transformer.distribute.worker_info import g_parallel_info
from transformers import AutoTokenizer

def all_gather(output):
    tensor_list = [torch.empty_like(output) for _ in range(g_parallel_info.tp_size)]
    tensor_list[g_parallel_info.tp_rank] = output
    torch.distributed.all_gather(tensor_list, output)
    output = torch.cat(tensor_list, dim=output.dim() - 1).contiguous()
    return output

class Embedding(torch.nn.Module):
    def __init__(self, all_gather):
        super().__init__()
        self._emb = None
        self._scalar: Optional[float] = None
        self._all_gather = all_gather

    def set_weight(self, emb):
        self._emb = emb

    def set_scalar(self, scalar):
        self._scalar = scalar

    def forward(self, input):
        output = F.embedding(input, self._emb)
        if self._scalar:
            output = output * self._scalar
        if self._all_gather:
            return all_gather(output)
        return output

class Linear(torch.nn.Module):
    def __init__(self, all_gather):
        super().__init__()
        self._w = None
        self._b = None
        self._all_gather = all_gather

    def set_weight(self, w, b):
        self._w = w
        self._b = b

    def forward(self, input):
        output = F.linear(input, self._w, self._b)
        if self._all_gather:
            return all_gather(output)
        return output

class GPT(BaseModel):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()

        # 兼容逻辑
        if os.environ.get('USE_BLOCK_CACHE') is not None:
            os.environ["REUSE_CACHE"] = os.environ.get('USE_BLOCK_CACHE')

        self.config = config
        compute_dtype = to_torch_dtype(self.config.data_type)

        if self.config.use_attention_linear_bias:
            slopes = torch.Tensor(self.get_slopes(self.config.head_num))
            slopes = self.split_slopes_tp(slopes)
            self.linear_bias_slopes = slopes.to(to_torch_dtype(compute_dtype)).cuda()
        # torch.classes.load_library(os.path.abspath(lib_path)) # type: ignore

        # Embeddings to encode or decode tokens.
        hidden_dim = self.config.gpt_init_params.hidden_size
        all_gather = self.config.tp_split_emb_and_lm_head and g_parallel_info.tp_size > 1
        assert(hidden_dim != 0)


        self.prefix_encoder = None
        if config.pre_seq_len is not None and config.pre_seq_len > 0:
            self.prefix_tokens = torch.arange(config.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)

        self.medusa_head: Optional[torch.nn.Module] = None
        if config.gpt_init_params.use_medusa:
            self.medusa_head = MedusaHead(config)

        if g_parallel_info.is_pp_first:
            self.word_embedding = Embedding(all_gather)
            if self.config.has_positional_encoding:
                self.position_encoding =torch.nn.Embedding(self.config.max_seq_len, hidden_dim, dtype=compute_dtype, device='cuda:0')
            else:
                self.position_encoding = None

            if self.config.type_vocab_size > 0:
                self.token_type_embeddings = torch.nn.Embedding(self.config.type_vocab_size, hidden_dim)
            else:
                self.token_type_embeddings = None

            if self.config.has_pre_decoder_layernorm:
                if self.config.norm_type == 'layernorm' or self.config.norm_type == 'alphanorm':
                    self.pre_decoder_layernorm = torch.nn.LayerNorm(hidden_dim, eps=self.config.layernorm_eps, dtype=compute_dtype).to('cuda:0')
                elif self.config.norm_type == 'rmsnorm':

                    self.pre_decoder_layernorm = RMSNorm(hidden_dim, eps=self.config.layernorm_eps, use_bias=True).to('cuda:0')
            else:
                self.pre_decoder_layernorm = None

        if g_parallel_info.is_pp_last:
            if self.config.has_post_decoder_layernorm:
                if self.config.norm_type == 'layernorm' or self.config.norm_type == 'alphanorm':
                    self.post_decoder_layernorm = torch.nn.LayerNorm(hidden_dim, eps=self.config.layernorm_eps, dtype=compute_dtype).to('cuda:0')
                elif self.config.norm_type == 'rmsnorm':
                    self.post_decoder_layernorm = RMSNorm(hidden_dim, eps=self.config.layernorm_eps, use_bias=True).to('cuda:0')
            else:
                self.post_decoder_layernorm = None
            if self.config.has_lm_head:
                self.lm_head = Linear(all_gather)
            else:
                self.lm_head = None

        self.load_tokenizer()
        self.load()

    def split_slopes_tp(self, slopes: torch.Tensor):
        local_head_num = 1 if self.config.head_num == 1 else self.config.head_num // g_parallel_info.tp_size
        start_pos = local_head_num * g_parallel_info.tp_rank
        return slopes[start_pos: start_pos + local_head_num]

    @staticmethod
    def get_context_decoder_cls(config: GptInitModelParameters):
        return GptContextDecoder.from_config(config)

    @staticmethod
    def get_decoder_cls(config: GptInitModelParameters):
        return GptDecoder.from_config(config)

    @staticmethod
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
                GPT.get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

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

    @staticmethod
    def get_weight_cls() -> ModelDeployWeightInfo:
        raise NotImplementedError

    def update(self, lora_infos: Dict[str, str]):
        with Timer() as timer:
            self.weight.lora_resource.update(lora_infos)
        logging.info(f'update lora weights time: {timer.cost_ms() / 1000 :.2f} s')

    def load(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        self._load_weights(self.config.ref_model, device)
        self._initialize_from_weight(device)

    def _load_weights(self, ref_model: Optional[torch.nn.Module] = None, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        device = device or get_device()
        compute_dtype = to_torch_dtype(self.config.data_type or self.dtype)


        with Timer() as timer:
            weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
            if ref_model is not None:
                database = ModuleDatabase(ref_model)
            else:
                database = CkptDatabase(self.config.ckpt_path)
                database.load_ptuning(self.config.ptuning_path)
            if self.config.lora_infos is not None and len(self.config.lora_infos) == 1:
                for name, path in self.config.lora_infos.items():
                    database.load_lora(name, path)
            load_parallel_num = estimate_load_parallel_num(
                self.config, g_parallel_info.tp_size)
            model_weights_loader = get_model_weights_loader(weights_info, database, compute_dtype=compute_dtype)
            self.weight = model_weights_loader.load_weights_from_scratch(num_process=load_parallel_num)
            model_weights_loader.show_warns()

            self.weight.lora_resource = LoraResource({}, database, weights_info, LoRAMap())
            self.weight.lora_resource.model_weights_loader = model_weights_loader
            if self.config.lora_infos is not None and len(self.config.lora_infos) > 1:
                self.update(self.config.lora_infos)

        logging.info(f'load weights time: {timer.cost_ms() / 1000 :.2f} s')

    def _initialize_from_weight(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        compute_dtype = to_torch_dtype(self.config.data_type or self.dtype)
        
        assert self.word_embedding is not None
        self.word_embedding.set_weight(self.weight.steal_pytorch_weight(W.embedding))
        if (self.config.input_embedding_scalar - 1 > 1e-6):
            self.word_embedding.set_scalar(self.config.input_embedding_scalar)
        if self.lm_head is not None:
            if self.weight.has_pytorch_weight(W.lm_head):
                lm_head_w = self.weight.steal_pytorch_weight(W.lm_head)
            else:
                lm_head_w = self.word_embedding._emb
            if self.config.normalize_lm_head_weight:
                self.lm_head.set_weight(F.normalize(lm_head_w), self.weight.steal_pytorch_weight(W.lm_head_b))
            else:
                self.lm_head.set_weight(lm_head_w, self.weight.steal_pytorch_weight(W.lm_head_b))
        if self.lm_head is not None:
            if self.config.tp_split_emb_and_lm_head:
                self.vocab_size_padded = self.lm_head._w.shape[0] * g_parallel_info.tp_size
            else:
                self.vocab_size_padded = self.lm_head._w.shape[0]

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str):
            # np_w is 1-D array since a bin file doesn't have shape info.
            print(f"load {fname} to {param.data.shape}")
            param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to('cuda:0')

        def _safe_load_prefix_encoder_weight_from_module(param: torch.nn.Parameter, fname: str, ctype: torch.dtype):
            # np_w is 1-D array since a bin file doesn't have shape info.
            param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to(ctype).to('cuda:0')

        def _safe_load_medusa_head_weight_from_module(module: torch.nn.Module, ctype: torch.dtype):
            named_parameters = {k: v for k,v in module.named_parameters()}
            for key in named_parameters.keys():
                named_parameters[key].data = self.weight.steal_pytorch_weight(key).reshape(named_parameters[key].data.shape).to(ctype).to('cuda:0')

        #TODO@miji check tp
        if self.prefix_encoder is not None:
            _safe_load_prefix_encoder_weight_from_module(
                self.prefix_encoder.embedding.weight,
                W.prefix_w,
                compute_dtype)
            if self.prefix_encoder.prefix_projection:
                raise Exception("not implement prefix_projection yet")

        if self.medusa_head is not None:
            _safe_load_medusa_head_weight_from_module(
                self.medusa_head,
                compute_dtype)

        # pylint:disable=line-too-long
        if g_parallel_info.is_pp_first:
            if self.is_multimodal():
                self.load_vit_weight(compute_dtype)
            if self.position_encoding is not None:
                self.position_encoding.weight.data = \
                    (self.weight.steal_pytorch_weight(W.positional_embedding))[:self.config.max_seq_len].reshape(self.position_encoding.weight.data.shape).to('cuda:0')
            if self.token_type_embeddings is not None:
                self.token_type_embeddings.weight.data = \
                    (self.weight.steal_pytorch_weight(W.token_type_embedding)).reshape(self.token_type_embeddings.weight.data.shape).to('cuda:0')
            if self.pre_decoder_layernorm is not None:
                _safe_load_from_module(self.pre_decoder_layernorm.weight, W.pre_decoder_ln_gamma)
                _safe_load_from_module(self.pre_decoder_layernorm.bias, W.pre_decoder_ln_beta)
        if g_parallel_info.is_pp_last:
            if self.post_decoder_layernorm is not None:
                _safe_load_from_module(self.post_decoder_layernorm.weight, W.final_ln_gamma)
                _safe_load_from_module(self.post_decoder_layernorm.bias, W.final_ln_beta)

        torch.cuda.empty_cache()

    def update_pre_seq_len(self, config: GptInitModelParameters) -> None:
        config_json_path = os.path.join(config.ckpt_path, "config.json")
        if not os.path.exists(config_json_path):
            return
        with open(config_json_path, 'r') as reader:
            config_json = json.loads(reader.read())
        config.pre_seq_len = config_json.get('pre_seq_len', 0)
        config.prefix_projection = config_json.get('prefix_projection', False)
