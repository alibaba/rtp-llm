import functools
import gc
import logging
import os
import torch
from maga_transformer.utils.ckpt_file_info import CkptFileInfo
from typing import List, Tuple, Union, Optional, Dict, Any
from maga_transformer.utils.database import BaseDatabase, CkptDatabase
from maga_transformer.utils.model_weight import W, CkptWeightInfo, WeightStyle, choose_available, identity, tolerate_failed
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight, CompositeWeight
from maga_transformer.model_loader.ffn_weight import FfnConfig, FfnWeight, MoeWithSharedWeight
from maga_transformer.model_loader.attn_weight import AttnConfig

class ModelWeightInfo:
    layer_weights: Union[List[WeightModule], List[List[WeightModule]]]
    weights: List[WeightModule]

    def __init__(self, weights: List[WeightModule],
                 layer_weights: Union[List[WeightModule], List[List[WeightModule]]]) -> None:
        self.weights = weights
        self.layer_weights = layer_weights
        if len(self.layer_weights) == 0:
            return


    def set_weight_dtype(self, dtype: torch.dtype):
        if self.layer_weights:
            for weight in self.layer_weights:
                weight.data_type = dtype

    def get_layer_weight_info(self, layer_id: int, name: str) -> Optional[WeightModule]:
        from collections import deque
        queue = deque(self.layer_weights[layer_id])

        while queue:
            weight = queue.popleft()
            if weight.name == name:
                return weight
            if isinstance(weight, CompositeWeight):
                queue.extend(weight.sub_weights)

        return None


    def to_quant_weight_info(self, quant_algo: Any):
        if quant_algo is None or not quant_algo.isQuant():
            raise ValueError("quant_algo is None or not quant_algo.isQuant()")
        weights = []
        if self.weights:
            for weight in self.weights:
                weights.append(weight.create(weight, quant_algo))
        layer_weights: Union[List[WeightModule], List[List[WeightModule]]] = [] if self.layer_weights else None
        if self.layer_weights:
            for weight in self.layer_weights:
                if isinstance(weight, list):
                    layer_weight = []
                    for w in weight:
                        layer_weight.append(w.create(w, quant_algo))
                    layer_weights.append(layer_weight)
                else:
                    layer_weights.append(weight.create(weight, quant_algo))

        return ModelWeightInfo(weights, layer_weights)



class ModelDeployWeightInfo:

    TRT_ENGINE_LAYER_WEIGHT_MAP = {
        W.pre_ln_beta : 'transformer.layers.{i}.input_layernorm.bias',
        W.pre_ln_gamma : 'transformer.layers.{i}.input_layernorm.weight',
        W.attn_qkv_w : 'transformer.layers.{i}.attention.qkv.weight',
        W.attn_qkv_b : 'transformer.layers.{i}.attention.qkv.bias',
        W.attn_qkv_s : 'transformer.layers.{i}.attention.qkv.weights_scaling_factor',

        W.attn_o_w : 'transformer.layers.{i}.attention.dense.weight',
        W.attn_o_b : 'transformer.layers.{i}.attention.dense.bias',
        W.attn_o_s : 'transformer.layers.{i}.attention.dense.weights_scaling_factor',

        W.ffn_w3 : 'transformer.layers.{i}.mlp.fc.weight',
        W.ffn_b3 : 'transformer.layers.{i}.mlp.fc.bias',
        W.ffn_s3 : 'transformer.layers.{i}.mlp.fc.weights_scaling_factor',

        W.ffn_w2 : 'transformer.layers.{i}.mlp.proj.weight',
        W.ffn_b2 : 'transformer.layers.{i}.mlp.proj.bias',
        W.ffn_s2 : 'transformer.layers.{i}.mlp.proj.weights_scaling_factor',

        W.post_ln_gamma : 'transformer.layers.{i}.post_layernorm.weight',
        W.post_ln_beta : 'transformer.layers.{i}.post_layernorm.bias',

    }

    TRT_ENGINE_LAYER_WEIGHT_MAP2 = {
        W.pre_ln_beta : 'transformer.layers.{i}.input_layernorm.bias',
        W.pre_ln_gamma : 'transformer.layers.{i}.input_layernorm.weight',
        W.attn_qkv_w : 'transformer.layers.{i}.attention.qkv.weight',
        W.attn_qkv_b : 'transformer.layers.{i}.attention.qkv.bias',
        W.attn_qkv_s : 'transformer.layers.{i}.attention.qkv.weights_scaling_factor',

        W.attn_o_w : 'transformer.layers.{i}.attention.dense.weight',
        W.attn_o_b : 'transformer.layers.{i}.attention.dense.bias',
        W.attn_o_s : 'transformer.layers.{i}.attention.dense.weights_scaling_factor',

        W.ffn_w1 : 'transformer.layers.{i}.mlp.fc.weight',
        W.ffn_b1 : 'transformer.layers.{i}.mlp.fc.bias',
        W.ffn_s1 : 'transformer.layers.{i}.mlp.fc.weights_scaling_factor',

        W.ffn_w2 : 'transformer.layers.{i}.mlp.proj.weight',
        W.ffn_b2 : 'transformer.layers.{i}.mlp.proj.bias',
        W.ffn_s2 : 'transformer.layers.{i}.mlp.proj.weights_scaling_factor',

        W.ffn_w3 : 'transformer.layers.{i}.mlp.gate.weight',
        W.ffn_b3 : 'transformer.layers.{i}.mlp.gate.bias',
        W.ffn_s3 : 'transformer.layers.{i}.mlp.gate.weights_scaling_factor',

        W.post_ln_gamma : 'transformer.layers.{i}.post_layernorm.weight',
        W.post_ln_beta : 'transformer.layers.{i}.post_layernorm.bias',

    }
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        self.config = config
        self._use_qk_norm = config.use_qk_norm
        self._hidden_size = config.hidden_size
        self._inter_size = config.inter_size
        self._inter_padding_size = config.inter_padding_size
        self._moe_inter_padding_size = config.moe_inter_padding_size
        self._head_num = config.head_num
        self._head_num_kv = config.head_num_kv
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.num_nodes: int = config.num_nodes
        self.ffn_tp_rank = config.ffn_tp_rank
        self.ffn_tp_size = config.ffn_tp_size
        self._size_per_head = config.size_per_head
        self.phy2log_ = config.phy2log
        if self._head_num_kv == -1:
            self._head_num_kv = self._head_num
        self._quant_algo = config.quant_algo
        self._num_layers = config.num_layers
        self._layer_head_num = config.layer_head_num
        self._layer_inter_padding_size = config.layer_inter_padding_size
        self._has_prefix_encoder = False
        self._is_sparse_head = config.is_sparse_head
        self._layer_head_num = config.layer_head_num
        self._src_quantization_bit = config.src_quantization_bit
        self.tp_split_emb_and_lm_head = config.tp_split_emb_and_lm_head


        self._is_gated_activation = config.gpt_init_params.isGatedActivation()
        self.expert_num_ = config.gpt_init_params.expert_num
        self.moe_n_group_ = config.moe_n_group
        self.enable_eplb_ = config.enable_eplb
        self.phy_exp_num_ = config.phy_exp_num
        self.enable_merge_w13_ = config.enable_merge_w13
        self.moe_k_      = config.gpt_init_params.moe_k
        self.moe_layer_index_ = config.gpt_init_params.moe_layer_index
        self.moe_style_ = config.gpt_init_params.moe_style
        self._moe_inter_padding_size = config.moe_inter_padding_size

        self.tie_word_embeddings = config.tie_word_embeddings
        self.need_ffn_act_scale = config.need_ffn_act_scale
        self.use_expert_attention = config.use_expert_attention
        self.weight_style = WeightStyle.RTP_LLM_STYLE if config.is_ft_style_weight else WeightStyle.NONE


        # for mla
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.routed_scaling_factor = config.routed_scaling_factor
        self.is_ft_style_weight = config.is_ft_style_weight

        # for vit sep
        self.vit_separation = config.vit_separation

        # for eplb
        self.phy2log = config.phy2log
        # for moe
        self._use_stack_weight = False

    @property
    def support_lora(self):
        return False

    @property
    def attn_config(self):
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv
        )
        return attn_config

    @property
    def ffn_config(self):
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            inter_padding_size=self._inter_padding_size,
            is_moe=False
        )
        return ffn_config
            
    def get_weight_info(self) -> ModelWeightInfo:
        weight_info = self._get_weight_info()
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        if use_fp32:
            weight_info = weight_info.set_weight_dtype(torch.float32)

        if weight_info.layer_weights and not isinstance(weight_info.layer_weights[0], List):
            layer_weights = []
            for _ in range(self._num_layers):
                layer_weights.append(weight_info.layer_weights)
            weight_info.layer_weights = layer_weights

        if self.weight_style != WeightStyle.NONE:
            logging.info("fix weight style")
            weight_info = self._fix_weight_style_layer_weight(weight_info)
            
        if self.enable_merge_w13_:
            logging.info("fix merge_w13")
            weight_info = self._fix_merge_w1_w3(weight_info)
            
        if self._quant_algo is not None and self._quant_algo.isQuant():
            weight_info = weight_info.to_quant_weight_info(self._quant_algo)

        if self.tie_word_embeddings:
            logging.info("fix tie_word_embeddings")
            weight_info = self._fix_tie_lm_head(weight_info)
        if self._is_sparse_head:
            logging.info("Skiping load empty weight for head_num == 0")
            weight_info = self._process_sparse_weight(weight_info)

        return weight_info

    def _fix_weight_style_layer_weight(self, origin_weight_info: ModelWeightInfo):
        global_weights = []
        m1 = {
            W.embedding : 'transformer.vocab_embedding.weight',
            W.lm_head : 'lm_head.weight',
            W.final_ln_gamma : 'transformer.ln_f.weight'
        } if self.weight_style == WeightStyle.TRT_ENGINE else {}

        def __update_weight_style(weight: WeightModule, name_map: Dict[str, str]):
            if isinstance(weight , AtomicWeight):
                weight.weight_style = self.weight_style
                if weight.name in name_map:
                    if len(weight.weights) == 1:
                        weight.weights[0].name = name_map[weight.name]
                    elif weight.name in [W.attn_qkv_b, W.attn_qkv_w]:
                        weight.weights = [CkptWeightInfo(name_map[weight.name])]
                        weight.process_fun = identity
                        logging.error(f"{weight.name} have many weight, maybe cause bug {weight.weights}")
                    elif len(weight.weights) >= 2:
                        raise ValueError(f"{weight.name} should have only one or zero weight, {weight.weights}")
                    logging.info(f"update weight style for {weight.name}: {weight.weights[0].name}")
            elif isinstance(weight, CompositeWeight):
                weight.weight_style = self.weight_style
                for _, sub_weight in weight.sub_weights.items():
                    __update_weight_style(sub_weight, name_map)


        for _, weight in enumerate(origin_weight_info.weights):
            __update_weight_style(weight, m1)
            global_weights.append(weight)
        origin_weight_info.weights = global_weights

        layer_weights = []
        for weights in origin_weight_info.layer_weights:
            ffn_weight = [weight for weight in weights if weight.name == W.ffn]
            assert len(ffn_weight) == 1
            if ffn_weight[0].w1 is not None and self.weight_style == WeightStyle.TRT_ENGINE:
                m2 = self.TRT_ENGINE_LAYER_WEIGHT_MAP2
            elif self.weight_style == WeightStyle.TRT_ENGINE:
                m2 = self.TRT_ENGINE_LAYER_WEIGHT_MAP
            else:
                m2 = {}

            fix_weight = []
            for weight in weights:
                __update_weight_style(weight, m2)
                fix_weight.append(weight)
            layer_weights.append(fix_weight)

        origin_weight_info.layer_weights = layer_weights
        logging.info(f"fix weight style {origin_weight_info.layer_weights[0]}")
        return origin_weight_info

    def _fix_merge_w1_w3(self, origin_weight_info: ModelWeightInfo):
        def __update_weight_config(weight: WeightModule):
            if isinstance(weight , FfnWeight) or isinstance(weight, MoeWithSharedWeight):
                weight.config.enable_merge_w13 = True
                params = weight.extract_params(weight.__class__, weight, None)
                return weight.__class__(**params)
            else:
                return weight

        layer_weights = []
        for weights in origin_weight_info.layer_weights:
            fix_weight = []
            for weight in weights:
                fix_weight.append(__update_weight_config(weight))
            layer_weights.append(fix_weight)

        origin_weight_info.layer_weights = layer_weights
        logging.info(f"fix weight config when need_merge_w13 {origin_weight_info.layer_weights[0]}")
        return origin_weight_info

    def _fix_tie_lm_head(self, origin_weight_info: ModelWeightInfo) -> ModelWeightInfo:
        word_emb_idx = -1
        word_emb = None
        lm_head_idx = -1
        lm_head = None
        for idx, weight in enumerate(origin_weight_info.weights):
            if weight.name == W.embedding:
                word_emb_idx = idx
                word_emb = weight
            elif weight.name == W.lm_head:
                lm_head = weight
                lm_head_idx = idx
        if not lm_head or not word_emb:
            return origin_weight_info

        assert len(lm_head.weights) == 1 and len(word_emb.weights) == 1
        lm_head_ckpt_weigth_infos = [CkptWeightInfo(w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)) for w in lm_head.weights]
        lm_head_ckpt_weigth_infos.extend([CkptWeightInfo(w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)) for w in word_emb.weights])
        lm_head_merge_funcs = [lm_head.process_fun, word_emb.process_fun]
        lm_head = AtomicWeight(W.lm_head, lm_head_ckpt_weigth_infos, functools.partial(choose_available, origin_func_list = lm_head_merge_funcs))
        origin_weight_info.weights[lm_head_idx] = lm_head
        return origin_weight_info


    def _process_sparse_weight(self, origin_weight_info: ModelWeightInfo) -> ModelWeightInfo:
        if not isinstance(origin_weight_info.layer_weights[0], list):
            raise Exception("model weight use sparse config should be list(list())")
        new_layer_weights = []
        for i, layer_weight in enumerate(origin_weight_info.layer_weights):
            if self._layer_head_num[i] == 0:
                new_weights = [weight for weight in layer_weight if weight.name not in W.skip_weights_list]
            else:
                new_weights = layer_weight
            new_layer_weights.append(new_weights)
        return ModelWeightInfo(origin_weight_info.weights, new_layer_weights)

    def _get_weight_info(self) -> ModelWeightInfo:
        raise NotImplementedError()

    def create_model_weight_info(self, database: BaseDatabase) -> ModelWeightInfo:
        if isinstance(database, CkptDatabase):
            self.process_meta_from_ckpt(database.PretrainFileList)
            self.process_meta_from_ckpt(database.FinetuneFileList)
            if not self.is_ft_style_weight:
                return self.get_weight_info()
        else:
            raise Exception("Unknown database class")

    def process_meta_from_ckpt(self, ckpt_metas: List[CkptFileInfo]):
        if len(ckpt_metas) == 0:
            return
        if not self.is_ft_style_weight:
            # call subclass process_meta
            meta_dicts = [ckpt_file.get_metadata() for ckpt_file in ckpt_metas]
            weight_keys = set(functools.reduce(lambda x,y:x+y, [list(meta.keys()) for meta in meta_dicts], []))
            self._process_meta(meta_dicts, weight_keys)

    def _process_meta(self, meta_dict, weight_keys):
        pass

    def _get_layer_start_end_id(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @staticmethod
    def _contains(keys: List[str], val: str):
        for key in keys:
            if val in key:
                return True
        return False

    def create_load_config(self,
                           compute_dtype: torch.dtype,
                           database: BaseDatabase,
                           exported_device: Optional[Any] = None):
        merge_lora = False
        if not self.is_ft_style_weight:
            merge_lora = database.has_lora() and bool(os.environ.get("MERGE_LORA", 1))

        if database.has_lora() and not self.support_lora:
            raise Exception(f"current weights_info: {self.__class__} not support lora, but database has lora")

        load_config = LoadConfig(
            database = database,
            num_layers = self._num_layers,
            hidden_size = self._hidden_size,
            head_num = self._head_num,
            head_num_kv = self._head_num_kv,
            size_per_head = self._size_per_head,
            use_stack_weight = self._use_stack_weight,
            need_ffn_act_scale = self.need_ffn_act_scale,
            inter_size = self._inter_size,
            moe_layer_index = self.moe_layer_index_,
            moe_n_group = self.moe_n_group_,
            inter_padding_size = self._inter_padding_size,
            moe_inter_padding_size = self._moe_inter_padding_size,
            expert_num = self.expert_num_,
            enable_eplb = self.enable_eplb_,
            phy_exp_num = self.phy_exp_num_,
            enable_merge_w13 = self.enable_merge_w13_,
            tp_size = self.tp_size,
            tp_rank = self.tp_rank,
            ep_size = self.ep_size,
            ep_rank = self.ep_rank,
            dp_size = self.dp_size,
            dp_rank = self.dp_rank,
            num_nodes = self.num_nodes,
            ffn_tp_rank = self.ffn_tp_rank,
            ffn_tp_size = self.ffn_tp_size,
            tp_split_emb_and_lm_head = self.tp_split_emb_and_lm_head,
            merge_lora = merge_lora,
            vit_separation = self.vit_separation,
            compute_dtype = compute_dtype,
            quant_algo = self._quant_algo,
            bit = self._quant_algo.getWeightBits(),
            is_ft_style_weight = self.is_ft_style_weight,
            phy2log=self.phy2log_,
            exported_device = exported_device
        )
        return load_config


class ModelWeights:
    def __init__(self, num_layers: int, device: str, dtype: torch.dtype):
        self.device = device
        self.weights: List[Dict[str, torch.Tensor]] = []
        self.global_weights: Dict[str, torch.Tensor] = {}
        self._dtype = dtype
        self.is_ft_style_weight: bool = False

        for _ in range(num_layers):
            self.weights.append({})

    def set_layer_weight(self, layer_id: int, name: str, tensor: torch.Tensor):
        self.weights[layer_id][name] = tensor
        gc.collect()

    def set_global_weight(self, name: str, tensor: torch.Tensor):
        self.global_weights[name] = tensor

    def get_global_weight(self, name: str):
        return self.global_weights.get(name, None)

    def steal_global_weight(self, name: str):
        if name not in self.global_weights:
            return None
        tensor = self.global_weights[name]
        del self.global_weights[name]
        return tensor

    @property
    def dtype(self):
        return self._dtype


    @staticmethod
    def layer_weight_prefix(tp_rank:int, dp_rank: int, ep_rank: int):
        return f"rank_{tp_rank:02d}_{dp_rank:02d}_{ep_rank:02d}.layers."

    @staticmethod
    def global_weight_prefix(tp_rank:int, dp_rank: int, ep_rank: int):
        return f"rank_{tp_rank:02d}_{dp_rank:02d}_{ep_rank:02d}.global."
