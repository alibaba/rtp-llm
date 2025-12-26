import functools
import logging

# Forward references for type hints
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.quant_config import Fp8PerTensorQuantConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnConfig, FfnWeight, MoeWithSharedWeight
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.ops import KvCacheDataType, VitSeparation
from rtp_llm.utils.ckpt_file_info import CkptFileInfo
from rtp_llm.utils.database import BaseDatabase, CkptDatabase
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    choose_available,
    identity,
    tolerate_failed,
)
from rtp_llm.utils.weight_type import WEIGHT_TYPE

if TYPE_CHECKING:
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import HWKernelConfig, ParallelismConfig


def create_scalar_ones(ts: List[torch.Tensor]):
    return torch.ones([1], dtype=torch.float32).to(ts[0].device)


class ModelWeightInfo:
    layer_weights: Union[List[WeightModule], List[List[WeightModule]]]
    weights: List[WeightModule]

    def __init__(
        self,
        weights: List[WeightModule],
        layer_weights: Union[List[WeightModule], List[List[WeightModule]]],
    ) -> None:
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

    def to_quant_weight_info(self, quant_config: QuantizationConfig):
        if quant_config is None:
            raise ValueError("quant_config is None ")
        weights = []
        if self.weights:
            for weight in self.weights:
                weights.append(weight.create(weight, quant_config))
        layer_weights: Union[List[WeightModule], List[List[WeightModule]]] = (
            [] if self.layer_weights else []
        )
        if self.layer_weights:
            for weight in self.layer_weights:
                if isinstance(weight, list):
                    layer_weight = []
                    for w in weight:
                        layer_weight.append(w.create(w, quant_config))
                    layer_weights.append(layer_weight)
                else:
                    layer_weights.append(weight.create(weight, quant_config))

        return ModelWeightInfo(weights, layer_weights)


class ModelDeployWeightInfo:

    TRT_ENGINE_LAYER_WEIGHT_MAP = {
        W.pre_ln_beta: "transformer.layers.{i}.input_layernorm.bias",
        W.pre_ln_gamma: "transformer.layers.{i}.input_layernorm.weight",
        W.attn_qkv_w: "transformer.layers.{i}.attention.qkv.weight",
        W.attn_qkv_b: "transformer.layers.{i}.attention.qkv.bias",
        W.attn_qkv_s: "transformer.layers.{i}.attention.qkv.weights_scaling_factor",
        W.attn_o_w: "transformer.layers.{i}.attention.dense.weight",
        W.attn_o_b: "transformer.layers.{i}.attention.dense.bias",
        W.attn_o_s: "transformer.layers.{i}.attention.dense.weights_scaling_factor",
        W.ffn_w3: "transformer.layers.{i}.mlp.fc.weight",
        W.ffn_b3: "transformer.layers.{i}.mlp.fc.bias",
        W.ffn_s3: "transformer.layers.{i}.mlp.fc.weights_scaling_factor",
        W.ffn_w2: "transformer.layers.{i}.mlp.proj.weight",
        W.ffn_b2: "transformer.layers.{i}.mlp.proj.bias",
        W.ffn_s2: "transformer.layers.{i}.mlp.proj.weights_scaling_factor",
        W.post_ln_gamma: "transformer.layers.{i}.post_layernorm.weight",
        W.post_ln_beta: "transformer.layers.{i}.post_layernorm.bias",
    }

    TRT_ENGINE_LAYER_WEIGHT_MAP2 = {
        W.pre_ln_beta: "transformer.layers.{i}.input_layernorm.bias",
        W.pre_ln_gamma: "transformer.layers.{i}.input_layernorm.weight",
        W.attn_qkv_w: "transformer.layers.{i}.attention.qkv.weight",
        W.attn_qkv_b: "transformer.layers.{i}.attention.qkv.bias",
        W.attn_qkv_s: "transformer.layers.{i}.attention.qkv.weights_scaling_factor",
        W.attn_o_w: "transformer.layers.{i}.attention.dense.weight",
        W.attn_o_b: "transformer.layers.{i}.attention.dense.bias",
        W.attn_o_s: "transformer.layers.{i}.attention.dense.weights_scaling_factor",
        W.ffn_w1: "transformer.layers.{i}.mlp.fc.weight",
        W.ffn_b1: "transformer.layers.{i}.mlp.fc.bias",
        W.ffn_s1: "transformer.layers.{i}.mlp.fc.weights_scaling_factor",
        W.ffn_w2: "transformer.layers.{i}.mlp.proj.weight",
        W.ffn_b2: "transformer.layers.{i}.mlp.proj.bias",
        W.ffn_s2: "transformer.layers.{i}.mlp.proj.weights_scaling_factor",
        W.ffn_w3: "transformer.layers.{i}.mlp.gate.weight",
        W.ffn_b3: "transformer.layers.{i}.mlp.gate.bias",
        W.ffn_s3: "transformer.layers.{i}.mlp.gate.weights_scaling_factor",
        W.ffn_w13: [
            "transformer.layers.{i}.mlp.fc.weight",
            "transformer.layers.{i}.mlp.gate.weight",
        ],
        W.ffn_b13: [
            "transformer.layers.{i}.mlp.fc.bias",
            "transformer.layers.{i}.mlp.gate.bias",
        ],
        W.ffn_s13: [
            "transformer.layers.{i}.mlp.fc.weights_scaling_factor",
            "transformer.layers.{i}.mlp.gate.weights_scaling_factor",
        ],
        W.post_ln_gamma: "transformer.layers.{i}.post_layernorm.weight",
        W.post_ln_beta: "transformer.layers.{i}.post_layernorm.bias",
    }

    def __init__(
        self,
        model_config: "ModelConfig",
        parallelism_config: "ParallelismConfig",
        hw_kernel_config: "HWKernelConfig",
        kv_cache_config: "KVCacheConfig",
        merge_lora: bool = False,
        vit_config: Optional["VitConfig"] = None,
        **kwargs,
    ):
        """Initialize ModelDeployWeightInfo with independent configuration objects."""
        self.model_config = model_config
        self.merge_lora = merge_lora

        self._use_swizzleA = hw_kernel_config.use_swizzleA
        self._use_qk_norm = model_config.qk_norm
        self._hidden_size = model_config.hidden_size
        # inter_size is now accessed from model config when needed, not stored
        self._quant_algo = model_config.quant_algo
        self._head_num = model_config.attn_config.head_num
        self._head_num_kv = model_config.attn_config.kv_head_num
        self.tp_size = parallelism_config.tp_size
        self.tp_rank = parallelism_config.tp_rank
        self.ep_size = parallelism_config.ep_size
        self.ep_rank = parallelism_config.ep_rank
        self.dp_size = parallelism_config.dp_size
        self.dp_rank = parallelism_config.dp_rank
        self.num_nodes: int = (
            parallelism_config.world_size // parallelism_config.local_world_size
        )
        self.ffn_tp_rank = parallelism_config.ffn_tp_rank
        self.ffn_tp_size = parallelism_config.ffn_tp_size
        self._size_per_head = model_config.attn_config.size_per_head
        if self._head_num_kv == -1:
            self._head_num_kv = self._head_num
        self._quant_config = model_config.quant_config

        # Calculate align_size and moe_align_size
        # These will be used by padding functions to compute padding dynamically
        # align_size should be 0 if no quantization and no swizzleA
        if self._quant_algo.isQuant() or self._use_swizzleA:
            if self._quant_algo.isGroupwise():
                self._align_size = self.tp_size * self._quant_algo.getGroupSize()
                self._moe_align_size = self._quant_algo.getGroupSize()
            else:
                self._align_size = self.tp_size * 64
                self._moe_align_size = 64
                if self._quant_algo.isFp8PTPC():
                    self._moe_align_size = 128
        else:
            # No quantization and no swizzleA: no padding needed
            self._align_size = 0
            self._moe_align_size = 64  # Keep moe_align_size for other uses

        # For MoE, adjust moe_align_size for padding if needed
        if self._quant_algo.isQuant():
            self._moe_align_size_for_padding = self._moe_align_size
        else:
            self._moe_align_size_for_padding = 8

        logging.info(
            f"Calculated align_size: {self._align_size}, "
            f"moe_align_size: {self._moe_align_size}, "
            f"moe_align_size_for_padding: {self._moe_align_size_for_padding}"
        )
        self._num_layers = model_config.num_layers
        self._has_prefix_encoder = False
        self._src_quantization_bit = model_config.src_quantization_bit

        self._is_gated_activation = model_config.isGatedActivation()
        self.expert_num_ = model_config.expert_num
        self.moe_n_group_ = model_config.moe_n_group
        self.enable_eplb_ = model_config.eplb_config.enable_eplb()
        self.phy_exp_num_ = model_config.eplb_config.phy_exp_num(
            model_config.expert_num
        )
        self.moe_k_ = model_config.moe_k
        self.moe_layer_index_ = model_config.moe_layer_index
        self.moe_style_ = model_config.moe_style

        self.tie_word_embeddings = model_config.tie_word_embeddings
        self.weight_style = WeightStyle.NONE

        # for mla
        self.kv_lora_rank = model_config.attn_config.kv_lora_rank
        self.nope_head_dim = model_config.attn_config.nope_head_dim
        self.rope_head_dim = model_config.attn_config.rope_head_dim
        self.v_head_dim = model_config.attn_config.v_head_dim
        self.vit_separation = (
            vit_config.vit_separation
            if vit_config is not None
            else VitSeparation.VIT_SEPARATION_LOCAL
        )

        # for moe
        self._use_stack_weight = False

        self.gen_dummy_reciprocal = (
            model_config.attn_config.kv_cache_dtype == KvCacheDataType.FP8
            and not isinstance(model_config.quant_config, Fp8PerTensorQuantConfig)
        )

        self.is_ffn_service = (
            parallelism_config.ffn_disaggregate_config.is_ffn_service()
        )

        # Calculate is_attn_model: True if FFN disaggregate is enabled but this is not an FFN service
        ffn_config = parallelism_config.ffn_disaggregate_config
        self.is_attn_model = (
            ffn_config.enable_ffn_disaggregate and not ffn_config.is_ffn_service()
        )

    @property
    def support_lora(self):
        return False

    @property
    def attn_config(self):
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        return attn_config

    @property
    def ffn_config(self):
        # FfnConfig uses align_size for dynamic padding calculation
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
            is_moe=False,
        )
        return ffn_config

    def get_weight_info(self) -> ModelWeightInfo:
        weight_info = self._get_weight_info()
        # avoid circular import
        from rtp_llm.models.multimodal.multimodal_mixin import BaseMultiModalWeightInfo

        if (
            isinstance(self, BaseMultiModalWeightInfo)
            and self.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE
            and self.tp_rank == 0
        ):
            weight_info = self._get_vit_info(weight_info)

        if weight_info.layer_weights and not isinstance(
            weight_info.layer_weights[0], List
        ):
            layer_weights = []
            for _ in range(self._num_layers):
                layer_weights.append(weight_info.layer_weights)
            weight_info.layer_weights = layer_weights

        if self.weight_style != WeightStyle.NONE:
            logging.info("fix weight style")
            weight_info = self._fix_weight_style_layer_weight(weight_info)

        logging.info("fix merge_w13")
        weight_info = self._fix_merge_w1_w3(weight_info)

        if self.gen_dummy_reciprocal:
            weight_info = self._add_attention_output_static_quant_reciprocal(
                weight_info
            )

        if self._quant_algo is not None and self._quant_algo.isQuant():
            weight_info = weight_info.to_quant_weight_info(self._quant_config)

        if self.tie_word_embeddings:
            logging.info("fix tie_word_embeddings")
            weight_info = self._fix_tie_lm_head(weight_info)
        return weight_info

    def _fix_weight_style_layer_weight(self, origin_weight_info: ModelWeightInfo):
        global_weights = []
        m1 = (
            {
                W.embedding: "transformer.vocab_embedding.weight",
                W.lm_head: "lm_head.weight",
                W.final_ln_gamma: "transformer.ln_f.weight",
            }
            if self.weight_style == WeightStyle.TRT_ENGINE
            else {}
        )

        def __update_weight_style(weight: WeightModule, name_map: Dict[str, str]):
            if isinstance(weight, AtomicWeight):
                weight.weight_style = self.weight_style
                if weight.name in name_map:
                    if len(weight.weights) == 1:
                        weight.weights[0].name = name_map[weight.name]
                    elif weight.name in [W.attn_qkv_b, W.attn_qkv_w]:
                        weight.weights = [CkptWeightInfo(name_map[weight.name])]
                        weight.process_fun = identity
                        logging.error(
                            f"{weight.name} have many weight, maybe cause bug {weight.weights}"
                        )
                    elif weight.name in [W.ffn_w13, W.ffn_b13, W.ffn_s13]:
                        weight.weights[0].name = name_map[weight.name][0]
                        weight.weights[1].name = name_map[weight.name][1]
                    elif len(weight.weights) >= 2:
                        raise ValueError(
                            f"{weight.name} should have only one or zero weight, {weight.weights}"
                        )
                    logging.info(
                        f"update weight style for {weight.name}: {weight.weights[0].name}"
                    )
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
            if (
                ffn_weight[0].w1 is not None or ffn_weight[0].w13 is not None
            ) and self.weight_style == WeightStyle.TRT_ENGINE:
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

    def _add_attention_output_static_quant_reciprocal(
        self, origin_weight_info: ModelWeightInfo
    ):
        for weights in origin_weight_info.layer_weights:
            attn_q_weight_info: Optional[CkptWeightInfo] = None
            for weight in weights:
                if (
                    isinstance(weight, AtomicWeight)
                    or isinstance(weight, AttnAtomicWeight)
                ) and weight.name == W.attn_qkv_w:
                    attn_q_weight_info = weight.weights[0]
                    break

            assert attn_q_weight_info is not None
            weights.append(
                AtomicWeight(
                    W.attention_output_static_quant_reciprocal,
                    [attn_q_weight_info],
                    create_scalar_ones,
                    torch.float32,
                )
            )
            logging.info(
                f"append attention_output_static_quant_reciprocal {weights[-1]}"
            )
        return origin_weight_info

    def _fix_merge_w1_w3(self, origin_weight_info: ModelWeightInfo):
        if len(origin_weight_info.layer_weights) == 0:
            return origin_weight_info

        def __update_weight_config(weight: WeightModule):
            if isinstance(weight, FfnWeight) or isinstance(weight, MoeWithSharedWeight):
                logging.info(f"src_weights: {weight}")
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
        logging.info(
            f"fix weight config when need_merge_w13 {origin_weight_info.layer_weights[0]}"
        )
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
        lm_head_ckpt_weigth_infos = [
            CkptWeightInfo(
                w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)
            )
            for w in lm_head.weights
        ]
        lm_head_ckpt_weigth_infos.extend(
            [
                CkptWeightInfo(
                    w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)
                )
                for w in word_emb.weights
            ]
        )
        lm_head_merge_funcs = [lm_head.process_fun, word_emb.process_fun]
        lm_head = AtomicWeight(
            W.lm_head,
            lm_head_ckpt_weigth_infos,
            functools.partial(choose_available, origin_func_list=lm_head_merge_funcs),
        )
        origin_weight_info.weights[lm_head_idx] = lm_head
        return origin_weight_info

    def _get_weight_info(self) -> ModelWeightInfo:
        raise NotImplementedError()

    def create_model_weight_info(self, database: BaseDatabase) -> ModelWeightInfo:
        if isinstance(database, CkptDatabase) and not database.is_ft_style:
            self.process_meta_from_ckpt(database.pretrain_file_list)
            self.process_meta_from_ckpt(database.finetune_file_list)
            return self.get_weight_info()
        elif database.is_ft_style:
            return None
        else:
            raise Exception("Unknown database class")

    def create_dynamic_weights(self) -> List[AtomicWeight]:
        dynamic_weights = []
        rope_w = self._create_rope_w()
        if rope_w:
            dynamic_weights.append(rope_w)
        return dynamic_weights

    def _create_rope_w(self) -> List[AtomicWeight]:
        return None

    def process_meta_from_ckpt(self, ckpt_metas: List[CkptFileInfo]):
        if len(ckpt_metas) == 0:
            return
        # call subclass process_meta
        meta_dicts = [ckpt_file.get_metadata() for ckpt_file in ckpt_metas]
        weight_keys = set(
            functools.reduce(
                lambda x, y: x + y, [list(meta.keys()) for meta in meta_dicts], []
            )
        )
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

    @staticmethod
    def _exist(keys: List[str], val: str):
        for key in keys:
            if val == key:
                return True
        return False

    def create_load_config(
        self,
        compute_dtype: torch.dtype,
        database: BaseDatabase,
        phy2log: Optional[List[List[int]]] = None,
        exported_device: Optional[Any] = None,
    ):
        merge_lora = False

        if not database.is_ft_style:
            merge_lora = database.has_lora() and self.merge_lora

        if database.has_lora() and not self.support_lora:
            raise Exception(
                f"current weights_info: {self.__class__} not support lora, but database has lora"
            )

        if (
            database.is_ft_style
            and database.ft_weight_params
            and self.vit_separation != VitSeparation.VIT_SEPARATION_ROLE
        ):
            # check ft_style ParallelInfo is match weight's ParallelInfo
            src_tp_size = int(database.ft_weight_params.get("TP_SIZE", self.tp_size))
            src_dp_size = int(database.ft_weight_params.get("DP_SIZE", self.dp_size))
            src_ep_size = int(database.ft_weight_params.get("EP_SIZE", self.ep_size))
            if (
                src_tp_size != self.tp_size
                or src_dp_size != self.dp_size
                or src_ep_size != self.ep_size
            ):
                raise ValueError(
                    f"ft_style ParallelInfo is not match weight's ParallelInfo,"
                    + f"tp_size: {src_tp_size} vs {self.tp_size}, dp_size: {src_dp_size} vs {self.dp_size}, ep_size: {src_ep_size} vs {self.ep_size}"
                )

        load_config = LoadConfig(
            database=database,
            num_layers=self._num_layers,
            hidden_size=self._hidden_size,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
            size_per_head=self._size_per_head,
            use_stack_weight=self._use_stack_weight,
            align_size=self._align_size,
            moe_align_size=self._moe_align_size_for_padding,
            moe_layer_index=self.moe_layer_index_,
            moe_n_group=self.moe_n_group_,
            expert_num=self.expert_num_,
            enable_eplb=self.enable_eplb_,
            phy_exp_num=self.phy_exp_num_,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            num_nodes=self.num_nodes,
            ffn_tp_rank=self.ffn_tp_rank,
            ffn_tp_size=self.ffn_tp_size,
            merge_lora=merge_lora,
            vit_separation=self.vit_separation,
            compute_dtype=compute_dtype,
            quant_algo=self._quant_algo,
            bit=self._quant_algo.getWeightBits(),
            is_ft_style_weight=database.is_ft_style,
            phy2log=phy2log,  # phy2log should be set before create_load_config is called
            exported_device=exported_device,
            use_swizzleA=self._use_swizzleA,
        )
        return load_config


class ModelWeights:
    def __init__(self, num_layers: int, device: str, dtype: torch.dtype):
        self.device = device
        self.weights: List[Dict[str, torch.Tensor]] = []
        self.global_weights: Dict[str, torch.Tensor] = {}
        self._dtype = dtype

        for _ in range(num_layers):
            self.weights.append({})

    def set_layer_weight(self, layer_id: int, name: str, tensor: torch.Tensor):
        self.weights[layer_id][name] = tensor

    def update_layer_weight(self, layer_id: int, name: str, data: torch.Tensor):
        if not isinstance(layer_id, int):
            raise TypeError(
                f"Invalid 'layer_id' type. Expected an integer, but received {type(layer_id).__name__}.\n"
                "Please ensure 'layer_id' is an integer representing the layer index."
            )
        if layer_id < 0 or layer_id >= len(self.weights):
            raise ValueError(
                f"Invalid 'layer_id'. It is out of the valid range for the model's weights.\n"
                f"Received 'layer_id': {layer_id}\n"
                f"Valid 'layer_id' range: 0 to {len(self.weights) - 1} (inclusive)."
            )
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Invalid 'data' type for layer weight. Expected 'torch.Tensor', "
                f"but received {type(data).__name__}.\n"
                "Please provide the weight data as a PyTorch tensor."
            )
        layer_weight_dict: dict[str, torch.Tensor] = self.weights[layer_id]
        if name not in layer_weight_dict:
            raise KeyError(
                f"Weight name '{name}' not found within layer {layer_id}.\n"
                f"Available weights in layer {layer_id}: {list(layer_weight_dict.keys())}\n"
                "Please check the provided weight 'name'."
            )
        ori_tensor = layer_weight_dict[name]
        self.check_data(
            ori_tensor, data
        )  # This will raise errors if shape, device, or dtype mismatch
        with torch.inference_mode():
            ori_tensor.copy_(data.to(ori_tensor.device))

    def set_global_weight(self, name: str, tensor: torch.Tensor):
        self.global_weights[name] = tensor

    def get_global_weight_or_none(self, name: str):
        return self.global_weights.get(name, None)

    def get_global_weight(self, name: str) -> torch.Tensor:
        return self.global_weights[name]

    def update_global_weight(self, name: str, data: torch.Tensor):
        if not isinstance(name, str):
            raise TypeError(
                f"Invalid 'name' type. Expected a string, but received {type(name).__name__}.\n"
                "Please provide the name of the global weight as a string."
            )
        if name not in self.global_weights:
            raise KeyError(
                f"Global weight with name '{name}' not found.\n"
                f"Available global weights: {list(self.global_weights.keys())}\n"
                "Please check the provided global weight name."
            )
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Invalid tensor type found in the provided 'data' dictionary. "
                f"Expected 'torch.Tensor', but found {type(data).__name__}.\n"
                "If 'data' is a dictionary, its values must be PyTorch tensors."
            )
        original_global_tensor = self.global_weights[name]
        # Use the check_data method to validate shape, device, and dtype
        self.check_data(original_global_tensor, data)
        with torch.inference_mode():
            original_global_tensor.copy_(data.to(original_global_tensor.device))

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
    def layer_weight_prefix(tp_rank: int, dp_rank: int, ep_rank: int):
        return f"rank_{tp_rank:02d}_{dp_rank:02d}_{ep_rank:02d}.layers."

    @staticmethod
    def global_weight_prefix(tp_rank: int, dp_rank: int, ep_rank: int):
        return f"rank_{tp_rank:02d}_{dp_rank:02d}_{ep_rank:02d}.global."

    def check_data(self, ori_tensor: torch.Tensor, update_tensor: torch.Tensor):
        if ori_tensor.shape != update_tensor.shape:
            raise ValueError(
                "Input error: The shape of your input tensor does not match the original tensor.\n"
                f"Input tensor shape: {update_tensor.shape}\n"
                f"Original tensor shape: {ori_tensor.shape}"
            )
        if ori_tensor.device != update_tensor.device:
            raise ValueError(
                "Input error: The device of your input tensor does not match the original tensor.\n"
                f"Input tensor device: {update_tensor.device}\n"
                f"Original tensor device: {ori_tensor.device}"
            )
        if ori_tensor.dtype != update_tensor.dtype:
            raise ValueError(
                "Input error: The data type of your input tensor does not match the original tensor.\n"
                f"Input tensor dtype: {update_tensor.dtype}\n"
                f"Original tensor dtype: {ori_tensor.dtype}"
            )
