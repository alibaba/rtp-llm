import functools
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import Int8PerTensorQuantConfig, QuantizationConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.w8a8_weight import (
    W8A8Int8AtomicWeight,
    create_w8a8_int8_weight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    expand_scale,
    get_tensor_from_scalar,
    get_tensor_reciprocal,
    identity,
    merge_qkv_transpose_concat0,
    sp_id,
    transpose,
    transpose_w13,
)

W_SUFFIX = ".weight"
B_SUFFIX = ".bias"
ACT_S_SUFFIX = "_input_scale"
W_S_SUFFIX = ".scales"


def _decoder_qkv_scales(values, widths):
    parts = []
    for value, width in zip(values, widths):
        parts.append(value.reshape(-1).expand(width))
    return torch.cat(parts)


def _gate_scales(values):
    gate, up = (s.reshape(-1) for s in values)
    width = max(gate.numel(), up.numel())
    return torch.cat((gate.expand(width), up.expand(width))).reshape(1, -1)


def _split_gate_scales(t, ffn_tp_size, ffn_tp_rank, **kwargs):
    gate, up = t.reshape(-1).chunk(2)
    if gate.numel() == 1:
        return t
    if gate.numel() % ffn_tp_size:
        raise ValueError("FT gate/up output scales must split on whole channels")
    return torch.cat(
        (gate.chunk(ffn_tp_size)[ffn_tp_rank], up.chunk(ffn_tp_size)[ffn_tp_rank])
    ).reshape(1, -1)


class PerTensorInt8QuantWeight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2,
        W.ffn_w13,
        W.pre_decoder_ln_gamma,  # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc
    ]
    EMB_NORM_S: str = "encoder.layer.0.attention.self.qkv_input_scale"
    FFN_OUTPUT_LAYERNORM_S_SUFFIX: str = ".qkv_input_scale"
    INT8_SCALE_MAP = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w1: W.ffn_s1,
    }

    INT8_ACT_SCALE_MAP = {
        W.attn_qkv_w: [
            (W.pre_ln_static_quant, get_tensor_reciprocal),
            (W.pre_ln_static_quant_reciprocal, get_tensor_from_scalar),
        ],
        W.attn_o_w: [
            (W.attention_output_static_quant, get_tensor_reciprocal),
            (W.attention_output_static_quant_reciprocal, get_tensor_from_scalar),
        ],
        W.ffn_w2: [
            (W.ffn_intermediate_weight2_static_quant, get_tensor_reciprocal),
            (
                W.ffn_intermediate_weight2_static_quant_reciprocal,
                get_tensor_from_scalar,
            ),
        ],
        W.ffn_w3: [
            (W.post_ln_static_quant, get_tensor_reciprocal),
            (W.post_ln_static_quant_reciprocal, get_tensor_from_scalar),
        ],
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Int8PerTensorQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and (
            src_weight_info.weight_style
            not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE]
        )

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        self.quant_config = quant_config
        kernel: AtomicWeight = None
        scale: Optional[AtomicWeight] = None
        act_scale: Optional[AtomicWeight] = None
        act_scale_inv: Optional[AtomicWeight] = None
        logging.debug(
            "PerTensorInt8QuantWeight : %s, %s", self.qs_suffix, self.qw_suffix
        )

        if src_weight_info.name == W.attn_qkv_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_qkv_quant_weight_info(
                src_weight_info
            )
        elif src_weight_info.name == W.attn_o_w:
            (kernel, scale, act_scale, act_scale_inv) = (
                self._get_attn_out_quant_weight_info(src_weight_info)
            )
        elif src_weight_info.name in [
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
            W.ffn_w13,
            W.moe_w1,
            W.moe_w2,
        ]:
            (kernel, scale, act_scale, act_scale_inv) = self._get_ffn_quant_weight_info(
                src_weight_info, quant_config
            )
        elif src_weight_info.name == W.pre_decoder_ln_gamma:
            (kernel, scale, act_scale, act_scale_inv) = self.get_pre_decoder_ln_weight(
                src_weight_info
            )
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")

        sub_weights = {
            kernel.name: kernel,
        }

        if scale is not None:
            sub_weights[scale.name] = scale
        if act_scale is not None:
            sub_weights[act_scale.name] = act_scale
        if act_scale_inv is not None:
            sub_weights[act_scale_inv.name] = act_scale_inv

        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale
        self.act_scale = act_scale
        self.act_scale_inv = act_scale_inv
        # Calibration steps are shared by all ranks. Only the gate/up weight
        # scales split, along the output channels within each branch.
        for field in (act_scale, act_scale_inv):
            if field is not None:
                field.gpt_style_tp_strategy = {field.name: sp_id}
        if scale is not None and scale.name == W.ffn_s13:
            scale.gpt_style_tp_strategy = {scale.name: _split_gate_scales}

    def get_pre_decoder_ln_weight(
        self, src_weight_info: AtomicWeight
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        w_name = src_weight_info.weights[0].name
        act_s = W.pre_decoder_ln_static_quant
        act_s_inv = W.pre_decoder_ln_static_quant_reciprocal
        prefix = (
            w_name.split(".")[0] + "."
        )  # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc

        return [
            create_w8a8_int8_weight(
                src_weight_info,
                src_weight_info.name,
                [CkptWeightInfo(w_name, identity)],
                identity,
                data_type=src_weight_info.data_type,
            ),
            None,
            create_w8a8_int8_weight(
                src_weight_info,
                act_s,
                [CkptWeightInfo(prefix + self.EMB_NORM_S)],
                get_tensor_reciprocal,
                data_type=torch.float32,
            ),
            create_w8a8_int8_weight(
                src_weight_info,
                act_s_inv,
                [CkptWeightInfo(prefix + self.EMB_NORM_S)],
                get_tensor_from_scalar,
                data_type=torch.float32,
            ),
        ]

    def _get_qkv_quant_weight_info(
        self, src_weight_info: AtomicWeight
    ) -> List[W8A8Int8AtomicWeight]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 3

        need_post_ln = src_weight_info.config.need_post_ln

        qkv_w_list: List[CkptWeightInfo] = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + self.qw_suffix)
            for sub_w in weights
        ]
        qkv_s_list: List[CkptWeightInfo] = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + self.qs_suffix)
            for sub_w in weights
        ]

        qkv_w = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_w_list,
            merge_qkv_transpose_concat0,
            data_type=torch.int8,
            config=src_weight_info.config,
        )

        decoder_q = weights[0].name.endswith(".q_proj.weight")
        cfg = src_weight_info.config
        scale_merge = functools.partial(expand_scale, hidden_size=cfg.hidden_size)
        if decoder_q:
            kv = cfg.head_num_kv if cfg.head_num_kv > 0 else cfg.head_num
            scale_merge = functools.partial(
                _decoder_qkv_scales,
                widths=(
                    cfg.head_num * cfg.size_per_head,
                    kv * cfg.size_per_head,
                    kv * cfg.size_per_head,
                ),
            )
        qkv_s = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_qkv_s,
            qkv_s_list,
            scale_merge,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        if need_post_ln:
            post_ln_w = (
                weights[0].name.replace("{i}", "{i_1}").rsplit(".", 2)[0]
                + self.FFN_OUTPUT_LAYERNORM_S_SUFFIX
            )  # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc
            return [
                qkv_w,
                qkv_s,
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.post_ffn_ln_static_quant,
                    [CkptWeightInfo(post_ln_w)],
                    get_tensor_reciprocal,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.post_ffn_ln_static_quant_reciprocal,
                    [CkptWeightInfo(post_ln_w)],
                    get_tensor_from_scalar,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
            ]
        if decoder_q:
            name = weights[0].name.removesuffix("q_proj.weight") + "qkv_input_scale"
            self._decoder_qkv_step = CkptWeightInfo(name)
            return [
                qkv_w,
                qkv_s,
                *[
                    create_w8a8_int8_weight(
                        src_weight_info,
                        field,
                        [CkptWeightInfo(name)],
                        transform,
                        data_type=torch.float32,
                        config=cfg,
                    )
                    for field, transform in self.INT8_ACT_SCALE_MAP[W.attn_qkv_w]
                ],
            ]
        return [qkv_w, qkv_s, None, None]

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        step = getattr(self, "_decoder_qkv_step", None)
        if step is not None:
            nominal = step.tensor_name(layer_id)
            stem = nominal.removesuffix("qkv_input_scale")
            aliases = [
                stem + suffix
                for suffix in (
                    "qkv_input_scale",
                    "qkv_proj_input_scale",
                    "q_proj_input_scale",
                    "qkv_proj.input_scale",
                )
                if tensor_source.has_tensor(stem + suffix)
            ]
            if len(aliases) != 1:
                raise ValueError(
                    f"Expected one calibrated INT8 QKV input scale for {stem}, found {aliases}"
                )

            source = tensor_source

            class ResolvedSource:
                def load_tensor(self, key, dtype):
                    return source.load_tensor(
                        aliases[0] if key == nominal else key, dtype
                    )

            tensor_source = ResolvedSource()
        return super()._load_raw_tensor(tensor_source, layer_id, device, load_config)

    def _get_attn_out_quant_weight_info(
        self, src_weight_info: WeightModule
    ) -> List[W8A8Int8AtomicWeight]:
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        act_s = self.INT8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.INT8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [
            create_w8a8_int8_weight(
                src_weight_info,
                W.attn_o_w,
                [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                transpose,
                data_type=torch.int8,
                config=src_weight_info.config,
            ),
            create_w8a8_int8_weight(
                src_weight_info,
                W.attn_o_s,
                [CkptWeightInfo(w_name + self.qs_suffix)],
                identity,
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_int8_weight(
                src_weight_info,
                act_s[0],
                [CkptWeightInfo(w_name + self.act_s_suffix)],
                act_s[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_int8_weight(
                src_weight_info,
                act_s_inv[0],
                [CkptWeightInfo(w_name + self.act_s_suffix)],
                act_s_inv[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
        ]

    def _get_ffn_quant_weight_info(
        self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_config: Any
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        weights = src_weight.weights
        ffn_w_name = src_weight.name
        assert weights[0].name.endswith(W_SUFFIX)
        assert ffn_w_name in [W.ffn_w13, W.ffn_w2, W.moe_w1, W.moe_w2, W.ffn_w3]

        if ffn_w_name in [W.ffn_w2]:
            assert len(weights) == 1
        w_name = weights[0].name[: -len(W_SUFFIX)]
        w: str = None
        s: str = None
        if ffn_w_name in [W.moe_w2, W.moe_w1]:
            raise ValueError(f"now not support {ffn_w_name}")
        elif ffn_w_name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            act_s = self.INT8_ACT_SCALE_MAP[W.ffn_w3][0]
            act_s_inv = self.INT8_ACT_SCALE_MAP[W.ffn_w3][1]

            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(w1_name + self.qw_suffix, identity),
                        CkptWeightInfo(w3_name + self.qw_suffix, identity),
                    ],
                    transpose_w13,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [
                        CkptWeightInfo(w1_name + self.qs_suffix, identity),
                        CkptWeightInfo(w3_name + self.qs_suffix, identity),
                    ],
                    _gate_scales,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    act_s[0],
                    [CkptWeightInfo(w_name + self.act_s_suffix)],
                    act_s[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    act_s_inv[0],
                    [CkptWeightInfo(w_name + self.act_s_suffix)],
                    act_s_inv[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
        else:
            w = src_weight.name
            s = self.INT8_SCALE_MAP.get(src_weight.name)

            w_list = [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    transpose,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    transpose,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
            if w in self.INT8_ACT_SCALE_MAP:
                act_s = self.INT8_ACT_SCALE_MAP[src_weight.name][0]
                act_s_inv = self.INT8_ACT_SCALE_MAP[src_weight.name][1]
                w_list.extend(
                    [
                        create_w8a8_int8_weight(
                            src_weight,
                            act_s[0],
                            [CkptWeightInfo(w_name + self.act_s_suffix)],
                            act_s[1],
                            data_type=torch.float32,
                            config=src_weight.config,
                        ),
                        create_w8a8_int8_weight(
                            src_weight,
                            act_s_inv[0],
                            [CkptWeightInfo(w_name + self.act_s_suffix)],
                            act_s_inv[1],
                            data_type=torch.float32,
                            config=src_weight.config,
                        ),
                    ]
                )
            else:
                w_list.extend([None, None])
            return w_list

    @property
    def qw_suffix(self) -> str:
        return ".qweight"

    @property
    def qs_suffix(self) -> str:
        return W_S_SUFFIX

    @property
    def act_s_suffix(self) -> str:
        return ACT_S_SUFFIX

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        from rtp_llm.models_py.quant_methods.int8 import PrequantizedInt8ExportModule

        # Keep FT's byte layout and activation-scale fields unchanged. The
        # BERT pre-decoder norm is floating point despite this recipe's name.
        module = PrequantizedInt8ExportModule(
            kernel_weight,
            per_channel=False,
            norm=self.kernel.name == W.pre_decoder_ln_gamma,
        )
        kernel_weight, _ = module.export()
        processed_res[self.kernel.name] = kernel_weight

        return processed_res
