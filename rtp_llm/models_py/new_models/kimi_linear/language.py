import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.kimi_linear import (
    KimiLinearDecoderLayer,
    KimiLinearMetadata,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, RMSResNorm
from rtp_llm.ops import HybridAttentionType, MlaOpsType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import (
    W,
    concat_0_tranpose,
    merge_qkv_hf,
    mla_pad_t,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    transpose_slice_k,
    transpose_slice_v,
)

logger = logging.getLogger(__name__)


def _param(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor.contiguous(), requires_grad=False)


def _split_dim(tensor: torch.Tensor, dim: int, size: int, rank: int) -> torch.Tensor:
    if size <= 1:
        return tensor.contiguous()
    return torch.split(tensor, tensor.shape[dim] // size, dim=dim)[rank].contiguous()


def _split_embedding(tensor: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    return _split_dim(tensor, -1, tp_size, tp_rank)


def _split_attn_o(tensor: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    return _split_dim(tensor, 0, tp_size, tp_rank)


def _split_lm_head(tensor: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor.contiguous()
    align_size = tp_size * 8
    padded_size = int(math.ceil(tensor.shape[0] * 1.0 / align_size) * align_size)
    pad_size = padded_size - tensor.shape[0]
    per_slice_size = padded_size // tp_size
    start = tp_rank * per_slice_size
    if pad_size != 0 and tp_rank == tp_size - 1:
        pad_shape = [pad_size] + list(tensor.shape[1:])
        return torch.cat(
            [
                tensor[start:, ...],
                torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype),
            ],
            dim=0,
        ).contiguous()
    return tensor[start : start + per_slice_size, ...].contiguous()


def _split_ffn_w1_w3(
    tensor: torch.Tensor, ffn_tp_size: int, ffn_tp_rank: int
) -> torch.Tensor:
    return _split_dim(tensor, -1, ffn_tp_size, ffn_tp_rank)


def _split_ffn_w2(
    tensor: torch.Tensor, ffn_tp_size: int, ffn_tp_rank: int
) -> torch.Tensor:
    return _split_dim(tensor, 0, ffn_tp_size, ffn_tp_rank)


def _split_moe_w1(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    moe_pure_tp_mode: bool,
) -> torch.Tensor:
    if not moe_pure_tp_mode:
        return tensor.contiguous()
    reshaped = tensor.reshape(tensor.shape[0], 2, -1, tensor.shape[-1])
    shard = torch.split(reshaped, reshaped.shape[2] // tp_size, dim=2)[tp_rank]
    return shard.reshape(shard.shape[0], -1, shard.shape[-1]).contiguous()


def _split_moe_w2(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    moe_pure_tp_mode: bool,
) -> torch.Tensor:
    if not moe_pure_tp_mode:
        return tensor.contiguous()
    return _split_dim(tensor, -1, tp_size, tp_rank)


def _split_kda_qkv(tensor: torch.Tensor, cfg: Any, tp_size: int, tp_rank: int):
    q_size = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    k_size = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_size = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    q, k, v = torch.split(tensor, [q_size, k_size, v_size], dim=1)
    q = _split_dim(q, 1, tp_size, tp_rank)
    k = _split_dim(k, 1, tp_size, tp_rank)
    v = _split_dim(v, 1, tp_size, tp_rank)
    return torch.cat([q, k, v], dim=1).contiguous()


def _split_kda_dim1(tensor: torch.Tensor, tp_size: int, tp_rank: int):
    return _split_dim(tensor, 1, tp_size, tp_rank)


def _split_kda_dt_bias(tensor: torch.Tensor, cfg: Any, tp_size: int, tp_rank: int):
    if tp_size <= 1:
        return tensor.contiguous()
    num_heads = cfg.linear_num_value_heads
    head_dim = cfg.linear_key_head_dim
    local_heads = num_heads // tp_size
    tensor = tensor.reshape(num_heads, head_dim)
    start = local_heads * tp_rank
    return tensor[start : start + local_heads].reshape(-1).contiguous()


def _split_kda_head(tensor: torch.Tensor, cfg: Any, tp_size: int, tp_rank: int):
    if tp_size <= 1:
        return tensor.contiguous()
    local_heads = cfg.linear_num_value_heads // tp_size
    start = local_heads * tp_rank
    return tensor[start : start + local_heads].contiguous()


def _split_kda_conv(tensor: torch.Tensor, cfg: Any, tp_size: int, tp_rank: int):
    q_size = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    k_size = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_size = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    q, k, v = torch.split(tensor, [q_size, k_size, v_size], dim=0)
    q = _split_dim(q, 0, tp_size, tp_rank)
    k = _split_dim(k, 0, tp_size, tp_rank)
    v = _split_dim(v, 0, tp_size, tp_rank)
    return torch.cat([q, k, v], dim=0).contiguous()


def _split_kda_out(
    tensor: torch.Tensor, cfg: Any, tp_size: int, tp_rank: int
) -> torch.Tensor:
    if tp_size <= 1:
        return tensor.contiguous()
    _, n = tensor.shape
    tensor = tensor.view(cfg.linear_num_value_heads, -1, n)
    local_heads = cfg.linear_num_value_heads // tp_size
    start = local_heads * tp_rank
    return tensor[start : start + local_heads].reshape(-1, n).contiguous()


def _split_mla_no_lora(
    tensor: torch.Tensor,
    head_num: int,
    size_per_head: int,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    q = torch.split(
        tensor[:, : head_num * size_per_head],
        head_num * size_per_head // tp_size,
        dim=-1,
    )[tp_rank]
    rest = tensor[:, head_num * size_per_head :]
    return torch.cat([q, rest], dim=-1).contiguous()


def _to_dtype(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device] = None,
    force_fp32: bool = False,
) -> torch.Tensor:
    target_dtype = torch.float32 if force_fp32 else dtype
    if device is not None or (
        target_dtype is not None and tensor.dtype != target_dtype
    ):
        return tensor.to(device=device, dtype=target_dtype)
    if force_fp32:
        return tensor.to(torch.float32)
    return tensor


class _WeightBucket:
    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: Optional[torch.device],
    ):
        self.tensors = tensors
        self.dtype = dtype
        self.device = device

    def take(self, name: str, force_fp32: bool = False) -> torch.Tensor:
        return _to_dtype(self.tensors[name], self.dtype, self.device, force_fp32)

    def take_cpu(self, name: str, force_fp32: bool = False) -> torch.Tensor:
        return _to_dtype(self.tensors[name], self.dtype, None, force_fp32)

    def maybe_take(self, name: str, force_fp32: bool = False) -> Optional[torch.Tensor]:
        tensor = self.tensors.get(name)
        if tensor is None:
            return None
        return _to_dtype(tensor, self.dtype, self.device, force_fp32)

    def to_target(self, tensor: torch.Tensor, force_fp32: bool = False) -> torch.Tensor:
        return _to_dtype(tensor, self.dtype, self.device, force_fp32)


class _LinearWeightView(nn.Module):
    def __init__(self, **weights: torch.Tensor):
        super().__init__()
        for name, tensor in weights.items():
            self.register_parameter(name, _param(tensor))


class _RuntimeWeightShell:
    def __init__(
        self,
        global_weights: Dict[str, torch.Tensor],
        layer_weights: List[Dict[str, torch.Tensor]],
    ):
        self.global_weights = global_weights
        self.weights = layer_weights

    def get_global_weight_or_none(self, name: str) -> Optional[torch.Tensor]:
        return self.global_weights.get(name)

    def get_global_weight(self, name: str) -> torch.Tensor:
        return self.global_weights[name]


def _attach_runtime_weight_buffers(
    module: nn.Module, weights: Dict[str, torch.Tensor]
) -> Dict[str, str]:
    """Attach RTP internal-layout tensors to a module and return key -> buffer name."""
    key_to_buffer_name: Dict[str, str] = {}
    for idx, (key, tensor) in enumerate(weights.items()):
        buffer_name = f"_runtime_weight_{idx}"
        module.register_buffer(buffer_name, tensor, persistent=False)
        key_to_buffer_name[key] = buffer_name
    return key_to_buffer_name


def _runtime_weight_view(module: nn.Module) -> Dict[str, torch.Tensor]:
    key_to_buffer_name = getattr(module, "_runtime_weight_buffer_names", {})
    return {key: getattr(module, name) for key, name in key_to_buffer_name.items()}


class _KimiLinearRuntimeModel(GptModelBase):
    def __init__(
        self,
        model_config: Any,
        load_config: Any,
        global_weights: Dict[str, torch.Tensor],
        layer_weights: List[Dict[str, torch.Tensor]],
    ):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)
        py_hw_kernel_config = getattr(load_config, "py_hw_kernel_config", None)
        moe_config = getattr(load_config, "moe_config", None)
        max_generate_batch_size = getattr(load_config, "max_generate_batch_size", 32)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.lm_head = _LinearWeightView(weight=global_weights[W.lm_head])
        self.embed_tokens = Embedding(
            model_config, parallelism_config, _param(global_weights[W.embedding])
        )
        enable_cuda_graph = (
            getattr(py_hw_kernel_config, "enable_cuda_graph", False)
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(
                    model_config,
                    parallelism_config,
                    layer_weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                )
                for idx in range(model_config.num_layers)
            ]
        )
        for layer, weights in zip(self.layers, layer_weights):
            layer._runtime_weight_buffer_names = _attach_runtime_weight_buffers(
                layer, weights
            )
        self.norm = RMSResNorm(
            _param(global_weights[W.final_ln_gamma]), eps=model_config.layernorm_eps
        )

    def initialize(self, init_resource):
        ok = super().initialize(init_resource)
        self._ensure_weight_shell()
        return ok

    def _ensure_weight_shell(self):
        global_weights = {
            W.embedding: self.embed_tokens.weight,
            W.final_ln_gamma: self.norm.weight,
            W.lm_head: self.lm_head.weight,
        }
        layer_weights = [_runtime_weight_view(layer) for layer in self.layers]
        self.weight = _RuntimeWeightShell(global_weights, layer_weights)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=attention_inputs.cu_seqlens,
                device=hidden_states.device,
            )
        attn_meta = KimiLinearMetadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            self._ensure_weight_shell()
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers):
            select_block_map_for_layer(attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


class KimiLinearForCausalLM(GptModelBase):
    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)
        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )
        self.model_config = model_config
        self.load_config = load_config
        self.model: Optional[_KimiLinearRuntimeModel] = None

    def load_weights(self, weights):
        weights_iter = iter(weights.items()) if isinstance(weights, dict) else weights
        tensors = self._collect_weights(weights_iter)
        logger.info("KimiLinear newloader collected %s tensors", len(tensors))
        bucket = _WeightBucket(
            tensors,
            getattr(self.load_config, "compute_dtype", torch.bfloat16),
            self._target_device(),
        )
        logger.info("KimiLinear newloader building global weights")
        global_weights = self._postprocess_weight_dict(
            self._build_global_weights(bucket)
        )
        logger.info(
            "KimiLinear newloader building %s layers", self.model_config.num_layers
        )
        layer_weights = []
        for idx in range(self.model_config.num_layers):
            layer_weights.append(
                self._postprocess_weight_dict(self._build_layer_weights(bucket, idx))
            )
            logger.info("KimiLinear newloader built layer %s", idx)
        logger.info("KimiLinear newloader constructing runtime model")
        self.model = _KimiLinearRuntimeModel(
            self.model_config,
            self.load_config,
            global_weights,
            layer_weights,
        )
        tensors.clear()
        logger.info("KimiLinear newloader finished load_weights")

    def initialize(self, init_resource):
        assert self.model is not None, "KimiLinear weights are not loaded"
        ok = self.model.initialize(init_resource)
        self.weight = self.model.weight
        return ok

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        assert self.model is not None, "KimiLinear weights are not loaded"
        ensure_weight_shell = getattr(self.model, "_ensure_weight_shell", None)
        if ensure_weight_shell is not None:
            ensure_weight_shell()
        self.weight = self.model.weight
        return self.model.prepare_fmha_impl(inputs, is_cuda_graph)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        assert self.model is not None, "KimiLinear weights are not loaded"
        return self.model(inputs, fmha_impl)

    def _collect_weights(
        self, weights_iter: Iterable[Tuple[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tensors = {}
        for name, tensor in weights_iter:
            if name.startswith("model."):
                name = name[len("model.") :]
            tensors[name] = tensor
        return tensors

    def _target_device(self) -> Optional[torch.device]:
        device = getattr(self.load_config, "device", None)
        if device:
            return torch.device(device)
        if not torch.cuda.is_available():
            return None
        return torch.device("cuda")

    def _postprocess_weight_dict(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        exported_device = getattr(self.load_config, "exported_device", None)
        if exported_device is None:
            return weights
        return {
            key: exported_device.maybe_rewrite_weight_by_key(key, tensor)
            for key, tensor in weights.items()
        }

    def _build_global_weights(self, bucket: _WeightBucket) -> Dict[str, torch.Tensor]:
        tp_size = self._attn_tp_size()
        tp_rank = self._attn_tp_rank()
        raw_embed = bucket.take("embed_tokens.weight")
        embed = _split_embedding(raw_embed, tp_size, tp_rank)
        use_fp32_lm_head = getattr(
            self.model_config,
            "enable_fp32_lm_head",
            getattr(self.load_config, "enable_fp32_lm_head", False),
        )
        lm_head = bucket.maybe_take("lm_head.weight", force_fp32=use_fp32_lm_head)
        if lm_head is None:
            lm_head = raw_embed.to(torch.float32) if use_fp32_lm_head else raw_embed
        lm_head = _split_lm_head(
            lm_head, self._lm_head_tp_size(), self._lm_head_tp_rank()
        )
        return {
            W.embedding: embed,
            W.final_ln_gamma: bucket.take("norm.weight"),
            W.lm_head: lm_head,
        }

    def _build_layer_weights(
        self, bucket: _WeightBucket, layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        prefix = f"layers.{layer_idx}."
        weights: Dict[str, torch.Tensor] = {
            W.pre_ln_gamma: bucket.take(prefix + "input_layernorm.weight"),
            W.post_ln_gamma: bucket.take(prefix + "post_attention_layernorm.weight"),
        }
        layer_type = self.model_config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if layer_type == HybridAttentionType.LINEAR:
            weights.update(self._build_kda_weights(bucket, prefix))
        else:
            weights.update(self._build_mla_weights(bucket, prefix))
        weights.update(self._build_ffn_weights(bucket, prefix, layer_idx))
        return weights

    def _build_kda_weights(
        self, bucket: _WeightBucket, prefix: str
    ) -> Dict[str, torch.Tensor]:
        cfg = self.model_config.linear_attention_config
        tp_size = self._attn_tp_size()
        tp_rank = self._attn_tp_rank()
        qkv = merge_qkv_hf(
            [
                bucket.take(prefix + "self_attn.q_proj.weight"),
                bucket.take(prefix + "self_attn.k_proj.weight"),
                bucket.take(prefix + "self_attn.v_proj.weight"),
            ]
        )
        conv = torch.cat(
            [
                bucket.take(prefix + "self_attn.q_conv1d.weight"),
                bucket.take(prefix + "self_attn.k_conv1d.weight"),
                bucket.take(prefix + "self_attn.v_conv1d.weight"),
            ],
            dim=0,
        ).contiguous()
        return {
            W.linear_attn_qkv_w: _split_kda_qkv(qkv, cfg, tp_size, tp_rank),
            W.linear_attn_b_w: _split_kda_dim1(
                transpose([bucket.take(prefix + "self_attn.b_proj.weight")]),
                tp_size,
                tp_rank,
            ),
            W.linear_attn_f_a_w: transpose(
                [bucket.take(prefix + "self_attn.f_a_proj.weight")]
            ),
            W.linear_attn_f_b_w: _split_kda_dim1(
                transpose([bucket.take(prefix + "self_attn.f_b_proj.weight")]),
                tp_size,
                tp_rank,
            ),
            W.linear_attn_g_a_w: transpose(
                [bucket.take(prefix + "self_attn.g_a_proj.weight")]
            ),
            W.linear_attn_g_b_w: _split_kda_dim1(
                transpose([bucket.take(prefix + "self_attn.g_b_proj.weight")]),
                tp_size,
                tp_rank,
            ),
            W.linear_attn_conv1d_w: _split_kda_conv(conv, cfg, tp_size, tp_rank),
            W.linear_attn_norm_w: bucket.take(prefix + "self_attn.o_norm.weight"),
            W.linear_attn_dt_b_kda: _split_kda_dt_bias(
                bucket.take(prefix + "self_attn.dt_bias", force_fp32=True),
                cfg,
                tp_size,
                tp_rank,
            ),
            W.linear_attn_alog: _split_kda_head(
                bucket.take(prefix + "self_attn.A_log", force_fp32=True).squeeze(),
                cfg,
                tp_size,
                tp_rank,
            ),
            W.linear_attn_out_w: _split_kda_out(
                transpose([bucket.take(prefix + "self_attn.o_proj.weight")]),
                cfg,
                tp_size,
                tp_rank,
            ),
        }

    def _build_mla_weights(
        self, bucket: _WeightBucket, prefix: str
    ) -> Dict[str, torch.Tensor]:
        cfg = self.model_config.attn_config
        tp_size = self._attn_tp_size()
        tp_rank = self._attn_tp_rank()
        weights = {
            W.attn_o_w: _split_attn_o(
                mla_pad_t(
                    [bucket.take(prefix + "self_attn.o_proj.weight")],
                    head_num=cfg.head_num,
                    nope_head_dim=cfg.v_head_dim,
                    rope_head_dim=0,
                ),
                tp_size,
                tp_rank,
            ),
            W.mla_kv_b_w: _split_dim(
                transpose([bucket.take(prefix + "self_attn.kv_b_proj.weight")]),
                -1,
                tp_size,
                tp_rank,
            ),
            W.mla_kv_a_ln_gamma: bucket.take(
                prefix + "self_attn.kv_a_layernorm.weight"
            ),
        }
        use_mla_absorb = (
            self.model_config.attn_config.use_mla
            and self.model_config.mla_ops_type != MlaOpsType.MHA
        )
        if cfg.q_lora_rank > 0:
            weights[W.mla_q_b_w] = _split_dim(
                transpose([bucket.take(prefix + "self_attn.q_b_proj.weight")]),
                -1,
                tp_size,
                tp_rank,
            )
            weights[W.mla_q_a_ln_gamma] = bucket.take(
                prefix + "self_attn.q_a_layernorm.weight"
            )
            weights[W.mla_fusedqkrope_w] = concat_0_tranpose(
                [
                    bucket.take(prefix + "self_attn.q_a_proj.weight"),
                    bucket.take(prefix + "self_attn.kv_a_proj_with_mqa.weight"),
                ]
            )
        else:
            fused = concat_0_tranpose(
                [
                    bucket.take(prefix + "self_attn.q_proj.weight"),
                    bucket.take(prefix + "self_attn.kv_a_proj_with_mqa.weight"),
                ]
            )
            weights[W.mla_fusedqkrope_no_lora_w] = _split_mla_no_lora(
                fused,
                cfg.head_num,
                cfg.size_per_head,
                tp_size,
                tp_rank,
            )
        if use_mla_absorb:
            kv_b = bucket.take(prefix + "self_attn.kv_b_proj.weight")
            weights[W.mla_kc] = _split_attn_o(
                transpose_slice_k(
                    [kv_b],
                    head_num=cfg.head_num,
                    nope_head_dim=cfg.nope_head_dim,
                    v_head_dim=cfg.v_head_dim,
                    lora_rank=cfg.kv_lora_rank,
                ),
                tp_size,
                tp_rank,
            )
            weights[W.mla_vc] = _split_attn_o(
                transpose_slice_v(
                    [kv_b],
                    head_num=cfg.head_num,
                    nope_head_dim=cfg.nope_head_dim,
                    v_head_dim=cfg.v_head_dim,
                    lora_rank=cfg.kv_lora_rank,
                ),
                tp_size,
                tp_rank,
            )
        return weights

    def _build_ffn_weights(
        self, bucket: _WeightBucket, prefix: str, layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        align_size = self._align_size()
        ffn_tp_size = self._ffn_tp_size()
        ffn_tp_rank = self._ffn_tp_rank()
        if layer_idx not in self.model_config.moe_layer_index:
            w1 = _split_ffn_w1_w3(
                transpose_pad(
                    [bucket.take(prefix + "mlp.gate_proj.weight")],
                    align_size=align_size,
                    dim=0,
                ),
                ffn_tp_size,
                ffn_tp_rank,
            )
            w3 = _split_ffn_w1_w3(
                transpose_pad(
                    [bucket.take(prefix + "mlp.up_proj.weight")],
                    align_size=align_size,
                    dim=0,
                ),
                ffn_tp_size,
                ffn_tp_rank,
            )
            return {
                W.ffn_w13: torch.cat([w1, w3], dim=-1).contiguous(),
                W.ffn_w2: _split_ffn_w2(
                    transpose_pad(
                        [bucket.take(prefix + "mlp.down_proj.weight")],
                        align_size=align_size,
                        dim=1,
                    ),
                    ffn_tp_size,
                    ffn_tp_rank,
                ),
            }
        shared_prefix = prefix + "block_sparse_moe.shared_experts."
        shared_w1 = _split_ffn_w1_w3(
            transpose_pad(
                [bucket.take(shared_prefix + "gate_proj.weight")],
                align_size=align_size,
                dim=0,
            ),
            ffn_tp_size,
            ffn_tp_rank,
        )
        shared_w3 = _split_ffn_w1_w3(
            transpose_pad(
                [bucket.take(shared_prefix + "up_proj.weight")],
                align_size=align_size,
                dim=0,
            ),
            ffn_tp_size,
            ffn_tp_rank,
        )
        weights = {
            W.ffn_w13: torch.cat([shared_w1, shared_w3], dim=-1).contiguous(),
            W.ffn_w2: _split_ffn_w2(
                transpose_pad(
                    [bucket.take(shared_prefix + "down_proj.weight")],
                    align_size=align_size,
                    dim=1,
                ),
                ffn_tp_size,
                ffn_tp_rank,
            ),
            W.moe_gate: transpose(
                [bucket.take(prefix + "block_sparse_moe.gate.weight")]
            ),
        }
        selected_experts = self._selected_experts(bucket, prefix, layer_idx)
        logger.info(
            "KimiLinear newloader layer %s building %s MoE experts",
            layer_idx,
            len(selected_experts),
        )
        if self._moe_pure_tp_mode():
            w3 = [
                bucket.take_cpu(
                    f"{prefix}block_sparse_moe.experts.{expert_id}.w3.weight"
                )
                for expert_id in selected_experts
            ]
            w1 = [
                bucket.take_cpu(
                    f"{prefix}block_sparse_moe.experts.{expert_id}.w1.weight"
                )
                for expert_id in selected_experts
            ]
            w2 = [
                bucket.take_cpu(
                    f"{prefix}block_sparse_moe.experts.{expert_id}.w2.weight"
                )
                for expert_id in selected_experts
            ]
            moe_w1 = _split_moe_w1(
                stack_moe_w1(w3 + w1),
                self._attn_tp_size(),
                self._attn_tp_rank(),
                True,
            )
            moe_w2 = _split_moe_w2(
                stack_(w2),
                self._attn_tp_size(),
                self._attn_tp_rank(),
                True,
            )
            moe_w1 = bucket.to_target(moe_w1)
            moe_w2 = bucket.to_target(moe_w2)
        else:
            moe_w1, moe_w2 = self._build_ep_moe_weights(
                bucket, prefix, selected_experts
            )
        exported_device = getattr(self.load_config, "exported_device", None)
        if exported_device is not None:
            moe_w1 = exported_device.shuffle_moe_weight(
                moe_w1, self.load_config.compute_dtype, W.moe_w1
            )
            moe_w2 = exported_device.shuffle_moe_weight(
                moe_w2, self.load_config.compute_dtype, W.moe_w2
            )
        weights[W.moe_w1] = moe_w1
        weights[W.moe_w2] = moe_w2
        bias = bucket.maybe_take(
            prefix + "block_sparse_moe.gate.e_score_correction_bias", force_fp32=True
        )
        if bias is not None:
            weights[W.e_score_correction_b] = bias
        return weights

    def _build_ep_moe_weights(
        self, bucket: _WeightBucket, prefix: str, selected_experts: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w3_name = f"{prefix}block_sparse_moe.experts.{selected_experts[0]}.w3.weight"
        w2_name = f"{prefix}block_sparse_moe.experts.{selected_experts[0]}.w2.weight"
        first_w3 = bucket.take_cpu(w3_name)
        first_w2 = bucket.take_cpu(w2_name)
        target = bucket.device
        if target is None:
            return (
                stack_moe_w1(
                    [
                        bucket.take_cpu(
                            f"{prefix}block_sparse_moe.experts.{expert_id}.w3.weight"
                        )
                        for expert_id in selected_experts
                    ]
                    + [
                        bucket.take_cpu(
                            f"{prefix}block_sparse_moe.experts.{expert_id}.w1.weight"
                        )
                        for expert_id in selected_experts
                    ]
                ),
                stack_(
                    [
                        bucket.take_cpu(
                            f"{prefix}block_sparse_moe.experts.{expert_id}.w2.weight"
                        )
                        for expert_id in selected_experts
                    ]
                ),
            )

        moe_w1 = torch.empty(
            [len(selected_experts), first_w3.shape[0] * 2, first_w3.shape[1]],
            dtype=bucket.dtype,
            device=target,
        )
        moe_w2 = torch.empty(
            [len(selected_experts)] + list(first_w2.shape),
            dtype=bucket.dtype,
            device=target,
        )
        for local_idx, expert_id in enumerate(selected_experts):
            if local_idx == 0:
                w3 = first_w3
                w2 = first_w2
            else:
                w3 = bucket.take_cpu(
                    f"{prefix}block_sparse_moe.experts.{expert_id}.w3.weight"
                )
                w2 = bucket.take_cpu(
                    f"{prefix}block_sparse_moe.experts.{expert_id}.w2.weight"
                )
            w1 = bucket.take_cpu(
                f"{prefix}block_sparse_moe.experts.{expert_id}.w1.weight"
            )
            moe_w1[local_idx, : first_w3.shape[0], :].copy_(w3)
            moe_w1[local_idx, first_w3.shape[0] :, :].copy_(w1)
            moe_w2[local_idx].copy_(w2)
        return moe_w1, moe_w2

    def _selected_experts(
        self, bucket: _WeightBucket, prefix: str, layer_idx: int
    ) -> List[int]:
        expert_count = self.model_config.expert_num
        getter = getattr(self.load_config, "get_selected_experts", None)
        if getter is not None:
            selected = list(getter(layer_idx, expert_count))
        else:
            selected = list(range(expert_count))

        available = set()
        expert_prefix = prefix + "block_sparse_moe.experts."
        suffix = ".w3.weight"
        for name in bucket.tensors.keys():
            if not name.startswith(expert_prefix) or not name.endswith(suffix):
                continue
            expert_id_part = name[len(expert_prefix) : -len(suffix)]
            if expert_id_part.isdigit():
                available.add(int(expert_id_part))
        if not available:
            return selected
        filtered = [expert_id for expert_id in selected if expert_id in available]
        if filtered:
            return filtered
        return sorted(available)

    def _parallelism(self):
        return getattr(self.load_config, "parallelism_config", None)

    def _attn_tp_size(self) -> int:
        pc = self._parallelism()
        if pc is not None:
            return pc.get_attn_tp_size()
        return getattr(self.load_config, "tp_size", 1)

    def _attn_tp_rank(self) -> int:
        pc = self._parallelism()
        if pc is not None:
            return pc.get_attn_tp_rank()
        return getattr(self.load_config, "tp_rank", 0)

    def _ffn_tp_size(self) -> int:
        pc = self._parallelism()
        if pc is not None:
            return pc.get_ffn_tp_size()
        return getattr(self.load_config, "ffn_tp_size", self._attn_tp_size())

    def _ffn_tp_rank(self) -> int:
        pc = self._parallelism()
        if pc is not None:
            return pc.get_ffn_tp_rank()
        return getattr(self.load_config, "ffn_tp_rank", self._attn_tp_rank())

    def _lm_head_tp_size(self) -> int:
        return getattr(self.load_config, "lm_head_tp_size", self._attn_tp_size())

    def _lm_head_tp_rank(self) -> int:
        return getattr(self.load_config, "lm_head_tp_rank", self._attn_tp_rank())

    def _moe_pure_tp_mode(self) -> bool:
        pc = self._parallelism()
        if pc is not None:
            return pc.get_attn_tp_size() > 1 and pc.dp_size == 1 and pc.ep_size == 1
        return (
            getattr(self.load_config, "tp_size", 1) > 1
            and getattr(self.load_config, "dp_size", 1) == 1
            and getattr(self.load_config, "ep_size", 1) == 1
        )

    def _align_size(self) -> int:
        quant_algo = getattr(self.model_config, "quant_algo", None)
        use_swizzle = getattr(self.model_config, "use_swizzleA", False)
        if quant_algo is not None and (quant_algo.isQuant() or use_swizzle):
            if quant_algo.isGroupwise():
                return self._attn_tp_size() * quant_algo.getGroupSize()
            return self._attn_tp_size() * 64
        return 0
