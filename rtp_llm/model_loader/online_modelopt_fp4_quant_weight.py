"""Online MODELOPT_FP4 quantization for MoE weights.

This module implements on-the-fly NVFP4 (modelopt_fp4) quantization of BF16/FP16
MoE weights. Non-MoE weights are kept in their original compute dtype because no
QuantWeight class registers support for them in this mode (``is_quanted()`` is
False).

The math follows ModelOptimizer's NVFP4 reference (block_size=16):
    weight_scale_2 = amax(W) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    q4 = round(W / (per_block_scale.fp8 * weight_scale_2))   -> packed e2m1 uint8

An explicit activation_amax supplies the activation dequant global scale;
otherwise the existing 1.0 contract is preserved. Per-block activation scaling
remains dynamic. Online weights are not a calibrated checkpoint.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    QuantizationConfig,
)
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK as MEGA_MOE_FP4_BLOCK
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    convert_fp8_moe_to_fp4_ue8m0,
    quantize_moe_weight_to_fp4_ue8m0,
)
from rtp_llm.utils.model_weight import W, identity

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16
_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)


class _Fp8ExpertTensorSource(TensorSource):
    """Decode one checkpoint expert at a time before online NVFP4 conversion."""

    def __init__(self, base: TensorSource, block_size: int):
        self.base = base
        self.block_size = block_size

    def load_tensor(self, name, data_type=torch.bfloat16):
        scale_name = name + "_scale_inv"
        if not self.base.has_tensor(scale_name):
            raise ValueError(f"FP8 expert is missing its block scale: {name}")
        weights = self.base.load_tensor(name, torch.float8_e4m3fn)
        scales = self.base.load_tensor(scale_name, torch.float32)
        if len(weights) != len(scales):
            raise ValueError("FP8 expert weight/scale source counts disagree")
        result = []
        for weight, scale in zip(weights, scales):
            if weight.ndim != 2 or scale.ndim != 2:
                raise ValueError("FP8 expert weight and block scale must be matrices")
            expected = tuple(
                (size + self.block_size - 1) // self.block_size for size in weight.shape
            )
            if tuple(scale.shape) != expected:
                raise ValueError("FP8 expert block scale geometry disagrees")
            expanded = scale.repeat_interleave(self.block_size, 0).repeat_interleave(
                self.block_size, 1
            )[: weight.shape[0], : weight.shape[1]]
            result.append((weight.float() * expanded).to(data_type))
        return result

    def has_tensor(self, name):
        return self.base.has_tensor(name)

    def get_database(self):
        return self.base.get_database()


_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_ODD_BOUNDS = _E2M1_BOUNDS[[1, 3, 5]]  # banker-rounding ties


def _cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map float values in [-6, 6] to packed E2M1 ordinals (uint8 in [0, 15])."""
    device = x.device
    sign_bit = (x < 0).to(torch.uint8)
    abs_x = x.abs()
    bounds = _E2M1_BOUNDS.to(device=device, dtype=abs_x.dtype)
    ord_ = torch.searchsorted(bounds, abs_x, out_int32=True).to(torch.uint8)
    odd_bounds = _E2M1_ODD_BOUNDS.to(device=device, dtype=abs_x.dtype)
    equals_odd = torch.any(abs_x.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord_ + equals_odd


def quantize_expert_to_nvfp4(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
):
    """Quantize a 2D BF16/FP16 weight tensor to NVFP4 (per-block fp8 + per-tensor fp32).

    Args:
        weight: shape ``[N, K]`` with K divisible by ``block_size``.

    Returns:
        Tuple of (packed uint8 ``[N, K // 2]``, per-block fp8 scale ``[N, K // block_size]``,
        per-tensor fp32 scale_2 scalar).
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

    w_fp32 = weight.to(torch.float32)
    amax = w_fp32.abs().amax().clamp_min(1e-8)
    weight_scale_2 = (amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)).to(torch.float32)

    blocked = w_fp32.view(N, K // block_size, block_size)
    per_block_amax = blocked.abs().amax(dim=-1)  # [N, K//block_size]
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    per_block_scale = torch.where(
        per_block_scale == 0,
        torch.ones_like(per_block_scale),
        per_block_scale,
    )
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    # Re-dequant fp8 scale to use the same value the kernel will see.
    effective_block_scale = per_block_scale_fp8.to(torch.float32)
    combined = (effective_block_scale * weight_scale_2).unsqueeze(-1)  # [N, K//bs, 1]
    scaled = blocked / combined
    scaled = scaled.clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(N, K)

    q = _cast_to_e2m1(scaled)  # uint8 [N, K]
    packed = (q[..., 1::2].to(torch.uint8) << 4) | q[..., 0::2].to(torch.uint8)
    packed = packed.contiguous()  # [N, K // 2]
    return packed, per_block_scale_fp8.contiguous(), weight_scale_2


def _moe_kernel_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1
    if name == W.moe_w2:
        return W.moe_w2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_2_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_s2
    if name == W.moe_w2:
        return W.moe_w2_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_input_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_i_s
    if name == W.moe_w2:
        return W.moe_w2_i_s
    raise ValueError(f"unsupported moe weight: {name}")


class OnlineModelOptFp4MoeWeight(CompositeWeight, QuantWeight):
    """Online NVFP4 weight loader for MoE ``moe_w1``/``moe_w2`` only."""

    moe_weight_list = [W.moe_w1, W.moe_w2]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, ModelOptFp4Config):
            return False
        if quant_config.is_quanted():
            return False
        if not isinstance(src_weight_info, MoeAtomicWeight):
            return False
        return src_weight_info.name in cls.moe_weight_list

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        # The kernel sub-weight inherits the original loader so BF16 weights are
        # read from the checkpoint via the existing MoE path (stack_moe_w1 etc.).
        kernel = MoeAtomicWeight(
            name=_moe_kernel_name(src_weight_info.name),
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,  # use load_config.compute_dtype (bf16/fp16)
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        # Companion sub-weights are synthesized in _load_raw_tensor; their
        # ``weights=[]`` keeps them out of the file-name listing path.
        scale = MoeAtomicWeight(
            name=_moe_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = MoeAtomicWeight(
            name=_moe_scale_2_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = MoeAtomicWeight(
            name=_moe_input_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        sub_weights = {
            kernel.name: kernel,
            scale.name: scale,
            scale_2.name: scale_2,
            input_scale.name: input_scale,
        }
        super().__init__(
            sub_weights,
            quant_config=quant_config,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self.scale_2 = scale_2
        self.input_scale = input_scale
        self._block_size = max(quant_config.group_size(), NVFP4_BLOCK_SIZE)
        self._source_fp8_block_size = getattr(
            quant_config, "source_fp8_block_size", None
        )
        if self._source_fp8_block_size not in (None, 128):
            raise ValueError("online NVFP4 supports only block128 FP8 source weights")
        self._cache = None
        if quant_config.online_cache_dir:
            from rtp_llm.model_loader.nvfp4_cache import Nvfp4ExpertCache

            self._cache = Nvfp4ExpertCache(quant_config.online_cache_dir)
        activation_amax = quant_config.activation_amax
        self._input_scale = (
            1.0
            if activation_amax is None
            else float(activation_amax) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
        )
        if not 0 < self._input_scale < float("inf"):
            raise ValueError("online NVFP4 activation amax must be finite and positive")

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        # Only the kernel sub-weight reads tensors from the checkpoint.
        names = self.kernel.get_tensor_names(layer_id, load_config)
        if self._source_fp8_block_size is not None:
            names |= {name + "_scale_inv" for name in names}
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        # Keep the temporary BF16 allocation bounded to one expert while
        # previously quantized layers remain resident.
        selected = load_config.get_selected_experts(
            layer_id, self.kernel.config.expert_num
        )
        if not selected:
            raise ValueError("online NVFP4 requires at least one local expert")
        if self.quant_config.require_contiguous_experts:
            count = self.kernel.config.expert_num // load_config.ep_size
            expected = list(
                range(count * load_config.ep_rank, count * (load_config.ep_rank + 1))
            )
            if list(selected) != expected:
                raise ValueError(
                    "CUTLASS NVFP4 requires contiguous logical expert ownership"
                )
        if self.kernel.stacked_ckpt_keys and tensor_source.has_tensor(
            self.kernel.weights[0].tensor_name(layer_id)
        ):
            # Share the CPU stacked cache across expert reads. Hide raw keys
            # from the nested loader so it consumes logical slices directly.
            class ExpertTensorSource(StackSplitTensorSource):
                def has_tensor(self, name):
                    if name in self._split_config:
                        return super().has_tensor(name)
                    return False

            tensor_source = ExpertTensorSource(
                tensor_source, self.kernel._build_split_config(layer_id, load_config)
            )
        if self._source_fp8_block_size is not None:
            if self.kernel.stacked_ckpt_keys:
                raise ValueError("FP8 source requires split expert checkpoint keys")
            tensor_source = _Fp8ExpertTensorSource(
                tensor_source, self._source_fp8_block_size
            )
        output = None
        for expert_index, expert_id in enumerate(selected):

            class ExpertLoadConfig:
                def __getattr__(self, name):
                    return getattr(load_config, name)

                def get_selected_experts(self, requested_layer, expert_count):
                    return [expert_id]

            kernel_dict = self.kernel._load_raw_tensor(
                tensor_source, layer_id, device, ExpertLoadConfig()
            )
            weight = kernel_dict[self.kernel.name]
            if weight.ndim != 3 or weight.shape[0] != 1:
                raise ValueError("online NVFP4 expected exactly one expert")
            if self._cache is None:
                packed, scale, scale_2 = quantize_expert_to_nvfp4(
                    weight[0], self._block_size
                )
            else:
                packed, scale, scale_2 = self._cache.get_or_quantize(
                    weight[0],
                    block_size=self._block_size,
                    input_scale=self._input_scale,
                    identity=dict(
                        layer=layer_id,
                        expert=int(expert_id),
                        weight=self.kernel.name,
                        source_templates=[item.name for item in self.kernel.weights],
                        **(
                            {"source_format": "fp8-block128"}
                            if self._source_fp8_block_size is not None
                            else {}
                        ),
                    ),
                )
                scale_2 = scale_2.reshape(())
            if output is None:
                output = {
                    self.kernel.name: torch.empty(
                        (len(selected), *packed.shape),
                        dtype=packed.dtype,
                        device=packed.device,
                    ),
                    self.scale.name: torch.empty(
                        (len(selected), *scale.shape),
                        dtype=scale.dtype,
                        device=scale.device,
                    ),
                    self.scale_2.name: torch.empty(
                        len(selected), dtype=torch.float32, device=packed.device
                    ),
                    # This explicit activation range is not calibration.
                    self.input_scale.name: torch.full(
                        (len(selected),),
                        self._input_scale,
                        dtype=torch.float32,
                        device=packed.device,
                    ),
                }
            output[self.kernel.name][expert_index].copy_(packed)
            output[self.scale.name][expert_index].copy_(scale)
            output[self.scale_2.name][expert_index].copy_(scale_2)
            del weight, kernel_dict, packed, scale, scale_2
        return output

    def _postprocess(
        self,
        tensor,
        device: str,
        load_config: LoadConfig,
    ):
        processed = super()._postprocess(tensor, device, load_config)
        kernel_w = processed[self.kernel.name]
        scale_w = processed[self.scale.name]
        kernel_w, scale_w = (
            load_config.exported_device.maybe_prepare_static_weights_for_fp4_moe(
                self.kernel.name,
                self.scale.name,
                kernel_w,
                scale_w,
            )
        )
        processed[self.kernel.name] = kernel_w
        processed[self.scale.name] = scale_w
        return processed


def _mega_moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


class OnlineMegaMoeFp4Weight(CompositeWeight, QuantWeight):
    """Load-time FP4+UE8M0 quantizer for mega-MoE stacked MoE weights
    (``moe_w1`` preserves its source stacked order, ``moe_w2`` = down)
    starting from BF16.

    The common ``MegaMoeExecutor`` consumes the packed weights and scales.
    Converting during loading releases the BF16 tensor before executor setup.

    Wired in by ``apply_mega_moe_fp4_wrappers`` (in ``model_weight_info.py``)
    when a MegaMoE strategy is selected. Inherits ``QuantWeight`` purely so that
    ``MoeWeight``'s sub-weight assertion passes; the ``support()`` classmethod
    is gated to never fire from the regular ``QuantWeight.create()`` registry
    (the explicit wrap pass is the only entry point).
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False  # explicit-wiring only; never auto-selected

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        block_size: int = MEGA_MOE_FP4_BLOCK,
        **kwargs: Any,
    ):
        if src_weight_info.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoeFp4Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_weight_info.name}"
            )

        kernel = MoeAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,  # follow load_config.compute_dtype (BF16)
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_weight_info.name),
            weights=[],  # synthesised in _load_raw_tensor
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self._block_size = block_size

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight = kernel_dict[self.kernel.name]
        packed, sf = quantize_moe_weight_to_fp4_ue8m0(weight, self._block_size)
        del weight
        kernel_dict[self.kernel.name] = packed
        kernel_dict[self.scale.name] = sf
        return kernel_dict

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        # No layout transform here: the deepgemm SF layout + mega-moe transform
        # require the (l1, l2) pair together and run in MegaMoeExecutor setup.
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }


class OnlineMegaMoeFp4FromFp8Weight(CompositeWeight, QuantWeight):
    """Load-time FP8 per-block → FP4+UE8M0 quantizer for mega-MoE.

    Takes FP8 per-block quantized weights (``float8_e4m3fn`` kernel +
    ``float32`` scale) and produces FP4 packed int8 + fp32 UE8M0 scale at
    load time for the common ``MegaMoeExecutor`` to consume directly.

    Wired in by ``apply_mega_moe_fp4_wrappers`` when a MegaMoE strategy is selected
    and the original MoE weight was a ``PerBlockFp8Weight``.
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False  # explicit-wiring only

    def __init__(
        self,
        src_kernel: MoeAtomicWeight,
        src_scale: MoeAtomicWeight,
        block_size: int = MEGA_MOE_FP4_BLOCK,
        **kwargs: Any,
    ):
        if src_kernel.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoeFp4FromFp8Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_kernel.name}"
            )

        # Sub-weight that loads the FP8 kernel from the checkpoint.
        kernel = MoeAtomicWeight(
            name=src_kernel.name,
            weights=src_kernel.weights,
            process_fun=src_kernel.process_fun,
            data_type=torch.float8_e4m3fn,
            config=src_kernel.config,
            stacked_ckpt_keys=getattr(src_kernel, "stacked_ckpt_keys", False),
        )
        # Sub-weight that loads the FP8 per-block scale; renamed with an
        # ``_fp8_input`` suffix so it does not collide with the FP4 output
        # scale tensor we emit.
        fp8_scale_name = _mega_moe_scale_name(src_kernel.name) + "_fp8_input"
        fp8_scale = MoeAtomicWeight(
            name=fp8_scale_name,
            weights=src_scale.weights,
            process_fun=src_scale.process_fun,
            data_type=torch.float32,
            config=src_scale.config,
            stacked_ckpt_keys=getattr(src_scale, "stacked_ckpt_keys", False),
        )
        # Output FP4 scale (synthesised in _load_raw_tensor).
        out_scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_kernel.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_kernel.config,
        )

        sub_weights = {
            kernel.name: kernel,
            fp8_scale.name: fp8_scale,
            out_scale.name: out_scale,
        }
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_kernel.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.fp8_scale = fp8_scale
        self.out_scale = out_scale
        self._block_size = block_size

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = self.kernel.get_tensor_names(layer_id, load_config)
        names |= self.fp8_scale.get_tensor_names(layer_id, load_config)
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        scale_dict = self.fp8_scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight_fp8 = kernel_dict[self.kernel.name]
        weight_scale = scale_dict[self.fp8_scale.name]

        packed, sf = convert_fp8_moe_to_fp4_ue8m0(
            weight_fp8, weight_scale, self._block_size
        )
        del weight_fp8, weight_scale
        return {
            self.kernel.name: packed,
            self.out_scale.name: sf,
        }

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.out_scale._split(
            {self.out_scale.name: tensor[self.out_scale.name]}, load_config
        )
        out: Dict[str, torch.Tensor] = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.out_scale.name: tensor[self.out_scale.name],
        }


from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    create_w8a8_fp8_per_block_weight,
)


class OnlineModelOptFp4HybridFp8AttnWeight(LoadQuantPerBlockFp8Weight):
    """FP8_PER_BLOCK loader used by ``MODELOPT_FP4`` hybrid mode for non-MoE weights."""

    _excluded_names = (W.moe_w1, W.moe_w2, W.mla_kc, W.mla_vc)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, ModelOptFp4Config):
            return False
        if quant_config.is_quanted():
            return False
        if getattr(quant_config, "hybrid_attn_quant_method", None) != "FP8_PER_BLOCK":
            return False
        if isinstance(src_weight_info, MoeAtomicWeight):
            return False
        name = src_weight_info.name
        if name in cls._excluded_names:
            return False
        return name in cls.w8a8_weight_list

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        # FP4 quant_config has group_size=16; for the FP8 block-wise path used by
        # attention weights we lock to the canonical FP8 per-block size (128).
        self.group_size = Fp8BlockWiseQuantConfig.DEFAULT_FP8_QUANT_BLOCK_SIZE
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )
        kernel: AtomicWeight = create_w8a8_fp8_per_block_weight(
            src_weight_info, **params
        )
        sub_weights = {kernel.name: kernel}
        scale_name = self.w8a8_weight_list.get(src_weight_info.name)
        scale = None
        if scale_name:
            scale_params = {**params}
            scale_params["name"] = scale_name
            scale = create_w8a8_fp8_per_block_weight(src_weight_info, **scale_params)
            sub_weights.update({scale.name: scale})

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return self.kernel.get_tensor_names(layer_id, load_config)


def wrap_moe_for_mega_moe(weight: WeightModule) -> WeightModule:
    """If ``weight`` is a MoE atomic / per-block FP8 wrapper for moe_w1/moe_w2,
    return a load-time FP4 quantizer that consumes BF16 or FP8 source. Otherwise
    return the weight unchanged.

    This is the single entry point used by ``apply_mega_moe_fp4_wrappers`` in
    ``model_weight_info.py``.
    """
    from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight

    if isinstance(weight, PerBlockFp8Weight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        kernel = weight.kernel
        scale = weight.scale
        if kernel is None or scale is None:
            return weight
        return OnlineMegaMoeFp4FromFp8Weight(
            src_kernel=kernel,
            src_scale=scale,
        )
    if isinstance(weight, MoeAtomicWeight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        return OnlineMegaMoeFp4Weight(weight)
    return weight
