"""ModelSlim pre-quantized (W8A8_MXFP8) weight loader for Ascend NPU.

Zero re-quantization, straight-through loading:
  * kernel: source structure verbatim (names/merge/process funs incl. stacked
    MoE), only data_type=float8_e4m3fn; raw layout same as the bf16 path.
  * scale: same structure with ModelSlim scale names, uint8 (E8M0); loads as
    [kp, N] / [E, N, kp] (all involved funs are pure layout ops).
  * _load_raw_tensor normalizes 2D to [N, K]/[N, kp] (MoE 3D untouched) for
    PerBlockFp8Weight._postprocess; swizzle runs there, after the TP split.

MUST NOT import torch_npu at top level (imported on every platform via
model_loader/__init__.py).
"""

import functools
import re
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import AscendW8A8MXFP8Config, QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    create_w8a8_fp8_per_block_weight,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight, CompositeWeight, WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W

# ModelSlim scale suffix (verified against a real ModelSlim export).
ASCEND_W8A8_MXFP8_SCALE_SUFFIX = ".weight_scale"

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)

# Weight families verified for the .T normalization (transpose-family
# process_fun outputs [K, N]; identity-family entries in w8a8_weight_list
# would be double-flipped). Keep in sync with test coverage.
_VERIFIED_W8A8_MXFP8_WEIGHTS = frozenset(
    [
        W.attn_qkv_w,
        W.attn_gate_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w2,
        W.ffn_w3,
        W.ffn_w13,
        W.moe_w1,
        W.moe_w2,
        W.linear_attn_qkvz_w,
    ]
)


def _ascend_w8a8_mxfp8_scale_ckpt_info(w: CkptWeightInfo) -> CkptWeightInfo:
    """Derive the ModelSlim scale CkptWeightInfo from a kernel one."""
    name = w.name
    if name.endswith(".weight"):
        name = name[: -len(".weight")] + ASCEND_W8A8_MXFP8_SCALE_SUFFIX
    else:
        # stacked MoE key without ".weight" suffix (e.g. "...experts.gate_up_proj")
        name = name + ASCEND_W8A8_MXFP8_SCALE_SUFFIX
    return CkptWeightInfo(name, w.merge_fun)


def _adapt_scale_align(
    fn: Callable[[List[torch.Tensor]], torch.Tensor], group_size: int
) -> Callable[[List[torch.Tensor]], torch.Tensor]:
    """Adapt a kernel-side process_fun for the scale tensor.

    Kernel align_size pads K in kernel elements; the scale holds kp = K/32
    groups on that dim, so its align must be kernel align // group_size
    (same convention as PerBlockFp8Weight._get_ffn_quant_weight). Otherwise
    e.g. down_proj (align_size=64) pads scale kp 96->128 while kernel K
    stays 3072 — rejected by aclnnQuantMatmulV5 mx mode. Padding on other
    dims (e.g. N-side transpose_pad) is group-agnostic and harmless.
    """
    if (
        isinstance(fn, functools.partial)
        and fn.keywords
        and fn.keywords.get("align_size")
    ):
        align_size = fn.keywords["align_size"]
        if align_size % group_size != 0:
            raise ValueError(
                f"kernel align_size {align_size} not divisible by group_size "
                f"{group_size}; scale padding invariant would break"
            )
        keywords = dict(fn.keywords)
        keywords["align_size"] = align_size // group_size
        return functools.partial(fn.func, *fn.args, **keywords)
    return fn


def _as_uint8_layout_fn(fn: Callable[[List[torch.Tensor]], torch.Tensor]):
    """Run a pure-layout function on uint8 views of fp8 inputs.

    torch_npu does not support fp8 cat/stack/contiguous; layout-only ops
    (transpose/concat/stack/slice) on uint8 views are byte-equivalent.
    Same trick as utils/model_weight.py::concat_0/concat_1.
    """

    @functools.wraps(fn)
    def wrapped(ts: List[torch.Tensor]) -> torch.Tensor:
        tensors = list(ts)
        fp8_flags = [t is not None and t.dtype in _FP8_DTYPES for t in tensors]
        if not any(fp8_flags):
            return fn(tensors)
        fp8_dtype = next(t.dtype for t, f in zip(tensors, fp8_flags) if f)
        u8_tensors = [
            t.view(torch.uint8) if f else t for t, f in zip(tensors, fp8_flags)
        ]
        out = fn(u8_tensors)
        if isinstance(out, torch.Tensor) and out.dtype == torch.uint8:
            if not out.is_contiguous():
                out = out.contiguous()
            out = out.view(fp8_dtype)
        return out

    return wrapped


def _template_matches_excludes(name_template: str, excludes: set) -> bool:
    """Match a ckpt name template (with ``{i}``/``{expert_id}`` placeholders)
    against concrete excluded module paths from quant_model_description.json.
    """
    if not excludes:
        return False
    if name_template in excludes:
        return True
    parts = re.split(r"\{i\}|\{expert_id\}", name_template)
    pattern = "^" + r"\d+".join(re.escape(p) for p in parts) + "$"
    rx = re.compile(pattern)
    return any(rx.match(ex) for ex in excludes)


class AscendW8A8MXFP8Weight(PerBlockFp8Weight):
    """ModelSlim pre-quantized (W8A8_MXFP8, 1x32 group along K) weight loader.

    Inherits from PerBlockFp8Weight: w8a8_weight_list, _postprocess NPU
    branch (real transpose + E8M0 swizzle), W8A8Fp8PerBlock* TP splits.
    """

    # Type marker gating the NPU MXFP8 layout branch in the base
    # _postprocess (transpose + E8M0 swizzle), instead of is_ascend().
    _use_npu_mxfp8_layout = True

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, AscendW8A8MXFP8Config) or not quant_config.is_quanted():
            return False
        name = src_weight_info.name
        if name not in cls.w8a8_weight_list or name in [W.mla_kc, W.mla_vc]:
            return False
        unverified = name not in _VERIFIED_W8A8_MXFP8_WEIGHTS
        # FLOAT-marked modules keep bf16 loading. Concrete names and MoE
        # expert templates are resolvable here; {i} templates are shared by
        # all layers, so partial-quant layers are decided per layer at load
        # time (_is_bf16_fallback).
        # Exclude (FLOAT-marked) check must precede the unverified raise
        # below: such weights (e.g. qwen3.5 linear_attn.out_proj) stay on
        # the bf16 path and never enter fp8 loading.
        if quant_config.exclude_modules and hasattr(src_weight_info, "weights"):
            for ckpt_w in src_weight_info.weights:
                if "{i}" in ckpt_w.name and "{expert_id}" not in ckpt_w.name:
                    if unverified and _template_matches_excludes(
                        ckpt_w.name, quant_config.exclude_modules
                    ):
                        return False  # unverified layout + FLOAT layers → bf16
                    continue
                if _template_matches_excludes(
                    ckpt_w.name, quant_config.exclude_modules
                ):
                    return False
        if unverified:
            raise ValueError(
                f"AscendW8A8MXFP8 does not support weight {name!r} yet "
                "(quantized weight with unverified layout)"
            )
        return True

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        self.group_size = quant_config.group_size()  # == 32 for MXFP8
        self.quant_config = quant_config
        self.src_weight_info = src_weight_info
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )

        # kernel: same structure as the source weight, fp8 dtype
        kernel_params = {**params, "data_type": torch.float8_e4m3fn}
        kernel: AtomicWeight = create_w8a8_fp8_per_block_weight(
            src_weight_info, **kernel_params
        )
        # wrap layout ops on uint8 views (post-construction so that
        # MoeAtomicWeight.__init__ introspection ran on the original fun)
        kernel.process_fun = _as_uint8_layout_fn(kernel.process_fun)
        kernel.weights = [
            CkptWeightInfo(w.name, _as_uint8_layout_fn(w.merge_fun))
            for w in src_weight_info.weights
        ]
        sub_weights = {kernel.name: kernel}

        scale: Optional[AtomicWeight] = None
        scale_name = self.w8a8_weight_list.get(src_weight_info.name)
        if scale_name:
            scale_params = {**params, "name": scale_name, "data_type": torch.uint8}
            scale_params["weights"] = [
                _ascend_w8a8_mxfp8_scale_ckpt_info(w) for w in src_weight_info.weights
            ]
            scale_params["process_fun"] = _adapt_scale_align(
                params.get("process_fun"), self.group_size
            )
            scale = create_w8a8_fp8_per_block_weight(src_weight_info, **scale_params)
            sub_weights[scale.name] = scale

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def _is_bf16_fallback(self, layer_id: Optional[int]) -> bool:
        """True when this concrete layer is excluded (kept FLOAT) in the ckpt:
        such layers carry a bf16 weight and no scale tensor, so they bypass
        the fp8 path and reuse the original bf16 structure.
        """
        if layer_id is None:
            return False
        exclude_modules = getattr(self.quant_config, "exclude_modules", None)
        if not exclude_modules:
            return False
        for ckpt_w in self.src_weight_info.weights:
            if "{expert_id}" in ckpt_w.name:
                # expert-granular exclusion is not resolved at this level
                return False
            if ckpt_w.tensor_name(layer_id) in exclude_modules:
                return True
        return False

    def load(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        if self._is_bf16_fallback(layer_id):
            return self.src_weight_info.load(
                tensor_source, layer_id, device, load_config
            )
        return super().load(tensor_source, layer_id, device, load_config)

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set:
        if self._is_bf16_fallback(layer_id):
            return self.src_weight_info.get_tensor_names(layer_id, load_config)
        return super().get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        if self._is_bf16_fallback(layer_id):
            # Nested composites (e.g. FfnWeight) drive the stages via
            # _load_raw_tensor/_split/_postprocess directly, bypassing load();
            # remember the decision (instances are per-layer, state is safe).
            self._bf16_fallback_active = True
            return self.src_weight_info._load_raw_tensor(
                tensor_source, layer_id, device, load_config
            )
        self._bf16_fallback_active = False
        kernel_res = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        res: Dict[str, torch.Tensor] = {}
        kernel = kernel_res.get(self.kernel.name)
        if kernel is not None:
            # [K, N] -> [N, K] (MXFP8 groups along K); MoE 3D needs no transpose.
            if kernel.dim() == 2:
                kernel = kernel.T.contiguous()
            res[self.kernel.name] = kernel.to(device)

        if self.scale is not None:
            scale_res = self.scale._load_raw_tensor(
                tensor_source, layer_id, device, load_config
            )
            scale = scale_res.get(self.scale.name)
            if scale is not None:
                # same transpose-equivariant normalization: [kp, N] -> [N, kp]
                if scale.dim() == 2:
                    scale = scale.T.contiguous()
                res[self.scale.name] = scale.to(device)

        return res

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        if getattr(self, "_bf16_fallback_active", False):
            return self.src_weight_info._split(tensor, load_config)
        return super()._split(tensor, load_config)

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        if getattr(self, "_bf16_fallback_active", False):
            return self.src_weight_info._postprocess(tensor, device, load_config)
        return super()._postprocess(tensor, device, load_config)

    # Non-fallback _postprocess: inherited (NPU branch: real transpose +
    # scale swizzle); _split: inherited from PerBlockFp8Weight/CompositeWeight
