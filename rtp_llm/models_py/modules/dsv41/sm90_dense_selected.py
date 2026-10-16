"""Explicit, bounded Humming candidate for the two V4.1 shared projections.

This is FP8/g32 *storage* with FP16 MMA or BF16 WGMMA, not native FP8 arithmetic.
It preserves the original quantized operands/scales and FP32 accumulation.
The selection is based on independent H20 component tests at M=1/16/512/4096;
other M values use bounded interval choices, not a claim of measured optimality.
Nothing in this module changes the default model dispatch.
"""

import dataclasses
import json

import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    expand_block32_scales,
    unpack_ue8m0_input_scales,
)


SUPPORTED_SHARED_SHAPES = frozenset(((4608, 5120), (5120, 2304)))


def supports_two_shared(n: int, k: int) -> bool:
    return (n, k) in SUPPORTED_SHARED_SHAPES


class Sm90SelectedG32Linear:
    """Candidate selecting gated Humming tiles, never native FP8 WGMMA.

    MMA weights are transformed directly from checkpoint bytes using a genuine
    MMA LayerConfig. They are never borrowed from a WGMMA-packed allocation.
    Large M uses a second FP8 weight packing for BF16 WGMMA arithmetic.
    There is no secondary Triton packing or global BF16 weight expansion.
    Warm each selected tile outside graph capture before capturing calls.
    """

    supports_two_shared = staticmethod(supports_two_shared)

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        if (not weight.is_cuda
                or torch.cuda.get_device_capability(weight.device) != (9, 0)):
            raise ValueError("Selected shared g32 candidate requires a CUDA SM90 device")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Expected two-dimensional E4M3 checkpoint weights")
        self.N, self.K = weight.shape
        if not supports_two_shared(self.N, self.K):
            raise ValueError(f"Unsupported shared projection {(self.N, self.K)}")
        if (scales.device != weight.device or scales.dtype != torch.float8_e8m0fnu
                or tuple(scales.shape) != (self.N // 32, self.K // 32)):
            raise ValueError("Expected same-device UE8M0 32x32 checkpoint scales")
        from humming.forward import humming_forward
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        self.device = weight.device
        self._forward = humming_forward
        original = prepare_layer_config(
            shape_n=self.N, shape_k=self.K,
            weight_schema=HummingWeightSchema(
                b_dtype="float8e4m3", bs_dtype="float8e8m0",
                weight_scale_group_size=32, weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="float8e4m3", input_scale_dtype="float32",
                input_scale_group_size=32, input_quant_mode="dynamic_group",
            ),
            pad_n_to_multiple=256, pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        self.config = dataclasses.replace(original, mma_type="mma", use_packed_k_layout=False)
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError("Selected g32 candidate requires lossless original UE8M0 scale layout")
        # The temporary clone protects the source checkpoint from any transform
        # aliasing and is released after preparation. There is no WG transform.
        tensors = transform_humming_tensors(self.config, {
            "weight": weight.contiguous().clone().view(torch.int32),
            "weight_scale": expand_block32_scales(scales, self.N, self.K),
        })
        self.weight = tensors["weight"]
        self.weight_scale = tensors["weight_scale"]
        self.weight_scale_2 = tensors.get("weight_scale_2")
        self.locks = torch.zeros(1024, dtype=torch.int32, device=weight.device)
        self.compute_config = json.dumps(dict(
            gemm_type="dense", use_f16_accum=False,
            use_m_major_input_scale=False, use_batch_invariant=True,
        ))
        num_sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.tuning_configs = {
            tile: json.dumps(dict(
                block_shape=(tile, 128, 64), warp_shape=(min(tile, 32), 32, 64),
                num_stages=3, num_ctas_per_sm=2, use_stream_k=False,
                use_tma=False, use_warp_spec=False, use_mbarrier=False,
                num_sms=num_sms,
            ))
            for tile in (16, 32, 64)
        }
        self._dispatches = {
            tile: dict(backend="humming_mma", config=f"humming_mma_m{tile}",
                       tile_m=tile, prepared_kernel_count=1,
                       arithmetic="FP8_G32_STORAGE_FP16_HMMA_FP32_ACCUM")
            for tile in (16, 32, 64)
        }
        from rtp_llm.models_py.modules.dsv41.sm90_dense_wgmma16 import Sm90Wgmma16Linear

        self._wgmma16 = Sm90Wgmma16Linear(weight, scales)
        self._dispatches["wgmma16"] = dict(
            backend="humming_wgmma16", config="humming_wgmma16_heuristic",
            prepared_kernel_count=2,
            arithmetic="FP8_G32_STORAGE_BF16_WGMMA_FP32_ACCUM",
        )
        self.last_dispatch = None
        self.last_compiled_kernel = None
        self.last_reduce_kernel = None

    def get_dispatch(self, rows: int):
        """Read-only selection evidence; no GPU work or compilation."""
        if rows < 0:
            raise ValueError("row count must be nonnegative")
        if rows <= 16:
            return dict(self._dispatches[16])
        if rows >= 512:
            return dict(self._dispatches["wgmma16"])
        return dict(self._dispatches[64 if self.N == 4608 else 32])

    def _output(self, q, out):
        if (q.ndim < 1 or q.dtype != torch.float8_e4m3fn
                or q.shape[-1] != self.K or q.device != self.device
                or not q.is_contiguous()):
            raise ValueError("Expected same-device contiguous E4M3 activations with matching K")
        rows = q.numel() // self.K
        if max(rows * self.K, rows * self.N) >= 2**31:
            raise ValueError("Candidate requires fewer than 2^31 indexed elements")
        shape = (*q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        elif (out.dtype != torch.bfloat16 or out.device != q.device
              or tuple(out.shape) != shape or not out.is_contiguous()):
            raise ValueError("out must be same-device contiguous BF16 with matching shape")
        inputs = (q, self.weight, self.weight_scale)
        if self.weight_scale_2 is not None:
            inputs += (self.weight_scale_2,)
        inputs += (self._wgmma16.weight, self._wgmma16.weight_scale)
        if any(torch._C._overlaps(out, tensor) for tensor in inputs):
            raise ValueError("output must not overlap a GEMM input")
        return rows, out

    def forward_quantized(self, q: torch.Tensor, packed: torch.Tensor, out=None):
        rows, out = self._output(q, out)
        if (packed.dtype != torch.int32 or packed.device != self.device
                or tuple(packed.shape) != (rows, self.K // 128)):
            raise ValueError("Expected RTP packed group32 UE8M0 activation scales")
        if torch._C._overlaps(out, packed):
            raise ValueError("output must not overlap activation scales")
        if rows == 0:
            return out
        return self.prepared(q, unpack_ue8m0_input_scales(packed), out)

    def prepared(self, q: torch.Tensor, float_scales: torch.Tensor, out, config=None):
        """Execute selected Humming arithmetic; all preparation stays inside."""
        if config is not None:
            raise ValueError("Selected candidate does not accept per-call config overrides")
        rows, out = self._output(q, out)
        if (float_scales.dtype != torch.float32 or float_scales.device != self.device
                or tuple(float_scales.shape) != (rows, self.K // 32)
                or any(stride <= 0 for stride in float_scales.stride())):
            raise ValueError("Expected same-device positive-stride FP32 [M,K/32] scales")
        if torch._C._overlaps(out, float_scales):
            raise ValueError("output must not overlap activation scales")
        if rows == 0:
            return out
        self.last_dispatch = self.get_dispatch(rows)
        self.last_compiled_kernel = None
        self.last_reduce_kernel = None
        if self.last_dispatch["backend"] == "humming_wgmma16":
            return self._wgmma16.prepared(q, float_scales, out)
        tile = self.last_dispatch["tile_m"]
        # Humming uses the row-major FP32 scale ABI; unpacking already
        # produces this layout. Non-contiguous prepared inputs are copied
        # explicitly and this cost remains inside the measured call.
        self._forward(
            self.config, inputs=q.view(rows, self.K),
            input_scale=float_scales.contiguous(),
            weight=self.weight, weight_scale=self.weight_scale,
            weight_scale_2=self.weight_scale_2,
            outputs=out.view(rows, self.N), locks=self.locks,
            compute_config=self.compute_config,
            tuning_config=self.tuning_configs[tile],
        )
        return out


__all__ = ["Sm90SelectedG32Linear", "SUPPORTED_SHARED_SHAPES", "supports_two_shared"]
