"""Shared helpers for the W8A8_MXFP8 tests (Ascend NPU, needs torch_npu).

Imported by:
  test_w8a8_mxfp8_weight_load.py    synthetic-ckpt loading chain
  test_w8a8_mxfp8_real_ckpt.py      real ModelSlim ckpt validation
  test_w8a8_mxfp8_moe_executor.py   MoE executor numerics
"""

import json
import os
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.w8a8_mxfp8_weight import AscendW8A8MXFP8Weight
from rtp_llm.model_loader.weight_module import WeightModule

# Import side effect: all consumers run on NPU device 0.
torch.npu.set_device(0)

MXFP8_GROUP = 32
MXFP8_DTYPE = torch.float8_e4m3fn


class FakeTensorSource(TensorSource):
    """In-memory TensorSource over a {name: tensor} dict."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def load_tensor(self, name: str, data_type=torch.float16) -> List[torch.Tensor]:
        if name not in self._tensors:
            raise KeyError(f"Tensor {name!r} not found")
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def get_database(self):
        return None


class RealTensorSource(TensorSource):
    """Loads tensors from a sharded safetensors ckpt via the index."""

    def __init__(self, ckpt_dir: str):
        index_path = os.path.join(ckpt_dir, "quant_model_weights.safetensors.index.json")
        with open(index_path) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._dir = ckpt_dir

    def load_tensor(self, name: str, data_type=torch.float16):
        fname = self._weight_map[name]
        with safe_open(os.path.join(self._dir, fname), framework="pt") as fh:
            t = fh.get_tensor(name)
        return [t.to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._weight_map

    def get_database(self):
        return None


class StubDevice:
    """No-op stand-in for load_config.exported_device (identity hooks)."""

    def maybe_rewrite_weight_by_key(self, key, weight):
        return weight

    def shuffle_moe_weight(self, x, datatype, name):
        return x


def make_load_config(
    hidden_size: int,
    head_num: int,
    head_num_kv: int,
    size_per_head: int,
    num_experts: Optional[int] = None,
):
    """Single-card LoadConfig stand-in (all parallelism degrees = 1)."""
    lc = MagicMock()
    lc.tp_size = 1
    lc.tp_rank = 0
    lc.dp_size = 1
    lc.dp_rank = 0
    lc.ep_size = 1
    lc.ep_rank = 0
    lc.ffn_tp_size = 1
    lc.ffn_tp_rank = 0
    lc.lm_head_tp_size = 1
    lc.lm_head_tp_rank = 0
    lc.compute_dtype = torch.bfloat16
    lc.merge_lora = False
    lc.moe_pure_tp_mode = False
    lc.bit = 8
    lc.hidden_size = hidden_size
    lc.head_num = head_num
    lc.head_num_kv = head_num_kv
    lc.size_per_head = size_per_head
    lc.get_selected_experts.return_value = (
        list(range(num_experts)) if num_experts else []
    )
    lc.exported_device = StubDevice()
    return lc


def mx_quantize(w_nk: torch.Tensor):
    """NPU MX-quantize [N, K] -> ckpt-layout (kernel, scale) on cpu.

    Returns (kernel [N, K] fp8, scale [N, kp] uint8), i.e. the layout a
    ModelSlim ckpt stores (unswizzled).
    """
    kernel, scale_3d = torch_npu.npu_dynamic_mx_quant(
        w_nk.to("npu"), dst_type=MXFP8_DTYPE
    )
    # flatten the pair-split [N, kp//2, 2] op output back to [N, kp]
    scale = scale_3d.reshape(w_nk.shape[0], w_nk.shape[1] // MXFP8_GROUP).cpu()
    return kernel.cpu(), scale


def quantize_weight_swizzled(w_nk: torch.Tensor):
    """[N, K] -> fp8 [N, K] + swizzled uint8 scale [kp//2, N, 2] (NPU tensors).

    Produces the runtime layout the executor consumes (post _postprocess).
    """
    n, k = w_nk.shape
    kernel, scale_3d = torch_npu.npu_dynamic_mx_quant(
        w_nk.to("npu"), dst_type=MXFP8_DTYPE
    )
    scale_2d = scale_3d.reshape(n, k // MXFP8_GROUP)
    kp = k // MXFP8_GROUP
    if kp % 2 != 0:
        scale_2d = torch.nn.functional.pad(scale_2d, (0, 1))
        kp += 1
    swizzled = scale_2d.reshape(n, kp // 2, 2).transpose(0, 1).contiguous()
    return kernel, swizzled


def stack_fp8(tensors, dim=0):
    """torch.stack does not support fp8 on NPU; stack via uint8 views."""
    u8 = torch.stack([t.view(torch.uint8) for t in tensors], dim=dim)
    return u8.view(torch.float8_e4m3fn)


def e8m0_dequant(kernel: torch.Tensor, scale_swizzled: torch.Tensor) -> torch.Tensor:
    """Dequantize [K, N] fp8 kernel with swizzled [kp//2, N, 2] E8M0 scale."""
    kp2, n, two = scale_swizzled.shape
    assert two == 2
    scale_2d = scale_swizzled.permute(1, 0, 2).reshape(n, kp2 * 2)  # [N, kp]
    exp = torch.pow(2.0, scale_2d.float() - 127.0)  # [N, kp]
    exp_full = exp.repeat_interleave(MXFP8_GROUP, dim=1)  # [N, K]
    return kernel.float().T * exp_full  # [N, K]


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def load_and_check(weight, qc, tensor_source, layer_id, load_config, expected):
    """Create the quant wrapper, load layer `layer_id`, assert shape/dtype/device."""
    qw = WeightModule.create(weight, qc)
    assert isinstance(qw, AscendW8A8MXFP8Weight), type(qw)
    res = qw.load(tensor_source, layer_id=layer_id, device="npu",
                  load_config=load_config)
    for name, (shape, dtype) in expected.items():
        assert name in res, f"{name} missing from {list(res.keys())}"
        got = res[name]
        assert tuple(got.shape) == tuple(shape), f"{name}: {tuple(got.shape)} != {tuple(shape)}"
        assert got.dtype == dtype, f"{name}: {got.dtype} != {dtype}"
        assert got.device.type == "npu"
    return res
