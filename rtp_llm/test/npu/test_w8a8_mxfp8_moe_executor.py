"""AscendW8A8MXFP8Executor verification on Ascend NPU (needs torch_npu).

Run inside the NPU container:
  PYTHONPATH=<repo> python -m pytest rtp_llm/test/npu/test_w8a8_mxfp8_moe_executor.py -v -s

Validates plan B (stepwise grouped GEMM path) against a per-expert bf16
reference, including the up|gate ordering of moe_w1 and the
grouped-contiguous compaction/scatter logic.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.executors.w8a8_mxfp8_executor import (
    AscendW8A8MXFP8Executor,
)
from rtp_llm.utils.model_weight import W

from test_w8a8_common import (
    cos_sim,
    quantize_weight_swizzled,
    stack_fp8,
)

torch.manual_seed(2024)

E = 4
T = 16
HIDDEN = 256
INTER = 128


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def w_bf16_pair():
    """Original bf16 expert weights: w1 [E, 2N, K] up|gate, w2 [E, N_hidden, K_inter]."""
    w1 = torch.randn(E, 2 * INTER, HIDDEN, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, HIDDEN, INTER, dtype=torch.bfloat16) * 0.05
    return w1, w2


@pytest.fixture(scope="module")
def weights(w_bf16_pair):
    """Executor-consumable weights: quantized, transposed and stacked."""
    w1_bf16, w2_bf16 = w_bf16_pair
    w1_list, w1s_list, w2_list, w2s_list = [], [], [], []
    for e in range(E):
        k1, s1 = quantize_weight_swizzled(w1_bf16[e])
        k2, s2 = quantize_weight_swizzled(w2_bf16[e])
        w1_list.append(k1.transpose(0, 1).contiguous())  # [K, 2N]
        w1s_list.append(s1)
        w2_list.append(k2.transpose(0, 1).contiguous())  # [K_inter, N_hidden]
        w2s_list.append(s2)
    w1 = stack_fp8(w1_list)              # [E, HIDDEN, 2*INTER]
    w1_scale = torch.stack(w1s_list, dim=0)  # [E, kp1, 2*INTER, 2]
    # w2: ckpt [E, N_hidden, K_inter] -> final [E, K_inter, N_hidden]
    w2 = stack_fp8(w2_list)              # [E, INTER, HIDDEN]
    w2_scale = torch.stack(w2s_list, dim=0)  # [E, kp2, HIDDEN, 2]
    return {
        W.moe_w1: w1.to("npu"),
        W.moe_w2: w2.to("npu"),
        W.moe_s1: w1_scale.to("npu"),
        W.moe_s2: w2_scale.to("npu"),
    }


@pytest.fixture(scope="module")
def masked_m():
    # deliberately includes an empty expert (0) and a full slot (T)
    return torch.tensor([5, 0, 16, 3], dtype=torch.int32, device="npu")


@pytest.fixture(scope="module")
def expert_x():
    return torch.randn(E, T, HIDDEN, dtype=torch.bfloat16, device="npu")


@pytest.fixture(scope="module")
def executor(weights):
    return AscendW8A8MXFP8Executor(
        config=MagicMock(), quant_config=MagicMock(), weights=weights
    )


@pytest.fixture(scope="module")
def out(executor, expert_x, masked_m):
    payload = ExpertForwardPayload(
        expert_x=expert_x,
        expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=masked_m),
    )
    return executor.execute(
        payload, activation="SiGLU", expert_map=None, a2_scale=None,
        apply_router_weight_on_input=False, extra_expert_args=None,
    ).fused_expert_output


@pytest.fixture(scope="module")
def ref(w_bf16_pair, expert_x, masked_m):
    """Per-expert bf16 reference (up|gate order of w1)."""
    w1_bf16, w2_bf16 = w_bf16_pair
    res = torch.zeros(E, T, HIDDEN, dtype=torch.float32)
    x_all = expert_x.float().cpu()
    for e in range(E):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        x = x_all[e, :m]  # [m, K]
        w1e = w1_bf16[e].float().T  # [K, 2N]
        upgate = x @ w1e  # [m, 2N]
        up, gate = upgate[:, :INTER], upgate[:, INTER:]
        act = torch.nn.functional.silu(gate) * up
        w2e = w2_bf16[e].float().T  # [K_inter, N_hidden]
        res[e, :m] = act @ w2e
    return res


# ---------------- tests ----------------


def test_output_shape_dtype(out):
    assert tuple(out.shape) == (E, T, HIDDEN), out.shape
    assert out.dtype == torch.bfloat16


def test_numeric_vs_bf16_reference(out, ref, masked_m):
    # compare only valid rows
    valid_rows = []
    for e in range(E):
        m = int(masked_m[e].item())
        if m:
            valid_rows.append(out[e, :m].float().cpu())
    valid = torch.cat(valid_rows, dim=0)
    ref_valid = torch.cat(
        [ref[e, : int(masked_m[e].item())] for e in range(E) if int(masked_m[e].item())],
        dim=0,
    )
    cs = cos_sim(valid, ref_valid)
    print(f"MoE executor plan B: cos_sim={cs:.6f}")
    assert cs > 0.99, cs


def test_empty_expert_rows_zero(out):
    assert (out[1] == 0).all(), "expert with 0 tokens should stay zero"
