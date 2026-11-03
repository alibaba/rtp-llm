"""AscendW8A8MXFP8Executor verification on Ascend NPU (needs torch_npu).

Run inside the NPU container:
  PYTHONPATH=<repo> python -m pytest rtp_llm/test/npu/test_w8a8_mxfp8_moe_executor.py -v -s

Validates the CANN-native pipeline (fused dispatch via npu_moe_init_routing_v2
-> grouped GEMMs -> fused combine via npu_moe_token_unpermute, incl. topk
weights) against a per-token per-expert bf16 reference, including:
  * empty experts (expert 1 receives no tokens)
  * multiple tokens routed to the same expert (grouping correctness)
  * the up|gate ordering of moe_w1
  * the topk weighted-sum semantics of the fused combine
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
T = 8  # tokens; several share experts (multi-token grouping), expert 1 stays empty
HIDDEN = 256
INTER = 128
TOPK = 2


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
def topk_ids():
    # expert 1 never selected -> empty expert; several tokens share experts.
    pool = torch.tensor([[0, 2], [0, 2], [0, 2], [2, 3], [2, 3], [0, 3], [0, 3], [2, 3]])
    return pool.to(torch.int32).to("npu")


@pytest.fixture(scope="module")
def topk_weights():
    g = torch.rand(T, TOPK) + 0.1
    return (g / g.sum(dim=1, keepdim=True)).float().to("npu")


@pytest.fixture(scope="module")
def expert_x():
    return torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="npu")


@pytest.fixture(scope="module")
def executor(weights):
    return AscendW8A8MXFP8Executor(
        config=MagicMock(), quant_config=MagicMock(), weights=weights
    )


@pytest.fixture(scope="module")
def out(executor, expert_x, topk_ids, topk_weights):
    payload = ExpertForwardPayload(
        expert_x=expert_x,
        expert_tokens_meta=ExpertTokensMetadata(),
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
    )
    return executor.execute(
        payload, activation="SiGLU", expert_map=None, a2_scale=None,
        apply_router_weight_on_input=False, extra_expert_args=None,
    ).fused_expert_output


@pytest.fixture(scope="module")
def ref(w_bf16_pair, expert_x, topk_ids, topk_weights):
    """Per-token per-expert bf16 reference (up|gate order of w1)."""
    w1_bf16, w2_bf16 = w_bf16_pair
    x_all = expert_x.float().cpu()
    ids = topk_ids.cpu().long()
    wts = topk_weights.cpu().float()
    res = torch.zeros(T, HIDDEN, dtype=torch.float32)
    for t in range(T):
        for j in range(TOPK):
            e = int(ids[t, j])
            x = x_all[t]  # [K]
            w1e = w1_bf16[e].float().T  # [K, 2N]
            upgate = x @ w1e  # [2N]
            up, gate = upgate[:INTER], upgate[INTER:]
            act = torch.nn.functional.silu(gate) * up
            w2e = w2_bf16[e].float().T  # [K_inter, N_hidden]
            res[t] += wts[t, j] * (act @ w2e)
    return res


# ---------------- tests ----------------


def test_output_shape_dtype(out):
    assert tuple(out.shape) == (T, HIDDEN), out.shape
    assert out.dtype == torch.bfloat16


def test_numeric_vs_bf16_reference(out, ref):
    cs = cos_sim(out.float().cpu(), ref)
    print(f"MoE executor (fused dispatch/combine): cos_sim={cs:.6f}")
    assert cs > 0.99, cs


def test_topk_weight_semantics(out, ref, topk_weights):
    # strongest on the tokens whose experts differ the most: per-token cos
    per_tok = []
    for t in range(T):
        a = out[t].float().cpu()
        b = ref[t]
        per_tok.append(float(a @ b / (a.norm() * b.norm())))
    worst = min(per_tok)
    print(f"per-token cos_sim: min={worst:.6f} mean={sum(per_tok) / T:.6f}")
    # empty expert (1) is never referenced; every token is a valid fusion check
    assert worst > 0.97, (worst, per_tok)
