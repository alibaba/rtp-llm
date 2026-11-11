"""Regression checks for global-to-local routing in native gfx942 MoE decode."""

import pytest
import torch

aiter = pytest.importorskip("aiter")
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx942",
    reason="requires gfx942",
)


def quantized_weight(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 240.0
    quantized = (weight / scale).clamp(-240, 240).to(torch.float8_e4m3fnuz)
    return shuffle_weight(quantized), scale, quantized.float() * scale


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("rank", [0, 1])
def test_ep_mask_and_changed_graph_inputs(rows, rank):
    torch.manual_seed(100 + rank)
    device = "cuda"
    w1, s1, reference_w1 = quantized_weight(
        torch.randn(2, 256, 256, device=device) * 0.05
    )
    w2, s2, reference_w2 = quantized_weight(
        torch.randn(2, 256, 128, device=device) * 0.05
    )
    hidden = (torch.randn(rows, 256, device=device) * 0.1).bfloat16()
    mask = torch.tensor([1, 0, 1, 0] if rank == 0 else [0, 1, 0, 1],
                        dtype=torch.int32, device=device)
    ids = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.int32,
                       device=device).repeat(rows // 3, 1)
    original_ids = ids.clone()
    weights = torch.full((rows, 2), 0.5, device=device)

    def execute(count=None):
        return run_flydsl_moe_gfx942(
            hidden, w1, w2, weights, ids, aiter.ActivationType.Silu,
            aiter.QuantType.per_Token, s1, s2, mask, count, 0,
            "16_16_16_False:default:0", 0.0,
        )

    reference = torch.zeros(rows, 256, device=device)
    for local, global_expert in enumerate((rank, rank + 2)):
        token, slot = torch.where(ids == global_expert)
        gate, up = (hidden[token].float() @ reference_w1[local].T).chunk(2, -1)
        value = (torch.nn.functional.silu(gate) * up) @ reference_w2[local].T
        reference.index_add_(0, token, value * weights[token, slot, None])
    output = execute()
    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.003)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    remote = torch.where(mask == 0)[0].to(ids.dtype)
    ids.copy_(remote.expand_as(ids))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(captured).item() == 0

    ids.copy_(original_ids)
    hidden.mul_(0.75)
    expected = execute().clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.002)
    with pytest.raises(NotImplementedError, match="num_local_tokens"):
        execute(torch.tensor([rows - 1], device=device))
