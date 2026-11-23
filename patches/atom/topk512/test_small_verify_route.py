"""Compare the MI308X D-side verifier route layout with the pinned AITER ABI."""
from unittest.mock import patch

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
    pytest.skip("requires gfx942", allow_module_level=True)

from aiter.fused_moe import moe_sorting
from atom.plugin.rtpllm.utils import small_verify_route
from atom.plugin.rtpllm.utils.topk512 import run


@pytest.mark.parametrize("rows", [4, 8])
def test_qwen4_verify_routes(rows):
    torch.manual_seed(123)
    for rank in (0, 1):
        mask = torch.zeros(514, device="cuda", dtype=torch.int32)
        mask[rank * 256 : (rank + 1) * 256] = 1
        mask[512] = 1
        ids = torch.full((rows, 12), 513, device="cuda", dtype=torch.int32)
        ids[rank::2, 10] = 512
        weights = torch.ones_like(ids, dtype=torch.float32)
        valid = torch.tensor([rows], device="cuda", dtype=torch.int32)
        logits = torch.randn((rows, 513), device="cuda", dtype=torch.bfloat16)
        with patch.object(small_verify_route, "pass_scheduled_tokens_gpu", return_value=valid):
            small_verify_route.prepare_flydsl_routes(logits, ids, mask)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused = small_verify_route.prepare_flydsl_routes(logits, ids, mask)
            for live in range(rows + 1):
                valid.fill_(live)
                logits.normal_()
                logits[:, :12] = 8  # Ties at the routing cutoff.
                graph.replay()
                run(logits, weights, ids, valid)
                reference = moe_sorting(ids, weights, 514, 2560, torch.bfloat16, 16, mask)
                padded = int(reference[3][0])
                for x, y, length in zip(fused, reference, (padded, padded, padded // 16, 2, rows)):
                    torch.testing.assert_close(x[:length], y[:length], rtol=2e-6, atol=1e-7)
