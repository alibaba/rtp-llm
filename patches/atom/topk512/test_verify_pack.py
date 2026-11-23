"""Check EP route packing against Opus on MI308X, including graph replay and padding."""
import pytest
import torch


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_qwen4_verify_pack(rows):
    if not torch.version.hip or not torch.cuda.is_available() or "MI308X" not in torch.cuda.get_device_name():
        pytest.skip("requires the MI308X wheel overlay")

    from aiter.fused_moe import QWEN4_VERIFY_PACK, moe_sorting

    for rank in (0, 1):
        # Skew every row toward the same experts to exercise multi-tile counts.
        ids = torch.full((rows, 12), 513, dtype=torch.int32, device="cuda")
        ids[:, :10] = torch.arange(10, device="cuda") + rank * 256
        ids[rank::2, 10] = 512
        weights = torch.rand((rows, 12), device="cuda")
        mask = torch.zeros(514, dtype=torch.int32, device="cuda")
        mask[rank * 256:(rank + 1) * 256] = 1
        mask[512] = 1
        live = torch.tensor([rows], dtype=torch.int32, device="cuda")
        args = (ids, weights, 514, 2560, torch.bfloat16, 32, mask, live)
        output = torch.zeros((rows, 2560), device="cuda", dtype=torch.bfloat16)
        moe_sorting(*args, dispatch_policy=QWEN4_VERIFY_PACK, output=output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = moe_sorting(*args, dispatch_policy=QWEN4_VERIFY_PACK, output=output)
        for count in (0, 1, rows // 2, rows - 1, rows):
            live.fill_(count)
            output.fill_(17)
            expected = moe_sorting(*args, output=output.clone())
            graph.replay()
            padded = int(expected[3][0])
            for left, right, length in zip(expected, actual, (padded, padded, padded // 32, 2, rows)):
                torch.testing.assert_close(left[:length], right[:length], rtol=0, atol=0)
