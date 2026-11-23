"""Compare the public QSA prefill adapter with an FP32 reference."""
import pytest


@pytest.mark.parametrize("total,prefix,batch", [(8192, 6080, 2), (10240, 7104, 4)])
def test_qsa_prefill(total, prefix, batch):
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.ops.qsa import qsa_sparse_prefill
    from atom.model_ops.qwen4_exp.qsa_attention import _flydsl_attention_op

    assert _flydsl_attention_op() is qsa_sparse_prefill
    torch.manual_seed(218)
    length = total - prefix
    heads = 12
    d = 256
    top = 1024
    width = 1027
    q = torch.randn(batch * length, heads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * total, 1, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    visible = prefix + torch.arange(length, device="cuda", dtype=torch.int32) + 1
    indices = torch.full((length, width), -1, device="cuda", dtype=torch.int32)
    indices[:, :top] = torch.arange(top, device="cuda", dtype=torch.int32)
    for t in range(3):
        indices[:, top + t] = torch.where(t < visible % 4, visible // 4 * 4 + t, -1)
    indices = indices.repeat(batch, 1).contiguous()
    actual = qsa_sparse_prefill(
        q,
        k,
        v,
        indices,
        query_lens=(length,) * batch,
        prefix_lens=(prefix,) * batch,
        softmax_scale=d**-0.5,
    )
    for start in range(0, q.shape[0], 32):
        stop = min(start + 32, q.shape[0])
        sel = indices[start:stop].long()
        req = torch.arange(start, stop, device="cuda") // length
        slots = sel.clamp_min(0) + req[:, None] * total
        keys = k[slots, 0].float()
        vals = v[slots, 0].float()
        logits = torch.einsum("rhd,rkd->rhk", q[start:stop].float(), keys) * d**-0.5
        logits.masked_fill_(sel[:, None, :] < 0, float("-inf"))
        expected = torch.einsum("rhk,rkd->rhd", logits.softmax(-1), vals)
        torch.testing.assert_close(
            actual[start:stop].float(), expected, atol=0.005, rtol=0.02
        )
