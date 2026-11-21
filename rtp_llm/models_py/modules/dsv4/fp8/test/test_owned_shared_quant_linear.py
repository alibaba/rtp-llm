"""Exercise the same owned projections and shared-input path as V4 prefill."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Linear
from rtp_llm.models_py.weight_mapper import get_all_weights


@torch.inference_mode()
def test_rank_local_projections_reuse_nonunit_activation_scales(tmp_path):
    # The producer runs once; two distinct native GEMMs consume its output.
    # Powers of two are exactly representable in the checkpoint, quantized
    # activation, and reference. Dropping/recreating the scales changes both.
    attn = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(attn)
    x = (
        torch.tensor([[1, 2, 4, 8], [8, 4, 2, 1]], device="cuda", dtype=torch.bfloat16)
        .repeat_interleave(128, dim=1)
        .contiguous()
    )
    expected = []
    for name, partition, offset in (("wq_a", "column", 0), ("wkv", "replicated", 2)):
        owner = Dsv4FP8Linear(512, 256, tp_size=2, tp_rank=1, partition=partition)
        owner = owner.to("cuda")
        scales = torch.tensor(
            [[125, 126, 127, 128], [128, 127, 126, 125]], dtype=torch.uint8
        ) + offset
        path = str(tmp_path / (name + ".pt"))
        torch.save(
            {
                "weight": torch.ones(256, 512).to(torch.float8_e4m3fn),
                "scale": scales.view(torch.float8_e8m0fnu),
            },
            path,
        )
        owner.load_weights(
            get_all_weights([path], read_spec_resolver=owner.checkpoint_read_spec)
        )
        owner.process_weights_after_loading()
        setattr(attn, name, owner)
        scale_ref = (
            (2.0 ** (scales.float() - 127))
            .repeat_interleave(128, 0)
            .repeat_interleave(128, 1)
        )
        if partition == "column":
            scale_ref = scale_ref.chunk(2, 0)[1]
        expected.append((x.float() @ scale_ref.cuda().T).bfloat16())

    assert attn._can_reuse_qkv_input_quant()
    # Wrapping the factory dispatcher in a weight owner must not select a
    # different RMSNorm kernel just because it can consume a shared pair.
    assert not attn.can_fuse_prefill_attn_norm_input_quant(x, torch.ones_like(x[0]))
    pair = attn.wq_a.quantize_input(x)
    assert pair[0].dtype == torch.float8_e4m3fn
    assert pair[1].dtype == torch.int32
    for owner, target in zip((attn.wq_a, attn.wkv), expected):
        result = attn._lin_from_shared_quant(owner, pair, torch.Size((1, *x.shape)))
        torch.testing.assert_close(result.flatten(0, 1), target, rtol=0, atol=0)
        out = torch.empty_like(target)
        assert owner.forward_quantized(*pair, out=out) is out
        torch.testing.assert_close(out, target, rtol=0, atol=0)
