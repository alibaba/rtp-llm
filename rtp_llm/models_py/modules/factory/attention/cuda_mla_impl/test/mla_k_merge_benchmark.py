"""Hot 64K MLA K merge comparison for the eager prefill path."""

import json
import statistics

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    concat_and_cast_mha_k_triton,
)


def timed_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def measure_group(name, fn):
    fn()  # Materialize any lazy Triton kernel and allocator state.
    torch.cuda.synchronize()
    warmup = []
    for _ in range(40):
        warmup.append(timed_ms(fn))
        if len(warmup) >= 10:
            mid = statistics.median(warmup[-3:])
            if max(abs(t - mid) / mid for t in warmup[-3:]) <= 0.05:
                break
    else:
        raise RuntimeError(f"{name} did not converge in 40 warmup iterations")

    torch.cuda.synchronize()
    samples = [timed_ms(fn) for _ in range(30)]
    return {
        "name": name,
        "warmup_ms": warmup,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
    }


def main():
    tokens, heads, nope_dim, rope_dim, value_dim = 65536, 12, 128, 64, 128
    torch.manual_seed(27)
    kv = torch.randn(
        tokens, heads, nope_dim + value_dim, dtype=torch.bfloat16, device="cuda"
    )
    k_nope = kv[..., :nope_dim]
    k_rope = torch.randn(
        tokens, 1, rope_dim, dtype=torch.bfloat16, device="cuda"
    )
    copy_out = torch.empty(
        tokens, heads, nope_dim + rope_dim, dtype=torch.bfloat16, device="cuda"
    )
    fused_out = torch.empty_like(copy_out)

    def copy_merge():
        copy_out[..., :nope_dim].copy_(k_nope)
        copy_out[..., nope_dim:].copy_(k_rope)

    def fused_merge():
        concat_and_cast_mha_k_triton(fused_out, k_nope, k_rope)

    copy_merge()
    fused_merge()
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_out, copy_out, rtol=0, atol=0)

    groups = [
        measure_group(name, fn)
        for name, fn in (
            ("two_copies_a", copy_merge),
            ("one_kernel_a", fused_merge),
            ("one_kernel_b", fused_merge),
            ("two_copies_b", copy_merge),
        )
    ]
    print(
        json.dumps(
            {
                "shape": {
                    "tokens": tokens,
                    "heads": heads,
                    "nope_dim": nope_dim,
                    "rope_dim": rope_dim,
                    "value_dim": value_dim,
                    "dtype": "BF16",
                    "k_nope_contiguous": k_nope.is_contiguous(),
                },
                "device": torch.cuda.get_device_name(),
                "warmup_rule": "at least 10, last three within 5% of median",
                "samples_per_group": 30,
                "groups": groups,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
