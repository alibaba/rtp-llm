"""Destructive CUDA-context error test: run only as parent-owned subprocess."""

import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=lambda s: int(s, 0), required=True)
    p.add_argument("--legacy", action="store_true")
    args = p.parse_args()
    import deep_gemm
    import torch
    import triton
    import triton.language as tl

    from rtp_llm.models_py.triton_kernels.common.mxfp8_scale_layout import (
        checked_ue8m0_exponent,
    )

    @triton.jit
    def invalid_scale_write(scale_ptr, out_ptr):
        group = tl.arange(0, 4)
        scale = tl.load(scale_ptr + group)
        tl.store(out_ptr + group, checked_ue8m0_exponent(scale, group < 4))

    signed = args.bits if args.bits < 2**31 else args.bits - 2**32
    s = torch.tensor(
        [[signed, 0x3F800000, 0x3F800000, 0x3F800000]], device="cuda", dtype=torch.int32
    ).view(torch.float32)
    if args.legacy:
        deep_gemm.transform_sf_into_required_layout(s, mn=1, k=128, recipe=(1, 32))
    else:
        invalid_scale_write[(1,)](s, torch.empty(4, device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()
    raise AssertionError("invalid scale unexpectedly accepted")
