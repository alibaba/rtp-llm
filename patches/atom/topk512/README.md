# Opt-in ROCm small-row TopK

For the pinned ATOM 0.1.8+g9af739efc wheel with the existing RTP adapters,
AITER 0.1.29+rocm7.2.0.git97d2389 and FlyDSL 0.4.2:

```sh
python3 patches/atom/topk512/apply.py
python3 patches/atom/topk512/apply.py --verify-only
python3 patches/atom/topk512/test_topk512.py
```

Restart workers after installing. Set `ATOM_RTP_TOPK512=1` only for the
validated decode role; unset it or use 0 for the baseline. The switch is
off by default. Dispatch is limited to ROCm RTP-LLM, BF16, 512 routed
experts, top-k 10, renormalization, and 1–4 token rows. All other cases
retain the existing AITER path. Keep CUDA Graph enabled for validation.

The specialized one-wave kernel statically unrolls top-k selection, keeps
lowest-index tie breaking, computes normalized selected weights, preserves
shared-column IDs, calculates the optional shared sigmoid score, and masks
padding using the device valid-row count on every graph replay. It does
not change MTP acceptance, cache reuse, sampling, or model weights.

The installer validates version and exact source/patch/output hashes,
rejects unknown local edits, and is idempotent. This is an optional
post-wheel patch; no Dockerfile dependency change or C++ rebuild is needed.
The test compares expert IDs exactly and FP32 routing weights with
rtol=1e-5/atol=1e-7 against AITER, including ties, strided logits,
shared experts, and dynamic graph padding.
