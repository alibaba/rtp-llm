# Publication checks — 2026-09-25

This is a personal BF16 development snapshot, not a green full-CI or merge-readiness claim.

## Source identity

40 changed/new product, build and test files were compared byte-for-byte by SHA256 against the live 145 worktree used for PD427. All matched. The publication adds documentation only beyond that product snapshot. No GPU service was restarted and no GPU test was launched for this publication.

113 changed/new Python source files relative to the integrated main base parsed successfully. Generated `.pyi` files were not treated as Python implementation files. `git diff --check` passed. Outgoing task commits use author and committer `luohaocheng.lhc <luohaocheng.lhc@alibaba-inc.com>`; existing main history is preserved.

## Fresh CPU contract run

145 existing runtime container, Python3.10, existing RTP runfiles dependencies and `k3-attnres-test-deps`, `CUDA_VISIBLE_DEVICES=''`:

```sh
python -m pytest -q \
  tests/kimi_k3/test_model_math.py \
  tests/kimi_k3/test_bf16_launch_profile.py \
  tests/kimi_k3/test_launch_gpu_capacity.py \
  tests/kimi_k3/test_smoke_contract.py \
  tests/kimi_k3/test_chunk_plan.py \
  tests/kimi_k3/test_chunk_inputs.py \
  tests/kimi_k3/test_forward_metadata.py \
  tests/kimi_k3/test_text_renderer.py
```

Result: **125 passed in 1.41 seconds**, exit0. This does not certify CUDA, native MLA, real cache commits or the whole test directory.

## Broader attempts and failures retained

1. Initial invocation lacked pytest on the bare Python path; no tests ran. Reused the existing test dependency directory without installing into the service environment.
2. `pytest -q tests/kimi_k3` with CUDA hidden failed collection in `test_mla_prefill.py`: importing the attention factory on the non-CUDA path could not import `FusedRopeKVCacheDecodeOp`. The module has GPU tests, but imports happen before the skip condition.
3. A diagnostic run ignoring only that GPU module reported **149 passed, 52 skipped, 7 failed, 7 errors**. One KDA checkpoint test allocates CUDA tensors without a CPU skip; seven SiTU cases initialize CUDA in their fixture. Those are incompatible with the deliberately hidden-GPU environment. The other **six failures are stale MLA geometry mocks** in `test_mla_update_geometry.py`, which still target the earlier `MlaFlashInferImplBase` adapter rather than the new `NativeMlaDecode` implementation. They need to be ported and rerun. They have not been removed or marked passing.

No product behavior or existing test was altered solely to turn these publication checks green. The preserved full93 GPU smoke and earlier C++ 61-test results have their own scope described in the handoff. A subsequent merge effort must fix test drift and run affected legacy-model regressions.

## Scope of actual functional evidence

PD427: 118 completed cases independently checked, three long repeat cases explicitly deferred by the user. The original wrapper terminated with smoke_exit=-15 and exit1 after user cancellation; these markers are preserved. No original all-suite success is claimed.

The raw responses include real PD routing and MTP acceptance. Sampled runtime windows establish 508 CUDA Graph replay start/end pairs and 2187 successful RDMA write callbacks. Eventual normal backend completion recovered all cache capacity on both endpoints. Prompt cancellation was not successful and remains a known issue.
