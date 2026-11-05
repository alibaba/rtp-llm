# Pre-pickle cache regression tests

Tested the already replayed public HEAD `e7e24839331d24968cc1cb718e81ccf9a449bdd5` in a temporary detached worktree while the third replay commit was paused on pickle compatibility.

- Container: `songchenyi.scy_cuda13`, non-root user `songchenyi.scy`; required pre-build check passed.
- Isolated Bazel output root: `/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_validation-public_cache`.
- Config: `cuda13`; fixed project remote cache and shared disk cache enabled; required `//rtp_llm/test/utils:gpu_lock` wrapper inserted.
- Targets: `//rtp_llm/cpp/cache/test:kv_cache_manager_cp_test` and `//rtp_llm/cpp/cache/test:prefill_cache_hit_metrics_contract_test`.
- Result: exit signal 0; both targets PASSED (7.9s and 7.6s). See adjacent `run.log` and `run.signal`.

This validates the first two rebased commits only; the full branch requires a later test run after pickle resolution and remaining commits.
