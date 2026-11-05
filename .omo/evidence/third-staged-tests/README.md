# Staged third-commit cache tests

A temporary detached worktree on `e7e24839331d24968cc1cb718e81ccf9a449bdd5` received the paused third commit's entire staged patch except the two unresolved pickle files (`ConfigInit.cc`, `kv_cache_config_pickle_test.py`). The SHA-256 of `git diff --cached --binary` with those two exclusions was identical in the paused worktree and test worktree: `8555844f5232a27c300ed973aa8508b6dbbb18696dea113b56a8ea0e1c67eccd`.

All Bazel invocations used non-root `songchenyi.scy_cuda13`, a passing `test-execution` precheck, `--config=cuda13`, output root `/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_third-staged_cache`, project remote cache, shared disk cache, and the exact GPU-lock wrapper.

- First C++ run failed to build the new `HybridCoordinatorCacheManagerCPShardTest.cc`: `PrefixCacheRegistration` was an incomplete type. The missing explicit header was added and staged in the paused branch. See `cpp-initial.log` and signal.
- Retry passed all three targets: `hybrid_kv_cache_allocator_cp_shard_test` (7.7s), `block_tree_loader_test` (6.6s), and `load_async_context_test` (6.0s). See `cpp-retry.log` and zero signal.
- The two-GPU `tail_reuse_cp_completion_integration_test` failed during Bazel analysis before any test ran because pinned wheels including `fast_safetensors` could not be downloaded from `rtp-maga.cn-zhangjiakou.oss.aliyuncs.com` (connection timeout). See `cp-integration.log` and nonzero signal.

These tests validate the staged non-pickle snapshot only. Full-branch tests remain required after pickle resolution and later feature commits.

Further staged-snapshot validation used the same non-root wrapper and precheck:

- `stream_cache_resource_test`, `decode_rpc_server_test`, `dsv4_cache_test`, and `kv_cache_manager_with_tier_cache_test` all passed. See `pd-metrics.log` and zero signal.
- `cp_transfer_projection_test` and `prefill_cache_hit_metrics_contract_test` passed. The same run exposed one stale assertion in `KVCacheManagerCPLayoutTest.CPShardedMallocAllowsPartialTailWithoutCacheKey`: it expected zero keys after a length-two partial tail, while the new runtime/P-D key policy yields one key. The observed assertion was `cacheKeys(0).size(): 1` versus expected `0` at `KVCacheManagerCPLayoutTest.cc:317`. See `cp-metrics-initial.log` and nonzero signal.
- The CP test now verifies one runtime key, one physical block, and `lastBlockAligned() == false`, preserving the complete-block publication boundary. `kv_cache_manager_cp_test` passed after this correction. See `cp-layout-retry.log` and zero signal.

The current excluded-path staged-diff SHA-256 matched between the paused rebase and temporary snapshot after this correction: `6ddf87b2f0be70c56f44a4921d4c0bc567667c770c68b6fd498830d3ad2a77f9`.
