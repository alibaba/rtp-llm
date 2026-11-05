# Four-commit split: complete

Exactly four local public commits; no pushes or CI triggers. Internal HEAD remains ab22e47dd296f72cb570d7ff1320732fcf66ee2a. Final tree is byte-identical to original 1b56fd1e680b7842e3a1cfd246603d21a61102bf: 4d2c36f1be0b2c20c8666f9deee6fb2e658e84a7.

## Exact-SHA build acceptance

- 6e6ab2ce857e0089084435a5db9b2313d90020ef: cp independently builds //rtp_llm:rtp_llm — PASS; `.omo/evidence/cache-metrics-split-20261026/builds/split_cp_6e6ab2ce_20261026.log`, `.omo/evidence/cache-metrics-split-20261026/builds/split_cp_6e6ab2ce_20261026.signal` (signal 0 plus successful target output).
- d117337a75d3d2e0d86cf83118c47d451cad6ac9: tail independently builds //rtp_llm:rtp_llm — PASS; `.omo/evidence/cache-metrics-split-20261026/builds/split_tail_d117337a_20261026.log`, `.omo/evidence/cache-metrics-split-20261026/builds/split_tail_d117337a_20261026.signal` (signal 0 plus successful target output).
- 8f842c28859357aa53f02c02268d36f62a0d7ab5: prefix independently builds //rtp_llm:rtp_llm — PASS; `.omo/evidence/cache-metrics-split-20261026/builds/split_prefix_8f842c28_20261026.log`, `.omo/evidence/cache-metrics-split-20261026/builds/split_prefix_8f842c28_20261026.signal` (signal 0 plus successful target output).
- 00bb35cacdd1f6f8a3f5b1d0bef5a01f00fb08d8: optional diagnostics independently builds //rtp_llm:rtp_llm — PASS; `.omo/evidence/cache-metrics-split-20261026/builds/split_optional_00bb35ca_20261026.log`, `.omo/evidence/cache-metrics-split-20261026/builds/split_optional_00bb35ca_20261026.signal` (signal 0 plus successful target output).

Environment: songchenyi.scy_cuda13, non-root songchenyi.scy, CUDA13, daily_aone_bazel_remote; fixed project remote header; isolated output root /data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache and shared disk cache /home/songchenyi.scy/.cache/disk_cache. Every invocation had its own successful precheck. Literal Bazel invocations are in commands.md and wrapper launch receipts; precheck receipts are alongside them. Default nvidia_wheel_env/GPU lock preserved.

## No-diagnostics functional acceptance

18 selected target suites PASS. regression-results.json lists each target, execution SHA, captured XML/log and skipped cases. 17 suites ran at b3b6ea490cc7f287e6cdbeecea8abd0dd60413c6. Only the private BlockTreeTest.cc fixture changed afterward: explicit functional token_ends replaced metadata capture formerly enabled by diagnostics. retained-test-equivalence.patch and block-tree-test-BUILD.txt prove the other test and production inputs unchanged. Full block_tree_test reran fresh at final commit3 8f842c28859357aa53f02c02268d36f62a0d7ab5 and all47 cases passed. Parent explicitly authorized retaining unaffected results. No failing functional assertions were removed or weakened.

Two manual real-KVCM engine cases were SKIPPED, not validated: testRealRemoteHistoryRestoresStreamAndPreparesInternalFork requires COMMON_PREFIX_REAL_KVCM_ADDRESS; testRealRemoteReadFailureDoesNotPublishInternalFork additionally requires COMMON_PREFIX_REAL_KVCM_BLOCK_DIR. Mock backend and async recovery coverage passed. No model smoke runtime claim is made.

mixed-functional-runtime.json directly verifies passing XML cases for six storer failure/queue/shutdown/refcount tests, two phase/idempotence guards and three internal-fork/alias/image stream tests. final-tests contains exact diagnostic-only test/assertion removal inventory. metadata-fixture-failure.md records the one runtime failure and owning-tail fixture fix. Earlier failed run artifacts remain for audit.

## Split and preservation

- CP owns its functional layout-test API/dependency adaptations and existing PrefillCacheHitMetrics compatibility.
- Tail owns token geometry/reuse, coupled GDN/KDA kernels, functional metadata and settlement phase guards.
- Prefix owns authenticated exact common-prefix recovery/readiness/registration. PrefixCachePlanPoint and RequestLayout token_ids remain functional dependencies.
- Optional commit4 owns new observations/config/events/remote query status/plan diagnostics and their tests, manual benefit benchmark plus three BUILD targets, and diagnostic-only mixed-smoke assertions. Functional physical-writer assertions remain before commit4.

Updated tail-no-diagnostics.json and prefix-no-diagnostics.json record git grep invocation and exit1 (no new diagnostic symbols/events/config) over each entire tracked rtp_llm tree. optional-staged.patch records the exact final delta; final-proof.json records four commits, identities, zero final diff, tree ID, public/internal status and unchanged internal HEAD. Public status is only existing untracked .omo/. Internal preexisting .codex deletion, .trellis/, CONTINUITY.md and modified submodule status preserved. Backup ref backup/cache-metrics-split-20261026-start retains original final. No original-workspace source modifications; test worker's accidental evidence placement was removed with receipt in tail-tests/relocation-receipt.json.
