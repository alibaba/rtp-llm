# Isolated partial-match rebase validation, 2026-10-25

## Branches and safety

- Internal: `5e6343ab9907e1e2651b419c00a9c09cd7b1a7fe` -> fresh `origin/main-internal=b4eb681767dbd73d8c3a552027fa7783346c94a9` -> `199f5f93dba0b73ca7203b47386ed8dffaf95bac`.
- Public: `cd7caa2eafd0fc16c26367b11738c923bc281412` -> fresh `upstream/main=aa1d38c87a232fd8c4599e7a10dcf909c99194f7` -> `183b75100df1fd58cc64bbb5d2be1345a0b1ceb2`.
- Both remote OIDs were rechecked after rebasing. Backup refs retain the original HEADs. No push, CI action, tag, or gitlink commit occurred.
- Internal branch contains two separate documentation commits; public branch contains ten separate replayed commits and one independent DSv4 profiler fix. Original workspace dirty edits were not touched.

## Semantic reconciliation

| Area | Result and evidence |
| --- | --- |
| CP geometry / shared-block detach | Reconciled CP slot geometry and mainline shared-block detach; partial LINEAR slots remain pinned until continuation creates private copies. CP layout and hybrid shard tests passed; see `public-third-detach/README.md`. |
| Async failed load | Kept the shared suffix protected on failed asynchronous load/continuation. `block_tree_loader_test` and `load_async_context_test` passed on the final branch. |
| KVCacheConfig serialization | Mainline v8 means CRC by user decision. v11 stores CRC, tail reuse, and common-prefix recovery separately; old private v9 remains readable. `th_transformer_config` built and all six exact pickle unit tests passed against that new extension. |
| P/D reuse statistics | Transferred KV remains handoff data, not historical prefix reuse. No `markLoadedCacheReuse` was reintroduced. Final `decode_rpc_server_test`, `cp_transfer_projection_test`, and `stream_cache_resource_test` passed. |
| Cache metrics | Preserved mainline collector and aggregation contract, including `publishReuseLengths` accounting; final `prefill_cache_hit_metrics_contract_test` passed. |
| DSv4 profiling / hidden-state capture | Kept `forward_prefill` within the MoE profiler suppression scope in independent commit `183b75100d`; added a range regression. Hidden-state capture integration remained intact in the replay. Profiler Bazel target was blocked during wheel analysis; prior read-only audit and Python syntax checks are recorded in `dsv4-profiler-fix/README.md`. |
| PPU / Python 3.12 | Internal/public baseline trees match for paired PPU and Python 3.12 changes; no extra gitlink move needed. Nine CPU-safe PPU wrapper/warmup tests passed under conda310. Final changed Python files (77) compiled as Python 3.12 syntax. PPU SDK and conda312 are absent locally. |

## Test evidence

- Non-root container `songchenyi.scy_cuda13`, user `songchenyi.scy` (UID 16799314), CUDA13; Bazel skill precheck and wrapper used separate output root `/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_cache`, remote/disk cache, and the exact GPU lock for tests.
- `../../../build_logs/final_cache_cpp_20261025.log` and `.signal`: 13/13 C++ targets passed (six executed, seven Bazel cached), covering CP layout, hybrid shard, metrics, CP transfer, tier cache, DSv4 cache, block tree, async load, stream resource, RPC server, and engine.
- `../../../build_logs/final_config_module_build_20261025.log` and `.signal`: `//:th_transformer_config` built successfully; the changed `ConfigInit.cc` compiled.
- Direct non-root run against that built extension: `kv_cache_config_pickle_test.py`, six tests passed. It injected only `KVCacheConfig` from the newly built module into `rtp_llm.ops` to avoid unrelated runtime imports; the test code itself was unchanged.
- Final Black check passed for four changed Python/config files. Python 3.12 syntax compilation passed for 77 changed `.py` files. `git diff upstream/main..HEAD --check` passed.
- `../../../build_logs/final_pickle_config_20261025.log` and `.signal`: Bazel `kv_cache_config_pickle_test` and `server_config_setup_test` did not start because pinned wheels (`fastsafetensors`, `fast_safetensors`, `flash_mla`, `torchvision`) timed out downloading from `rtp-maga.cn-zhangjiakou.oss.aliyuncs.com` during analysis.
- Earlier isolated snapshots also passed eight final non-pickle C++ targets and all ten third-commit cache/P-D targets; see `future-nonpickle/README.md` and `third-staged-tests/README.md`. The two-GPU CP completion target and DSv4 profiler Python Bazel target hit the same OSS wheel timeout. Direct hidden-state capture import hit missing `pydantic` in the base conda310 runtime, before test execution.
