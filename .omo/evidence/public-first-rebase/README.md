# First public rebase commit resolution

Scope: replay c62f2d881c onto aa1d38c8 once, then stop at next conflict.

| Scenario | Invocation | Binary observable | Artifact |
|---|---|---|---|
| Resolved patch whitespace | `git diff --check` | exit=0 | diff-check-before.txt |
| Explicit staging of resolved paths | `git add -- rtp_llm/cpp/cache/KVCacheManager.h rtp_llm/cpp/cache/CoordinatorCacheManager.cc rtp_llm/cpp/cache/test/HybridCoordinatorCacheManagerCPShardTest.cc rtp_llm/cpp/model_rpc/test/DecodeRpcServerTest.cc`; `git diff --name-only --diff-filter=U` | unmerged_count=0 before continue | unmerged-before.txt |
| Staged whitespace | `git diff --cached --check` | exit=0 | staged-check.txt |
| Replay first commit | `GIT_EDITOR=true git rebase --continue` | c3eaf2188f created, stopped at commit 2/10; exit=1 due next conflict | continue.log |
| State after one continue | `git rev-parse HEAD REBASE_HEAD`; `git status --short` | HEAD=c3eaf2188fd19f9a8acb6943a1e766419e282ab7; REBASE_HEAD=de48aa70cafb893a56e42c05ec4038331831befb | result-heads.txt; result-status.txt |

Resolutions:
- KVCacheManager.h retains RtpLLMCacheReuseMetricsCollector declaration and drops removed CPSlotMapper declaration.
- HybridCoordinatorCacheManagerCPShardTest.cc retains upstream detach rollback/refcount/kernel table checks, initializes finalized CP2 storage layout using resolveTestCpLayout, preserves upstream device_input_length field, and uses feature parameterized 4*cp_size reuse divisor. A newly added feature test's obsolete block_aligned_input_length field also uses device_input_length.
- DecodeRpcServerTest.cc keeps feature handoff fixtures makeGenerateStream, HandoffTestEngine and CapturingHandoffLoadStore alongside merged upstream tests.
- CoordinatorCacheManager.cc automatic merge left deleted cpEffectiveSeqLenForGroup/cp_slot_mapper_ use. Replaced with config_.localAllocTokens(tag, device_ready_reuse_length). CacheConfig.h:175 implements same old mapper formula ceil(ceil(tokens/B)/CP)*B only for BLOCK_ROUND_ROBIN; other groups pass tokens through. For test B4/CP2/reuse8: FULL gets 4 tokens (one slot), LINEAR gets 8 (two slots).

Full staged semantic diff captured in resolved.patch. Runtime/build tests are not claimed; parent owns test orchestration. Next conflicts in Types.h and StreamCacheResource.cc belong to commit two and were left unresolved.
