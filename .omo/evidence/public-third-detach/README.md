# Partial ready-slot detach integration

Root revised task after call-path inspection: preserve the partially ready slot for required continuation; do not detach it without copying ready KV.

Changes explicitly staged only in CoordinatorCacheManager.cc and HybridCoordinatorCacheManagerCPShardTest.cc. detach now starts at ceil(G/logicalBlockSpan), using the existing CP-aware span helper. The new RecomputePreservesPartialSlotsForRequiredContinuation test covers G17/B4 CP2 ranks0-1 and CP4 ranks0-3, with LINEAR and FULL shared blocks. It verifies unchanged ready slots after detach, private suffix, source refcounts, then partial-slot replacement by prepareRequiredContinuation with two source pins/copy mappings, and full cleanup. Existing RecomputeDetachesSuffixUsingGroupCPMapping G8 rollback test unchanged.

| Scenario | Invocation | Observable | Artifact |
|---|---|---|---|
| Before/after geometry | Python source/geometry assertions comparing staged-before formula and working-after formula | old LINEAR G17 fails expected begin5; new LINEAR/FULL CP2/CP4 and aligned G8 pass | geometry-check.txt |
| Whitespace | git diff --check -- two assigned paths | exit0 | scoped-check.txt |
| Stage | git add -- two assigned paths; git diff --cached --check | exit0 | staged-check.txt |
| Rebase paused | git rev-parse HEAD REBASE_HEAD; git diff --name-only --diff-filter=U | HEAD e7e2483933, REBASE_HEAD e9b2736f82; only two pickle conflicts | heads.txt; remaining-conflicts.txt |

Narrow change captured before staging in fix.patch. The numerical check is NOT production runtime validation. New C++ test has not been compiled/run; parent owns Bazel execution, so no runtime pass is claimed.

Call-path findings from direct source inspection:
- StreamCacheResource.cc initKVBlock creates prefix_cache_registration before passing partial_tail_reuse in snapshot; CoordinatorCacheManager.cc use_partial_tail requires request_layout.partial_tail_reuse and full token snapshot. Ordinary runtime requests without registration do not request precise nonaligned matching. Direct low-level allocator users can construct such snapshots, but the service path pairs them with registration.
- BlockTreeLoader.cc preciseLocalMatchLocked references matched source blocks before returning exact endpoint. Runtime keeps its request references after failed load.
- GenerateStateMachine.cc handleLoading returns to WAITING after load completion. handleWaiting reaches transitionToRunning, which calls prepareRequiredContinuation before setting RUNNING.
- CoordinatorCacheManager.cc prepareRequiredContinuation uses logicalBlockSpan and endpointBlockPosition, reserves replacement blocks, pins source, replaces live slot, and records copy mappings. NormalModelInputGatherer.cc adds registeredSaveCopyMapping to model inputs; NormalExecutor.cc invokes blockBatchCopyForForward before model forward.
- CP pages belong to rank logical_page % cp_size (CpTransferProjection.cc transferPageReadFromPeer); all ranks share block IDs, so an incomplete logical CP bundle needs one shared slot preserved even where an individual rank page is already complete.

Initial suspicion that CP FULL ceil skipped an unsafe slot was corrected: required continuation already owns that copy. Actual regression was LINEAR/non-round-robin floor detaching its partial slot too early and making continuation see a private but uninitialized block. The fix preserves that source until the established copy path.
