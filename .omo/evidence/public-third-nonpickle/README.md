# Third commit: four non-pickle conflict resolutions

Rebase remains paused at e9b2736f8208f638b349aadc246555e0f21a238c. HEAD remains e7e24839331d24968cc1cb718e81ccf9a449bdd5. No rebase continue performed.

| Scenario | Invocation | Binary observable | Artifact |
|---|---|---|---|
| Python/stub syntax | ast.parse on two Python tests and isolated KVCacheConfig stub class; compare full stub parse against HEAD | 3 scoped PASS lines; full stub blocked on existing illegal None annotation in both HEAD and resolved file | syntax-check.txt |
| Four resolved paths whitespace | git diff --check -- four assigned paths | exit=0 | scoped-check.txt |
| Explicit staging | git add -- four assigned paths; git diff --name-only --diff-filter=U | only ConfigInit.cc and kv_cache_config_pickle_test.py remain unresolved | remaining-conflicts.txt |
| Staged whitespace | git diff --cached --check | exit=0 | staged-check.txt |
| Whole working tree | git diff --check | fails on intentional unresolved pickle markers | full-check.txt |
| Rebase paused | git rev-parse HEAD REBASE_HEAD | hashes above | heads.txt |

Resolved patch in resolved.patch: retain upstream v6 restore assertion in server_config_setup_test.py; retain both upstream shared-block detach rollback test and feature exact endpoint CP2/CP4 test; expose both enable_crc and enable_prefill_partial_tail_reuse in stub; retain CRC default/env/CLI/invalid tests plus tail switch tests.

PROVISIONAL: server_config_setup_test.py still asserts v8 len70 and derives old states with state[2:-6]. These version-sensitive expectations MUST be updated with ConfigInit after user decides legacy private pickle migration. This staging resolves text only; runtime behavior is not claimed.

Overlap inspection:
- CoordinatorCacheManager.cc:370-392 retains detach atomic reservation/refcount rollback and new localAllocTokens API. Risk for exact partial CP reuse: G17/B4/CP2 yields localAllocTokens=12 and recompute_begin=3, while endpointBlockPosition(full,17)=2. If failed async fallback retains exact G17, the shared partial FULL slot2 is excluded from detach. Requires runtime reproduction/call-path proof; not changed in four-file scope.
- StreamCacheResource.cc finalizeAllocatorLoad retains upstream failure detach and retry status, exact endpoint publication, and remote effectiveness calculated after publication.
- DecodeRpcServer.cc:427 retains separation of transferred KV from historical cache hits; new handoff SWA masking only changes copied transfer block tables.
- No obsolete block_aligned_input_length, cp_slot_mapper_, report_match_latency, match_latency_us, or recordCacheHitTokens references found in cache and stream code. First-key prefill metrics gating and global request denominator retained.
- deepseek_v4_model.py:1180 retains cuda_graph_selection_mode and explicit rejection of generation-prefill capture; sparse-save capability is added separately at class start.
- deps/pip.bzl:45 retains /opt/conda312/bin/python3; .bazelrc:388 onward retains PPU CUDA13/platform settings. Third commit does not alter these platform configuration files. New sparse LINEAR save tests use L20_CU13. This is source inspection, not PPU or Python 3.12 runtime validation.

No runtime tests performed; root owns execution. No changes outside four assigned source paths plus this evidence directory. Existing repository conventions and module responsibilities preserved; no new parsing boundary or generic helper introduced.

Full generated stub syntax check is blocked by pre-existing `None: typing.ClassVar[QuantMethod]`; directly reproduced on HEAD and resolved stub. No unrelated stub repair attempted.
