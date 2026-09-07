# KV migration checkpoint

`cache_local_index` now has explicit programs for all three contracts: prefix
continuity, eviction propagation and per-engine admission isolation. Each has
BATCH/NON_BATCH programs across all four profiles (six variants, twelve instances).
The four other KV families remain pending. All three old callables remain retained.

| Original operation or assertion | Explicit stage |
| --- | --- |
| 2P/2D, decode12blocks, input10240/output2 | environment and each request shape |
| Slow both P workers to2000ms, wait1.5s | slow, perf_sync |
| First Schedule holds a live engine ledger within the original6s helper limit, second lands elsewhere | seed_first, first_pending, seed_second, two_holders |
| BATCH first request performs no Fetch until drain | continuity_batch overrides seed_first consume=deferred |
| Both seeds finish; restore100ms; key-set quiet >=3.5s within8s | first/second_terminal, restore_perf, seed_quiet |
| e1 loses key2; e2 loses keys9/10 | carve_gap, carve_tail |
| Continuous runs must equal1 and8, not total cached-key counts | carve_quiet, gap_prefix, tail_prefix |
| Five serial continuations; no request errors | continuation0..4 request/wait pairs, affinity.P6 |
| M3 concentration on contiguous holder | affinity.M3, original bands strict0.8/normal0.7/loose0.6 |
| Eviction: capacity16, prime, quiet3.5s, positive-control same holder, evict full family, quiet3.5s, no remaining family keys | evict variants through eviction_membership |
| Eviction: twenty fired requests,120ms after every fire including the last; BATCH Fetch deferred until drain | wave, wave_terminal; stream/drain budget30s |
| Eviction: max-share P1 .65/.75/.85 | spread.P1; all twenty terminal records are additionally required by spread.P6 |
| Isolation: seed A, pending6s, drain, seed B, require distinct; restore100ms then slow B5000ms and settle1.5s | isolation variants through b_perf_sync |
| Isolation: four families explicitly land on A, restore B, quiet3.5s; A contains all40 keys and B none | admit0..3, a_membership, b_isolation |
| Isolation: ten serial family continuations, P9 .95/.90/.80, final B contains neither A's four families nor family0 | fidelity, final_b_isolation |

The selected instance grade controls GradeReport; all bands and the achieved
grade accompany raw request records. Construction failures block later phases.
Cleanup still runs after failures. The key-set quiet window proves observed
stability; it does not directly prove the master received a particular version.

Added strict evidence: absent key sets are ERROR, duplicate request cohorts are
ERROR, eviction must visibly remove its requested keys, and all five terminal
records must carry consumer-exit verification. These strengthen the old partial
loop result boundary; they are explicit changes, not a claim of byte-for-byte
equivalence. No legacy assertion is removed. Engine pending is aggregate state,
as in the old helper, and is not claimed as per-RID ownership evidence.

Serial request wait/stream budgets remain15s; fired-wave and first-seed drains
use30s. The consumer RPC itself is also bounded, unlike the legacy independently
longer RPC with a shorter caller wait; this transport difference requires paired
execution. The isolation seed preserves the old drain-before-second-seed order;
its distinct-holder construction gate can fail and is never forced to pass.

Seven local tests execute all twelve shipped profile programs through actual
handlers/compiler/runtime with fake external transports. Wrong carve shape
fails before continuations; missing key sets error; a0.6 measured concentration
passes loose and fails normal/strict. Stale post-eviction concentration fails P1;
foreign final cache membership fails the separate isolation assertion. Real Java
paired execution is still needed before replacing the retained callables.
