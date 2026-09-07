# KV migration checkpoint

`cache_local_index` now has explicit programs for all three contracts: prefix
continuity, eviction propagation and per-engine admission isolation. Each has
BATCH/NON_BATCH programs across all four profiles (six variants, twelve instances).
The global-holder family is also implemented below. Affinity has three candidate
contracts; its leader-saturation finding, capacity and churn remain pending. All old callables remain retained.

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


## Global holder programs

`cache_global_holders` contains five old contracts as ten explicit delivery
variants and twenty profile instances. These are candidate implementations;
paired Java execution remains pending.

| Old contract | Construction and retained predicates |
| --- | --- |
| shared_block_both_match | Same shared seed (2000ms, 1.5s, pending6s, drain30s, restore100ms), quiet3.5s within8s; both holders contain the family; twenty serial requests; P1 .65/.75/.85, P2 at least two workers, P6 all land inside the holder union |
| full_release_no_ghost | Shared seed and quiet; evict both full copies; quiet then no family key on either engine; twenty fires with120ms after every issue including the last; P1 spread |
| partial_release_redirect | Evict first holder only; quiet; first has no key and second has the entire family; ten serial continuations with P9 .95/.90/.80 |
| engine_down_cleanup | 3P/2D discovery_file; seed two holders, third has no family key; graceful removal with the original default60000ms drain; exact Master alive2 within30s; five serial continuations with P9; survivor still has all keys |
| sync_convergence | Two family0 admits, first eviction, family1 admit, second family0 eviction, two family2 admits and first family2 eviction; one final quiet window; family0 has no key, family1 and family2 have exactly their observed full-family holders; family0 twenty-fire P1 plus five serial P9 continuations for each surviving family |

The mixed stream does not insert extra quiet windows between mutations. Holder
matching distinguishes full-family membership from any-key residue. Dynamic
engine references come from actual Schedule landing addresses; they do not
assume which engine wins. The removed worker is explicitly excluded from the
final survivor snapshot, and the Master alive count has its own observation
stage; neither is inferred from a removal acknowledgement.

All cohorts require the declared sample count and verified consumer completion,
including fired waves whose old helper could otherwise hide a drain failure.
As with local-index programs, absent cache sets, duplicate request IDs and
unobserved/duplicate expected holder identities are ERROR. Cache eviction has
an additional immediate visible-effect gate. These are declared evidence
strengthenings. RPC/caller deadline differences still need paired execution.
Each YAML instance tears down its owned environment, so the legacy reuse-only
finally/add_engine restoration is replaced by owned-environment cleanup; no
shared environment is left with a missing worker or altered performance.

Three additional local tests execute all twenty compiled programs through
actual handlers with a small cache and transport model. Negative cases cover
stale routing after acknowledged eviction, foreign landing outside the holder
union, acknowledged eviction without effect, unavailable cache observations,
and duplicate/unobserved holder expectations. These model runs are not Java,
Master cache-index version proof, or permission to remove old callables.


## Affinity checkpoint: three of four contracts

`cache_affinity` currently maps prefix stickiness, hot-prefix tension and mixed
hit tiers (four variants, twelve profile instances). The supplemental
`kv_leader_saturation_spill` finding is still pending, so this family is partial.

All three use the original smoke environment (2P/4D, default cache/performance),
15s serial waits and fixed2s post-seed cache sync. They do not replace that fixed
wait with the3.5s observed-quiet contract used by eviction cases.

| Old contract | Explicit retained program and predicates |
| --- | --- |
| prefix_stickiness | A8keys/input8192 first seed with2000ms/1.5s/pending6s; distinct B8keys before first drain30s; restore100ms, sync2s, distinct gate; thirty serial requests in3:2 order (18 same-family,12 unique free); P9 .95/.90/.80 and free P2 at least2 workers, complete cohorts P6 |
| hot_prefix_tension | One16key/input16384 seed, sync2s, forty serial requests in7:3 order (28 same-family,12 unique free); P9 .95/.90/.80; M2 .88/.93/.96 with seed+all40 requests as the41-record denominator; P2 at least1 free request off-holder; P6 complete cohorts |
| match_mixed | Full8key/input8192 seed and sync2s then10 full continuations; half4key/input4096 seed and sync2s then10 input8192 requests with the same4-prefix plus4 unique suffix keys each; finally10 completely unique8key requests; full/half M3 .8/.7/.6 and zero-hit P2 at least2 workers with P6 complete cohorts |

Fresh keys are explicit disjoint per-instance namespaces. They preserve key
counts, prefix overlap and uniqueness; their numeric values are not generated
from wire request IDs. No latency-benefit verdict or cache-key-count verdict is
added: those were observational in the legacy cases. Holder names still derive
from actual landing addresses. M2 uses the original request-share denominator;
token share is equal only because every included request has the same input_len.

Existing mandatory terminal-evidence/deadline differences apply. Legacy
best-effort shared-environment restoration is replaced by owned instance
teardown. Three tests execute all twelve compiled programs with the cache model,
lock the3:2 interleave, unique half-hit suffixes and seed-inclusive denominator,
and demonstrate lost affinity and holder overconcentration FAIL. They are local
model fixtures, not real Java acceptance or completion of the fourth contract.
