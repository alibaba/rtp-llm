# KV migration checkpoint

`cache_local_index` implements the continuous-prefix contract first. The other
two contracts in this family and the four other KV families remain pending.
The original `kv_pe_prefix_continuity` callable remains retained.

| Original operation or assertion | Explicit stage |
| --- | --- |
| 2P/2D, decode12blocks, input10240/output2 | environment and each request shape |
| Slow both P workers to2000ms, wait1.5s | slow, perf_sync |
| First Schedule holds a live engine ledger, second lands elsewhere | seed_first, first_pending, seed_second, two_holders |
| BATCH first request performs no Fetch until drain | continuity_batch overrides seed_first consume=deferred |
| Both seeds finish; restore100ms; key-set quiet >=3.5s within8s | first/second_terminal, restore_perf, seed_quiet |
| e1 loses key2; e2 loses keys9/10 | carve_gap, carve_tail |
| Continuous runs must equal1 and8, not total cached-key counts | carve_quiet, gap_prefix, tail_prefix |
| Five serial continuations; no request errors | continuation0..4 request/wait pairs, affinity.P6 |
| M3 concentration on contiguous holder | affinity.M3, original bands strict0.8/normal0.7/loose0.6 |

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

Five local tests execute all four shipped profile programs through actual
handlers/compiler/runtime with fake external transports. Wrong carve shape
fails before continuations; missing key sets error; a0.6 measured concentration
passes loose and fails normal/strict. Real Java paired execution is still needed
before replacing the retained callable.
