# Engine recovery migration

This is a partial checkpoint: 4 of 9 legacy cases, 4 variants and 16 profile
instances. It is not full family delivery or real Java execution evidence.
Base: `9d8576c44bf6dda4191f5d9070c3cc929b7fc5a0`; isolated sync-log support:
`3ac1bab6e9` (copied locally as `603c212206`).

| Legacy case | State | Contract and timing |
|---|---|---|
| recovery_generation_bump | Implemented | Prior 95 s drain is observation; six baseline successes; stop P0; transport retire <=30 s; restart/alive>=2 <=30 s; reconnect3 s; created count grows; target P batches and members zero before recovery request; recovery succeeds. |
| status_gap_no_bump | Implemented | Prior 95 s drain observation; four baseline successes; status_no_respond on P0 for0.045 s; clear; wait2 s; created count must not grow, discovered=alive=2; four requests succeed. |
| down_phases | Implemented | Baseline20/20, takeover>=18/20, recovery>=19/20; Master HTTP200 at three phases; eviction/recovery30 s; reconnect3 s; TTFT index p50 <=1.5x baseline; closing Master drain95. |
| flap | Implemented | Six stop0.8/start0.4 cycles under serial cold flow; HTTP200 each cycle; final discovered/alive convergence; recovery>=19/20; Master drain95; flow nonempty and >=50% success. |
| crash_after | Pending | EnqueueBatch only; exactly one stopped engine; disarm surviving engines before takeover5>=3; alive drop/restore30; settle2/3; residue<=1+failed takeover then no growth8 s; selected Prefill engines clean. |
| recovery_kv_resync | Pending | bw/sb/wn only; seed ten keys, real holder check8; quiet4.5; retire/restart/generation; intact holder >=4/5; evict family/absence8/quiet4.5; spread old-holder<=3/5. |
| recovery_no_resurrect | Pending | EnqueueBatch only; eight manual Schedule attempts, >=4 routed; slow Prefill2000 ms/settle1.5 and0.5; crash target next EnqueueBatch within15; retire30/restore30/settle3; target ledgers zero and five engine wipe fields; consume2 s per routed request with cancel fallback, zero successful old targets; engines clean10, Master drain95, recovery. |
| status_gap_long_retire | Pending | Slow Prefill2000 ms; eight routed/manual payloads; status gap until retire<=15 then hold1 s; resume/alive30/settle3; new generation; consume5 s per routed request (outcomes observation); Master drain95 and recovery while slow perf remains. |
| recovery_kv_usage_reset | Pending | Empty cache baseline/quiet4.5/used0; pressure4000000 observed8; retire30/restart/alive30/settle3/new generation; used_after0 remains observation only; three unique-key successes with zero lack_mem delta; pump target accepted growth15; recovery. |

Only the new action module, this document, the family YAML and owned tests are
modified. Existing case functions and shared compiler/backend/catalog remain
under the integration owner. The new actions never call the old case bodies.

Master sync logs must be under the current artifact directory. Marks capture
file identity and byte offset; missing files, replacement, truncation, invalid
UTF-8 or >8 MiB deltas are ERROR, never zero generations. Counts require the
actual advertised endpoint and exact generation/retirement line shapes, avoiding
address-prefix matches. This explicitly strengthens missing-evidence handling
compared with the old helper's silent zero fallback.

Each complete-request worker holds its concurrency slot until the consumer's
terminal fields and independent exit signal are verified. Known gRPC failures
remain failed request records; untyped errors remain ERROR. `generate_payload`
separately declares match_schedule versus legacy_default. Legacy NON_BATCH TTFT
must keep its original Generate default shape even when Schedule carried output2
and three keys; no silent workload correction or equivalence claim is made.
Manual NON_BATCH Schedule is routing only until a Generate stream is opened.

Local checkpoint validation: 13 focused tests pass. Eight program tests each
execute four profiles, including positive generation/short-gap programs and
negative missing-generation, residual-P-member, jitter-churn and incomplete-topology models. Down phases and flap execute their full programs as well. No
remote lease or Java process was started for this checkpoint.
