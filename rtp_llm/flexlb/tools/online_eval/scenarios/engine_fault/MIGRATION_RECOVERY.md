# Engine recovery migration

All nine legacy cases are mapped: one logical family, nine variants, 31 profile
instances and 298 checks. Independent acceptance and integration are separate
from local compilation; this is not real Java execution evidence.
Base: `9d8576c44bf6dda4191f5d9070c3cc929b7fc5a0`; isolated sync-log support:
`3ac1bab6e9` (copied locally as `603c212206`).

| Legacy case | State | Contract and timing |
|---|---|---|
| recovery_generation_bump | Implemented | Prior 95 s drain is observation; six baseline successes; stop P0; transport retire <=30 s; restart/alive>=2 <=30 s; reconnect3 s; created count grows; target P batches and members zero before recovery request; recovery succeeds. |
| status_gap_no_bump | Implemented | Prior 95 s drain observation; four baseline successes; status_no_respond on P0 for0.045 s; clear; wait2 s; created count must not grow, discovered=alive=2; four requests succeed. |
| down_phases | Implemented | Baseline20/20, takeover>=18/20, recovery>=19/20; Master HTTP200 at three phases; eviction/recovery30 s; reconnect3 s; TTFT index p50 <=1.5x baseline; closing Master drain95. |
| flap | Implemented | Six stop0.8/start0.4 cycles under serial cold flow; HTTP200 each cycle; final discovered/alive convergence; recovery>=19/20; Master drain95; flow nonempty and >=50% success. |
| crash_after | Implemented | EnqueueBatch only; exactly one stopped engine; disarm surviving engines before takeover5>=3; alive drop/restore30; settle2/3; residue<=1+failed takeover then no growth8 s; selected Prefill engines clean. |
| recovery_kv_resync | Implemented | bw/sb/wn only; seed ten keys, real holder check8; quiet4.5; retire/restart/generation; intact holder >=4/5; evict family/absence8/quiet4.5; spread old-holder<=3/5. |
| recovery_no_resurrect | Implemented | EnqueueBatch only; eight manual Schedule attempts, >=4 routed; slow Prefill2000 ms/settle1.5 and0.5; crash target next EnqueueBatch within15; retire30/restore30/settle3; target ledgers zero and five engine wipe fields; consume2 s per routed request with cancel fallback, zero successful old targets; engines clean10, Master drain95, recovery. |
| status_gap_long_retire | Implemented | Slow Prefill2000 ms; eight routed/manual payloads; status gap until retire<=15 then hold1 s; resume/alive30/settle3; new generation; consume5 s per routed request (outcomes observation); Master drain95 and recovery while slow perf remains. |
| recovery_kv_usage_reset | Implemented | Empty cache baseline/quiet4.5/used0; pressure4000000 observed8; retire30/restart/alive30/settle3/new generation; used_after0 remains observation only; three unique-key successes with zero lack_mem delta; pump target accepted growth15; recovery. |

Only the new action module, this document, the family YAML and owned tests are
modified. Existing case functions and shared compiler/backend/catalog remain
under the integration owner. The new actions never call the old case bodies.

Master sync logs must be under the current artifact directory. Marks capture
file identity and byte offset; missing files, replacement, truncation, invalid
UTF-8 or >8 MiB deltas are ERROR, never zero generations. Counts require the
actual advertised endpoint and exact generation/retirement line shapes, avoiding
address-prefix matches. This explicitly strengthens missing-evidence handling
compared with the old helper's silent zero fallback.

Each request worker holds its concurrency slot through the legacy result-observation
window, not merely through Schedule. The legacy `stream_timeout_s` is that wait,
while the actual stream RPC retains its independent 60-second deadline. A timed-out
observation does not cancel the transport; its success bit is frozen and cannot
be changed by a later completion. Pending consumers remain owned and must provide
terminal fields and independently verified exit during cleanup. Known gRPC
failures are retained as failed observations; untyped failures are ERROR.

The TTFT batch keeps a first-output wait of 15 seconds followed by a 15-second
end wait. Its TTFT uses the polling observer timestamp, preserving the old
2 ms observation granularity; the consumer receive timestamp remains separate raw
evidence. `generate_payload` independently declares match_schedule versus
legacy_default: NON_BATCH TTFT keeps Generate's default payload while Schedule
carries output2 and three keys. E3/E5 manual payloads likewise open the legacy
default Generate payload only in their consume stage. A manual NON_BATCH Schedule
is routing only and is never presented as evidence of engine execution.

E2 keeps the original three-profile selection (bw/sb/wn), full ten-key owner
construction, 4.5-second cache quiet intervals, >=4/5 intact-holder and <=3/5
wiped-holder bars. Eviction control returns acknowledgement only; absence is
polled independently for eight seconds. E6 uses the original ABSOLUTE
`/set_kv_pressure` control, not the additive `/inject kv_pressure` channel. Its
post-restart `kv_tokens_used` sample stays observational; the hard checks remain
generation turnover, first-wave success and zero LACK_MEM delta, accepted-counter
growth and recovery. The fixture where used_after stays nonzero passes this
intentional old boundary.

Crash-after trigger, takeover and recovery explicitly keep the old default output
length of 10; other families retain their explicit output length of 2. Alive,
cache ownership, KV usage and drain polls keep the old 0.5-second cadence;
retirement and topology polls keep 0.2 seconds. Predicate polling does not add
a new sample at an expired observation deadline. Missing-source handling and
owned cleanup remain explicit differences, so wall-clock execution is not claimed
to be identical.

Crash controls arm either the first EnqueueBatch or the selected engine's current
EnqueueBatch counter plus one. The actual stopped set chooses restart targets;
survivors are disarmed before takeover. The no-resurrection path selects only
successful actual routes, requires >=4 routed payloads, observes every selected
endpoint retire with its own 30-second window, and checks all five wipe fields
before opening old payload streams. Per-request consume observations retain the
old 2/5-second windows and Master/extra-NON_BATCH-worker cancellation fallback;
a failed Master Cancel still skips worker Cancel as in the old helper;
subsequent client transport cleanup supplies mandatory consumer exit proof without
changing the already-captured resurrection verdict. The uncertain crash residue
bound is 1 + failed takeover observations, followed by the original 8-second
non-growth sample. It is not a global zero assertion.

Request dispatch stage budgets cover every serial Schedule plus its legacy result
window, with execution overhead allowance. The accepted pump also permits a
request begun just before its 15-second window to finish Schedule and its
10-second observation; the loop keeps the old 0.2-second cadence and final
counter read. Execution caps do not shorten the five serial
KV samples or the two TTFT waves. Requests, log readers, controls and repeated
probes are bounded. The two serial
pumps allow at most 256 attempts (above the 0.2-second cadence's maximum in their
30-second configurable window). Log reads cap deltas at 8 MiB; observations cap
4,000 frames / 32 MiB. Shared old environments become fresh per-instance owned
environments; teardown replaces shared-state restoration after failures. Explicit
perf/pressure restore callbacks remain owned and execute before environment
teardown. Old callables remain retained, and no old case body is invoked.

Local validation: 28 focused tests pass, including 72 full-program executions
across all nine variants and positive/negative profile combinations. Additional
unit evidence covers exact endpoint log matching, truncation/missing-log errors,
missing target ledgers, independent Schedule/Generate shapes, upper-index TTFT
p50 and the real-thread observation/transport deadline separation with late
success remaining failed. These deterministic Python models do not establish
real Java performance, all-service correctness or replacement eligibility.

## Executable predicate index

### generation_bump

Legacy: `engine_fault_recovery_generation_bump`.

`baseline_succeeds`, `transport_retired`, `alive_back`, `generation_is_new`, `recovered_prefill_batch_ledger_zero`, `recovered_prefill_member_ledger_zero`, `recovery_succeeds`

### status_gap_no_bump

Legacy: `engine_fault_status_gap_no_bump`.

`baseline_succeeds`, `no_new_generation`, `discovered_intact`, `alive_intact`, `post_gap_succeeds`

### down_phases

Legacy: `engine_fault_down_phases`.

`baseline_succeeds`, `baseline_health_http_200`, `survivor_only_alive`, `downtime_health_http_200`, `takeover_at_least_ninety_percent`, `alive_back`, `recovery_at_least_ninety_five_percent`, `ttft_recovers`, `recovered_health_http_200`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### flap

Legacy: `engine_fault_flap`.

`master_cycle_1_http_200`, `master_cycle_2_http_200`, `master_cycle_3_http_200`, `master_cycle_4_http_200`, `master_cycle_5_http_200`, `master_cycle_6_http_200`, `flow_availability`, `topology_discovered`, `topology_alive`, `post_flap_recovers`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### kv_resync

Legacy: `engine_fault_recovery_kv_resync`.

`seed_succeeds`, `holder_has_whole_family`, `holder_retired`, `holder_alive_back`, `holder_generation_bumped`, `memory_intact_holder_survives`, `wiped_family_absent`, `memory_lost_old_holder_spreads`

### kv_usage_reset

Legacy: `engine_fault_recovery_kv_usage_reset`.

`baseline_usage_zero`, `pressure_construction_observed`, `pressure_generation_retired`, `alive_back`, `new_capacity_generation`, `first_wave_succeeds`, `first_wave_no_lack_mem`, `target_receives_traffic`, `recovery_succeeds`

### crash_after

Legacy: `engine_fault_crash_after`.

`exactly_one_crashed`, `master_observed_loss`, `survivor_serves_sixty_percent`, `crashed_engine_rediscovers`, `recovery_succeeds`, `residue_window`, `prefill_engines_clean_inflight`, `prefill_engines_clean_leaks`

### no_resurrect

Legacy: `engine_fault_recovery_no_resurrect`.

`routed_payload_minimum`, `all_targets_crashed`, `retire_all`, `all_prefills_alive`, `new_prefill_batches_zero`, `new_prefill_members_zero`, `wiped_running`, `wiped_inflight`, `wiped_cache_keys`, `wiped_held_blocks`, `wiped_accepted`, `old_requests_never_complete`, `engine_drain_inflight`, `engine_drain_leaks`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `recovery_succeeds`

### status_gap_long_retire

Legacy: `engine_fault_status_gap_long_retire`.

`long_gap_retires`, `prefill_alive_back`, `long_gap_creates_generation`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `recovery_succeeds`
