# Priority migration checkpoint

Candidate only: 4 of 19 old priority cases. `priority_queue::same_level_fifo`
maps `prio_same_level_fifo` on `single-nonbatch`. No legacy callable is invoked
or removed. The remaining normalization queue case is pending here. The separate preemption module is owned by agent4.
Default catalog registration is owned by the central framework task; this
checkpoint's tests explicitly merge the exported HANDLERS.

The YAML retains Q1: 1 Prefill / 4 Decode, PRIORITY ordering, SINGLE / NON_BATCH
from the selected profile, delivery quota 1, waiting cap 8, queue timeout omitted.
It sets Prefill to 3000 ms, waits 1.5 s, then starts seven priority=50 requests,
input=2048/output=2, with 0.15 s submission gaps including after the last start.
Schedule workers run concurrently. Each uses 90 s Schedule and 120 s Generate
RPC budgets; after all Schedule workers settle each consumer receives its own
35 s drain wait, all bounded by the instance/stage deadline. Omitted priorities
stay omitted. This first variant explicitly uses priority 50 for all seven.

| Old predicate | Explicit stages/checks |
| --- | --- |
| PR2 dispatch equals submitted request order and all seven succeed | terminal -> fifo.PR2 |
| Prefill dispatch arbitration | actual running_ms, then Schedule settled time rank, then wire RID |
| P6 every request completes | fifo.P6_terminal |
| P6 Master scheduler, P batch ledgers, D load empty within 30 s | owner_clean.all_owners_zero |
| Perf restore and owned environment hygiene | restore, teardown, registered fallback cleanup |

The cohort registers cleanup before issuing any RPC. Every RequestBatch has a
unique artifact, actual core consumer thread, completion event and terminal
verification; Schedule workers also publish an exit event and are joined.
Cleanup cancels owned transports, joins workers and verifies consumers. It does
not claim client cancellation proves server KV release.

Declared differences: core RPC failures/timeouts remain ERROR/TIMEOUT rather
than ordinary legacy failure text. Missing/ambiguous running_ms evidence is
ERROR instead of sorting the missing row last. Core requires business-finished
and consumer-exit evidence, stronger than the old stream's no-error completion.
Stage failure stops subsequent diagnostics (including the explicit owner-clean
stage); registered cleanup still runs. Environment teardown replaces old
best-effort perf/drain hygiene after a failed stage. Master cleanup uses the
current endpoint schema; this is not a claim about all historical schemas.

Validation: formal load_scenarios -> compile_scenarios -> execute_instance with
actual RequestBatch Schedule and consumer threads, external RPC/HTTP only faked.
FIFO passes; inverted dispatch fails; missing lifecycle is ERROR. No real Java
run, independent static signoff or old/new paired run is claimed at this point.

## Low-priority completion

`low_no_starvation` maps `prio_low_no_starvation` in the single-nonbatch
shared-profile configuration: 2P4D, FIFO ordering, queueTimeoutMs=60000,
no explicit delivery/wait cap overrides. Both Prefill workers receive the perf
control and restore. The family default is the shared profile; Q1 overrides
are scoped exclusively to same_level_fifo. An empty variant dict is not used
to erase inherited settings because the compiler merges config_overrides.
Fresh instance isolation replaces reuse of the runner shared environment.
Prefill50ms -> sync1.5s -> two waves of eight (30x4 then70x4), each
Schedule settled before the next 1.5s gap (including after the last request).
Each wave drains all consumers, checks owners independently within30s, then
waits2s (including the final wave). Final owner-clean30s precedes one P6
completion check over all16 records; each priority must complete8/8.
The latency split remains diagnostic; no PR8 deadline band is applied.
Tests use real RequestBatch threads and fake RPCs; low-case sleeps are skipped
in fixtures while exact configured gaps/quiet periods are separately asserted.
An unfinished stream fails the final completion check. Missing records raise
ERROR. Existing FIFO seven-request tests still run unchanged in scope; tests
load only priority_queue.yaml so independent preemption definitions coexist.

## Queue expiry

`queue_timeout_terminal` maps `prio_queue_timeout_terminal`, single-nonbatch
T1 1P4D/PRIORITY/SINGLE/NON_BATCH/delivery1/wait8/queueTimeout8000.
Prefill10000ms+1.5s -> priority70 placeholder Schedule settles -> at least one
Prefill waiting/running within6s at0.1s intervals -> 30,30,30,70 wave at0.15s
including last gap. All wave Schedule workers settle before the placeholder
35s drain, followed by each wave consumer's own35s drain. Schedule90 and
Generate120 remain unchanged. No extra owner-clean verdict is inserted: the
old timeout case only used hygiene in finally. Explicit restore and owned
teardown remain, with registered cleanup on failure.

`expiry.P6` requires the placeholder success and all four exact Schedule8511
responses. `expiry.PR8` uses max of three low Schedule-settled minus per-worker
submitted timestamps divided by8, via GradeReport at the instance grade;
strict1.25/normal1.50/loose2 come from the registry. No lower bound is added.
Whole-program external-RPC fixtures verify one actual consumer/four rejected
requests, wrong8503 FAIL and stage ordering. A measured-record fixture checks
1.4 ratio strictFAIL/normalPASS. Synthetic immediate expiry fixtures are not
real8s deadline measurements and do not claim Java coverage.

## Mixed priority order

`order_basic` maps `prio_order_basic` using the same Q1 as FIFO. P3000+1.5s,
placeholder50, pending6s/.1s, then30a/30b/50a/50b/70a/70b with .15s gaps.
All Schedule settle before draining placeholder then each peer, preserving
90/120/35 budgets. PR1 excludes placeholder and first peer30a from inversion
scoring; PR2 keeps30a inside same-priority FIFO and requires wave order
30a,70a,70b,50a,50b,30b. As in the old design_final_pattern the wave-only
shape does not add a new placeholder-first requirement. PR6 requires the
shape and all seven successful code200 outcomes. P6 combines terminal
checks and a separate30s owner clean (shared d955 correction required).
Priority inversion and same-level inversion are independently exercised:
swapping70a/70b leaves PR1 zero but fails PR2. All7 consumers are real core
threads with fake RPC; no Java coverage is claimed.
