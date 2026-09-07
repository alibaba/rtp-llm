# Priority migration checkpoint

Candidate only: 1 of 19 old priority cases. `priority_queue::same_level_fifo`
maps `prio_same_level_fifo` on `single-nonbatch`. No legacy callable is invoked
or removed. The remaining queue cases and priority_preemption are pending.
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
