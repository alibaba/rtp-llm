# Engine fault migration checkpoint

`engine_rpc_fault.yaml` implements explicit programs for all four planned RPC
contracts, with five variants and fourteen profile instances. The separate
recovery family remains pending. All legacy cases remain selected until paired
execution and independent contract review are complete.

| Legacy contract | Explicit program and checks | Profiles |
| --- | --- | --- |
| engine_fault_enqueue_delay | baseline request, inject enqueue_delay=1500 on both P workers, delayed request, clear, recovery request; three success checks, total latency delta >=1.2s, recovery delta <=1.0s, scheduler count=0 within10s | batch-window, single-batch |
| engine_fault_generate_delay | same stages with generate_delay=1500; three success checks, stream first-output latency delta >=1.2s, recovery delta <=1.0s; scheduler count=0 within10s for BATCH only | all four profiles |
| engine_fault_no_respond | inject both P workers, one5s stream probe, require observed request fault, clear, cancel successful delivery, wait3s, successful fresh recovery request, conditional scheduler-only cleanup95s | all four profiles |
| engine_fault_enqueue_error | same program with enqueue_error and10s probe; incomplete business result also satisfies original require_error_detail condition | all four profiles |

Environment is the smoke shape: 2P/4D, default perf, unmodified selected profile,
input2048/output10. Each request retains its RID and terminal consumer evidence.
Control acknowledgements do not establish latency effects. Missing records,
missing terminal timestamps or reused cohorts produce ERROR. Measured threshold
violations produce FAIL. Neither is a declared finding.

Parity work remains before replacement: legacy TTFT starts after `start_stream`
returns whereas the new record starts immediately before opening the stream RPC;
legacy first-output/end waits and Schedule RPC limits must be compared with the
core consumer's bounded deadlines. Legacy shared-environment preliminary TTL
draining is absent from these isolated programs; the final10s owner assertion is
preserved. No paired Java execution has yet established timing/config equivalence.

The fault probes use an explicit request-level RPC status policy (UNKNOWN,
INTERNAL, UNAVAILABLE, DEADLINE_EXCEEDED). Internal errors, missing source data,
unlisted statuses and stage deadlines remain ERROR/TIMEOUT; no broad exception
is counted as proof that the injected fault worked. A request-level deadline
requires completed consumer/transport evidence. These are deliberate stronger
evidence boundaries than the old catch-all exception handler. The5s/10s observation
windows now also bound the stream RPC itself; paired execution must confirm the
transport difference before replacement. Server Cancel only applies when
Schedule returned success, and scheduler cleanup only applies to successful
BATCH delivery, preserving the original resource-owner conditions.

Eight local tests execute the shipped programs through the real compiler/runtime,
fault lifecycle and measurement handlers with fake external RPCs. They verify
all fourteen instance programs, threshold FAIL, missing-evidence ERROR, cohort
identity rejection, unexpected transport ERROR and stage TIMEOUT preservation.
The eight error-program instances run actual RequestBatch consumer threads.
They are framework tests, not service performance results.
