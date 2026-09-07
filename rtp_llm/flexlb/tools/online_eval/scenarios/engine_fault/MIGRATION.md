# Engine fault migration checkpoint

`engine_rpc_fault.yaml` currently implements delay programs for two of the four
planned RPC contracts. The no-response and enqueue-error programs are pending.
The separate recovery family is also pending. All legacy cases remain selected.

| Legacy contract | Explicit program and checks | Profiles |
| --- | --- | --- |
| engine_fault_enqueue_delay | baseline request, inject enqueue_delay=1500 on both P workers, delayed request, clear, recovery request; three success checks, total latency delta >=1.2s, recovery delta <=1.0s, scheduler count=0 within10s | batch-window, single-batch |
| engine_fault_generate_delay | same stages with generate_delay=1500; three success checks, stream first-output latency delta >=1.2s, recovery delta <=1.0s; scheduler count=0 within10s for BATCH only | all four profiles |

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

Four local tests execute the shipped programs through the real compiler/runtime,
fault lifecycle and measurement handlers with fake external RPCs. They verify
all six instance programs, threshold FAIL, missing-evidence ERROR and cohort
identity rejection. They are framework tests, not service performance results.
