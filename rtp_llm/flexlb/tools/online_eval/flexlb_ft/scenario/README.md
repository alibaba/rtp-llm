# Declarative scenarios

A file defines one scenario. Explicit variants and profiles compile into separate
`scenario_id::variant_id::profile` instances; definition counts are not instance
counts. Existing Python cases remain registered until independent contract mapping
and execution evidence establish replacement coverage.

`scenario_runner.py --source scenarios --list-json` validates files and lists
metadata without importing the Java process harness or starting services.
`--instances` selects exact IDs; empty, duplicate and missing IDs fail with exit 2.
The compile-only module also renders the core configuration for review.

Execution requires `--out-dir` and the parent's `--lease-json`, with its five port
environment variables already set. The parent owns lane locks/preflight; the child
verifies the declared windows and worker capacity before starting its private
Java mock environment. Each instance uses a fresh environment. Dynamic additions
through `ctx.ops.add_engine` count attempts and reject explicit ports or exhausted
budgets. An adapter must declare its maximum additions and use this facade.

Environment fields include worker counts, cache block counts, controlled
`config_overrides`, `discovery: file|discovery_file`,
`perf_preset: default|fault_env` and `debug_enabled: true|false`. Profile axes come
from the existing flexlb_cfg generator. Arbitrary imports, expressions, raw config,
master environment variables and undeclared backend topologies are rejected.

Stages execute in order. Typed references use
`{$ref: stages.submit.output.requests}` and can only reference earlier outputs.
Resource handles are registered identities scoped to an environment epoch;
historical resources may only be read explicitly with `allow_stale=True`.
Setup, request, wait, cancel, check and teardown are builtin actions. Additional
adapters are registered explicitly in `catalog.py`; data files cannot import code.
Every scenario must declare a check, and executing zero checks cannot produce PASS.

A request stage issues a finite batch. `consume: deferred` requires batch dispatch
and performs no Fetch until a wait stage. Client transport cancellation, owner
Cancel RPCs, business FINISHED and consumer exit remain separate record fields.
Cleanup cancels owned transports and joins their consumer threads before stopping
owned environment processes. It does not claim that client cancellation proves
server-side business completion. `process-cleanup.json` records owned and remaining
PIDs, while the request artifacts retain incomplete and errored records.

Stage and instance budgets apply to actual RPC deadlines and the child main
thread's POSIX wall-clock guard. Cleanup gets an independent budget and runs in
reverse registration order, including after partial setup or submission. SIGTERM
requests cancellation and unwinds active stage work; the parent must reserve the
cleanup budget before its final forced termination. A failed callback, unjoined
consumer or unreaped process cannot produce a green result. Custom adapters must
implement cancel and bounded joins for any background work they create.

Ordinary FAIL blocks dependent stages. Collect independent measurements before a
final verdict stage when all evidence must survive an assertion failure. Only
explicit check IDs can be declared findings; setup, evidence, timeout and cleanup
errors remain errors. Exit 0 includes PASS and declared findings; actual failures
or cleanup errors yield 1, invalid configuration or selection yields 2.

The child writes `scenarios.json` with schema version, summary and instance rows,
plus per-instance `result.json` and artifacts. Listing, parsing, dryrun and fake
transport tests are not substitutes for the required real Java mock execution.
