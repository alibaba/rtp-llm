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
The optional `metric_whitelist` field maps only to
`FLEXLB_MONITOR_METRIC_WHITELIST`; it accepts 1..16 comma-separated metric
identifiers such as `flexlb_auto_tpm_request_count`. It does not change the
rendered scheduler configuration or open a general environment-variable channel.

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

Variants may supply a complete `stages` list instead of the top-level default.
Each such program gets its own prior-stage reference namespace and resource
budget; it cannot also use `stage_overrides`. This expresses distinct flows
explicitly, without choosing hidden Python case implementations. Variant execution
budgets, additional requirements, findings and narrowed `legacy_case_ids` are also
explicit. If the document omits top-level stages, every variant must provide them.

`PlanContext.environment` is an isolated copy of the variant's environment fields;
`PlanContext.profiles` is its selected profile tuple. Validators may reject
incompatible action/layout combinations before startup. These fields do not imply
that processes exist or that any runtime capability has been observed.

`master_layout: single|dual_standalone` selects one master or two actual standalone
masters at the leased A/B port groups; it does not activate ZK or claim elected HA.
`master_stable_window_s: 0` explicitly disables the usual stable-window prewarming
for cold-start experiments. Started process objects remain in the backend ledger
even when a restart helper clears the current environment slot.

List output includes separate selected `logical_scenarios`, `variants`, `instances`
and declared `checks` counts. Its filesystem storage key hashes the public ID to
avoid JVM `-Xlog` colon delimiters; explicit output roots containing colons are
rejected before process startup.

`environment_reconfigure` replaces the complete typed `config_overrides` for a
new environment within the same instance and leased ports. Worker counts, layout,
decision and dispatcher stay fixed. It first cleans all existing consumers and
processes; any failure prevents the next startup and retains failed callbacks
for final cleanup. A new epoch invalidates old live handles. Earlier scalar
results and explicitly historical snapshots remain available for comparison.
This is reconstruction, not runtime configuration reload. Allow at least 30 s
for intermediate cleanup plus the normal startup budget in the stage timeout.

`environment_startup_probe` accepts a valid typed base and one of three bounded
raw mutations: `removed_auto_tpm`, `fifo_default_priority`, or
`owned_without_cancellation`. It launches Java with the resulting raw config,
records owned Master PIDs/exit codes and private parser logs, then cleans the
attempt. Python validation failure does not count as Java rejection. `rejected`
requires a failed launch, an exited owned Master and a matching parser message;
`environment_absent` records the manager state **before** forced cleanup, so an
unexpected successful startup cannot pass that condition by being stopped later.
Probe outputs require explicit case checks; generic startup failure or missing
parser evidence is not a parser pass. A normal initial setup remains required,
which is an additional baseline construction compared with a negative-only case.

The first environment retains the existing artifact layout; later environments
write under `environment-epoch-N`. Each epoch retains its cleanup report, and the
root cleanup report remains the latest cumulative PID ledger. Startup probes
write raw configuration and observations even when they cannot complete.
All single-Master scenarios also use private per-epoch application/PV/FlexLB logs;
adapters can obtain the owned directory from `ctx.master_log_dir` or
`ctx.env.master_log_dir`. Dual-Master startup retains its separate directories
for each owner. No scenario needs to infer ownership from offsets in a shared
home-directory log.

RequestBatch stream errors record `stream.trailer_error_code` separately from
transport status and in-band `business_error_code`. `stream.error_trailer` retains
bounded original `grpc-status-details-bin` bytes as base64, length and SHA256.
Missing, duplicate, malformed or default-only protobuf payloads leave the typed
code unset with an explicit evidence status. Recording a CANCELLED transport does
not imply engine cancellation. The ordinary request wait still rejects failed
streams; expected-error scenarios must implement their own explicit verdicts.
