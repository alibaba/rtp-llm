# Status protocol migration

Two explicit YAML programs cover all 25 legacy status cases and the two added
Master-debug/noFetch cases. Profile declarations are retained: 27 variants expand
to 51 instances. See [coverage.md](coverage.md) for each old predicate, its named
checks, observational items, findings and explicit semantic changes.

The implementation uses `status_prepare`, `status_dispatch`, `status_control`,
`status_perf`, `status_sample`, `status_check` and `status_outcomes`. The YAML lists
every experiment stage and assertion. It never dispatches a legacy case function.
The shared catalog owner registers the exported `HANDLERS`; this change does not
edit the global catalog or the 29-scenario inventory.

Prepared IDs are bound before injections. Submission concurrency is bounded and
uses the core RPC driver. Deferred requests do not Fetch until `wait`. A consumer
must supply its own done signal and exit record, then be joined; cancelled or
incomplete records cannot satisfy expected-fault checks. Only explicit typed RPC
statuses are allowed by a fault cohort. Unknown Python errors and stage deadlines
remain ERROR/TIMEOUT. Each control mutation installs epoch-scoped cleanup before
its first HTTP request, and all targets are cleared even after partial failure.

Snapshots retain raw owner fields, source timestamps and environment epochs.
Required source failures are errors, with partial artifacts retained. Decode
`total_load` and its individual admission layers are read from actual fields;
missing legacy fields are never silently interpreted as zero. Prefill batches,
Prefill members, Decode admission and mock engine lifecycle remain distinct owners.

Verification so far:

- 91 local scenario tests pass, including 11 status-specific tests for prepared
  cohorts, typed error boundaries, consumer exit evidence, source failures,
  independent owner metrics, cleanup and the complete legacy/profile mapping.
- Isolated host-111 Java mock runs used the unchanged 9821d9dc73 Java sources and
  matching hashed JARs. ACK multi-error, foreign batch ID and debug tombstone
  passed. noFetch and fetch-error passed on targeted follow-up runs; all cleanup
  steps passed and owned processes exited.
- The first noFetch run reported ERROR for an incomplete required debug snapshot.
  It remains evidence of sampling sensitivity; the later PASS did not relax
  completeness. The first fetch-error run reported ERROR because its declared RPC
  allowlist expected INTERNAL. The mock's RuntimeException actually maps to typed
  gRPC UNKNOWN, now explicitly declared only for that fault.
- Other variants/profiles are compiled and mapped, not claimed remotely passing.
  No GPU test, load test, production diagnosis or C++/KV-lifetime test was run.

The adapter requires the shared consumer-exit fix `48a520ddf4` (included in newer
core `81286a7e7d`) and the variant-program interface `32cc65a1b0`. Integration
should take only the status-specific commits, not duplicate their core dependency
cherry-picks. The old Python registry remains available for comparison during
migration.
