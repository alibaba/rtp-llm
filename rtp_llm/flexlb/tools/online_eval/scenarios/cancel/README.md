# Cancellation lifecycle and fence scenarios

All 19 legacy cancel cases are represented by two YAML programs: 28 variants,
66 profile instances and 524 checks. The original callables remain intact.
Independent acceptance and remote correctness are separate from compilation.
The [contract mapping](coverage.md) records every old predicate, ignored observation,
owner boundary and explicit representation difference.

The cancellation adapter reuses the shared bounded request driver and prepared
cohorts. YAML explicitly chooses Master Cancel, direct original-worker Cancel,
Schedule.future transport cancellation, or Fetch/Generate stream interruption.
Engine RPC receipt, client thread exit, business completion and each Master/engine
owner are checked independently. Direct Enqueue fence probes use the original
scheduled route and send one RPC; TCP readiness checks do not retry admission.

Baseline is core `7120c1ff19446f694cbd99c5c9c5545bbd11e0c2`, shared status corrections
through `3a79155e4c`, and core preemption/effective-axis support `eba715e911`.
Only this family's files are delivered here; the global catalog is registered by
the integration owner. Importing the adapter does not mutate the global catalog.

Local validation at the complete mapping checkpoint: 19 focused cancellation
regressions and 226 total `test_scenario*.py` tests pass. Tests exercise real Python
consumer/submission threads with fake transports, typed and late receipts,
source completeness, per-owner state, same-RID direct probes and all 19 legacy
mappings. No remote cancellation run has been made for this checkpoint; this is
not a claim that all 66 Java mock instances pass, and carries no C++/GPU KV claim.

Independent static acceptance covers all 19 old contracts at `72c867b94c`.
Integrated commit `8e1c46a4f7` contains byte-identical owned implementation/YAML
files and registers all ten cancellation handlers in its default catalog. A clean
archive of that commit compiles the two programs to 28 variants, 66 instances and
524 checks without adding handlers manually; its original 15 cancellation tests
pass. With the new program-test file only overlaid, all 19 cancellation tests pass.
The four program tests execute the complete unknown-ID YAML across four profiles
(16 executions): correct result, wrong found response, unexpected ledger mutation,
and missing owner data. They verify ordinary FAIL versus ERROR and successful
cleanup. These are deterministic Python fixtures, not Java or all-program runs.
