# Core performance tests

The public unit suite was reduced from **50 files / 504 methods to 15 files / 138 methods** against baseline `5a163f51d9b28d8939c61a8b8f1a398b3cf1efc1`. All 138 approved class/method names remain; 366 other methods were deliberately deleted. The manual production entry `offline_bench_test.py` is excluded from these unit counts and remains unchanged.

Formatted unit-source plus shared-fixture size is **17,297 → 7,423 lines (57.1% smaller)**. `test_support.py` contains 262 lines of fixtures only, including fresh configuration section defaults with per-case overrides. Shared helpers have explicit domain names, and helper-only imports cannot discover duplicate TestCase classes. Internal imports of `prefill_profile_test._setup` and `_formal_bytes` remain available.

| Unit target | Methods | Retained behavior |
| --- | ---: | --- |
| `decode_executor_test` | 11 | DP rounds/admission, successful tokens, checkpoint continuation, TPOT and scheduling |
| `decode_multi_node_test` | 9 | current-run persisted completion, rank behavior, diagnostics/config delivery |
| `engine_session_test` | 9 | session startup/teardown, typed failure, one session across actual prefill/decode entry |
| `entry_test` | 12 | real entry/help/dry-run, early public rejection, runtime source closure |
| `offline_runner_test` | 6 | workload generation, entry cleanup/concurrency, cancellation metrics and failures |
| `plan_test` | 12 | canonical identity, immutability/round-trip, extension seams, deterministic planning/budgets |
| `prefill_profile_test` | 8 | isolated profiling, trace lifecycle, TPOT selection, formal artifacts unchanged |
| `prefill_report_test` | 7 | numeric reporting, diagnostic exclusion, artifacts/fallback, merge identity |
| `prefill_resume_test` | 9 | resume identity/locking, interruptions, atomic persistence and torn-tail repair |
| `prefill_runner_test` | 12 | reuse/seed failures, concurrent timing, grouped members, probes and external sessions |
| `result_envelope_test` | 9 | numeric evidence, schema round-trips, structured failure diagnostics and finalization |
| `runtime_adapter_test` | 8 | resolved launch contract, HTTP/DashSc wire parity, business failure and retry |
| `ssh_prefill_test` | 9 | mocked SSH/gang topology, args/env quoting, rank dispatch and public install |
| `unified_config_test` | 8 | representative schema/defaults, fingerprints, distribution paths, typed failures |
| `workload_test` | 9 | traffic/random export, deterministic sampling/distributions, prompt geometry |

The unit scope deliberately drops exhaustive schema/default and invalid-value permutations, docs/example enumeration, static and interactive chart appearance/layout assertions, detailed formula fitting tests, redundant lower-level wrappers and non-core legacy branches. Selected loops retain distinct operational modes, DP allocations, ranks, success/failure outcomes and essential numerical integrity checks.

Within retained methods, unknown-field locations shrink 5→1, CLI source conflicts 4→2, concurrent-evidence mutations 14→5, and invalid TPOT diagnostic fields 6→1. Representative explicit defaults replace a copied schema-default snapshot. Charts retain readable HTML, parseable SVG, positive latency/throughput data and missing-plotly fallback, without presentation labels or layout checks. Core reuse, round timing, successful-token throughput, partial failures, persisted evidence, TPOT search and SSH/gang startup contracts remain.

Validation used `/opt/conda310/bin/python`: baseline public discovery passed 504 tests; final public discovery passed 138 tests; internal discovery from the parent checkout with this public checkout on `PYTHONPATH` passed 232 tests. Independent actual discovery matches the approved set exactly, with no duplicate or missing IDs. Every baseline method has a retained destination or categorized deletion reason in the implementation audit. Both repository trees were searched for obsolete executable references, with none found; historical documentation references remain untouched.

Black and isort repository hooks pass. All six production Bazel library/binary definitions are byte-for-byte unchanged; only test-only rules, sources and data change. Exactly the 15 unit targets above remain alongside the unchanged manual benchmark target. All 15 core Bazel targets were executed and passed on the final test tree (64.133 seconds; 30 remote actions). Independent task and final integration reviews passed. Validation uses controlled test doubles rather than live GPU/model or multi-host workloads.
