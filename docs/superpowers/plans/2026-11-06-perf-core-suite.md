# Core Perf Regression Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Reduce public perf unit coverage from 50 files / 504 methods to 15 files / 138 core methods, emphasizing real behavior and reducing maintenance cost.

**Architecture:** Retain the explicit core selection and merge it into 15 behavioral suites. Delete non-selected tests instead of hiding them in helpers, disabled discovery, inherited test classes, or large parameter loops. Keep small domain fixtures and only relevant representative assertions.

**Tech Stack:** Python unittest; Bazel py_test and testonly support libraries.

**Spec:** User approved the 12–15 file / 100–150 test target, dropping parameter combination enumeration, plot layout/controls, documentation text/example counts and duplicated layer tests while preserving core prefill/decode, metrics, SSH, cleanup, persistence/resume, and necessary public dependency boundaries.

## Global Constraints
- No production behavior/source changes, no changes to manual offline_bench_test.py, and no weakening tests to force green results.
- Preserve the selected behavioral assertions, except simplify selected chart generation test to valid artifacts, essential data, and missing-plotly fallback; drop appearance/labels/geometry checks.
- Production config/model/README example files are not golden unit-test inputs. Fixtures use minimal shared valid config plus case-specific overrides; keep necessary workload sizes/measurements where those are the tested behavior.
- No broad parameter invalid-value permutations, brittle exact error strings, HTML/SVG layout checks, or documentation text assertions in the retained core suite. Small existing loops that test genuinely distinct core modes/ranks/outcomes are allowed.
- Preserve existing local worktree cache_grid_main, user untracked files (including docs/architecture.md), parent gitlink and public-only history. Do not push or run live model/GPU/SSH/Aone workloads.
- New support is testonly, contains no discarded test methods or TestCase classes, and must not enter production libraries. Do not create one helper module per deleted test file.
- Preserve internal backend_measurement_test imports of prefill_profile_test._setup and _formal_bytes, using re-exports from shared test support if needed; keep that selected filename, avoid editing internal repo.
- Every baseline method must be recorded as retained/moved or intentionally removed. Independent discovery must match selection, with no duplicate IDs or retained tests silently uncollected.

## Review Focus
- Core output correctness must survive pruning: reuse checks, round timing/token metrics, partial/failure diagnostics, TPOT success/failure, and persisted evidence validation.
- SSH/gang startup bugs just fixed must remain covered, including args/env, rank topology, diagnostic directories and quoting.
- Same-named helpers and class-local fixtures must retain semantics after merging.
- Source checkout and Bazel runfiles must both discover/execute all selected tests; include moved fixtures, scripts, config data and native prerequisites.
- Report smoke should validate generation/fallback with meaningful data, without layout/control snapshots.

### Task 1: Implement the selected core suite and audit removed coverage

**Files:** Public rtp_llm/test/perf_test/*_test.py, BUILD, test_support.py (and a few domain support modules only if helpful), docs/test_consolidation.md, this plan. Production files untouched.

**Selection:** Read `.superpowers/sdd/2026-11-06-perf-core-suite/selection.json` first. It specifies exactly 138 baseline old_id → destination new_id mappings, across these 15 files: entry_test, unified_config_test, plan_test, workload_test, prefill_runner_test, prefill_resume_test, decode_executor_test, result_envelope_test, engine_session_test, runtime_adapter_test, ssh_prefill_test, decode_multi_node_test, prefill_profile_test, prefill_report_test, offline_runner_test. Retain class/method names for audit. Selected assertions are behavior contracts, not literal implementation text; small fixture/refactor edits allowed without reducing intent.

- [x] Capture baseline discovery and AST inventory on base 5a163f51d9b28d8939c61a8b8f1a398b3cf1efc1; run baseline suite before changes.
- [x] Retain/move the selected methods, remove all non-selected methods and obsolete test modules/targets. Do not change production offline_bench_test.py. Prune unused imports and unreachable fixtures/helpers, not only tests.
- [x] Resolve global/helper collisions explicitly. Extract shared fixtures only where multiple suites need them; consolidate compatible configuration fixtures to minimal defaults with per-case overrides. Preserve internal fixture compatibility as stated. Avoid wholesale copy of old modules into support.
- [x] Trim PrefillChartTest.test_standalone_charts_and_static_fallback to report artifacts valid/readable, essential numeric data available, and valid fallback when plotly is missing. Remove static/interactive layout tests and docs/example enumeration entirely. A representative normal prefill/decode config must still run through real entry (existing selected entry/engine tests cover this).
- [x] Inspect selected tests for brittle whole-object snapshots/default-field expectations and unnecessary negative matrices. Keep only assertions that prove their named core behavior. Record any trimmed subcases/assertions separately in audit, rather than silently weakening a selected requirement.
- [x] Update BUILD to exactly the selected unit targets and their test-only dependencies/data. Remove obsolete executable references; search both repos. Do not add compatibility aliases that run entire merged suites repeatedly. No changes to production target source closures.
- [x] Write a complete scratch coverage-audit JSON mapping all 504 baseline IDs to retained destinations or explicit deletion reasons (parameter enumeration, appearance/docs, redundant layer, non-core edge/legacy branch). Update docs/test_consolidation.md with core scope, file counts, retained behavior groups, intentionally dropped coverage and validation. Do not make durable docs depend on ignored scratch files.
- [x] Run public suite with /opt/conda310/bin/python -m unittest discover -s rtp_llm/test/perf_test -p '*_test.py' -q and internal suite from parent with PYTHONPATH to this checkout. Audit actual discovery against selection, no duplicate IDs or unexplained omissions. Root will run all 15 Bazel tests; do not run Bazel yourself.
- [x] Self-review and commit only task files, including this plan. Run formatting hooks and relevant verification on final tree. Report full commands/output summaries, source+fixture LOC before/after, actual collection and semantic audit, target list and any concerns. Root owns reviews and final branch alignment.
