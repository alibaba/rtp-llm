# Perf Test Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Consolidate the public unified perf tests from 60 unit-test files to approximately 50 while preserving meaningful behavioral coverage.

**Architecture:** Merge small suites into related existing modules, consolidate docs/example validation, and extract the shared report fixture into a test-only support module. Keep production code unchanged and audit old-to-new test identities; retain distinct regression boundaries instead of reducing counts through non-discovery.

**Tech Stack:** Python unittest, Bazel py_test/py_library.

**Spec:** User-approved approach in this conversation: merge small files, remove explicitly redundant low-value tests, preserve SSH/build-boundary/metrics/failure/resume/compatibility regressions, record where every moved or removed test goes.

## Global Constraints
- Production perf behavior and production source files remain unchanged.
- Preserve all assertions except the two enumerated test-local wrapper cases and genuinely duplicated docs command parsing; retain real entry rejection/side-effect checks and the complete capability matrix.
- Keep current cache_grid_main worktree; do not touch user untracked files, including docs/architecture.md, or the parent gitlink. No push, live model/GPU/SSH/Aone workload, or native build dependency refactor.
- Update Bazel srcs/data/deps to match moved suites, including transitive example files and extracted helpers. Test support must not enter production libraries.
- Every original test class/method must map to a collected test or one documented removal; prevent duplicate discovery from imported TestCase classes.

## Review Focus
- Helpers with matching names must not silently change the merged suites' fixtures.
- Bazel runfiles must contain the union of required data and support sources after merging.
- Direct imports of moved test modules in public/internal tests must be updated or avoided.
- Docs/examples must preserve distinct planning, action coverage, and preview assertions while sharing command parsing.
- Existing SSH startup and runtime source-closure tests stay intact.

### Task 1: Consolidate public test modules and verify coverage correspondence

**Files:** rtp_llm/test/perf_test/BUILD; affected *_test.py below; new test_support.py; new docs/test_consolidation.md. Do not edit production modules.

**Interfaces:** Keep existing TestCase class and test method names when moving tests. Extract prefill_report_test._setup into test_support.setup_prefill_report_run, consumed by report tests and prefill_offline_integration_test. No need to migrate unrelated helper families this round.

- [x] Capture baseline discovered test IDs and source method inventory, then run the public suite. Exclude offline_bench_test.py from unit-file count (it is a manual microbenchmark).
- [x] Merge modules using this mapping (retain original classes, rename colliding private helpers):
  - case_planning_test.py and external_plan_policy_test.py -> plan_test.py
  - entry_execution_test.py -> entry_test.py
  - execution_scheduler_test.py -> runtime_adapter_test.py
  - engine_paths_test.py -> engine_session_test.py
  - prefill_chart_test.py -> prefill_report_test.py
  - prefill_examples_test.py and decode_examples_test.py -> unified_docs_test.py
  - prefill_finalize_test.py -> prefill_adapter_test.py
  - atomic_storage_test.py -> run_directory_test.py
- [x] Delete only CapabilitiesTest.test_unsupported_combination_fails_before_all_side_effects and DecodeExamplesTest.test_decode_follow_up_actions_and_aone_fail_before_side_effects, because they assert sequencing of test-local wrapper functions. Keep complete capability-matrix tests and public_boundary_test real-entry tests; ensure unsupported decode follow-up side-effect behavior is covered by a real-entry assertion if existing entry coverage does not do so. Any extra case needed for this is meaningful, not an artificial replacement.
- [x] Consolidate docs command extraction/parse traversal and reuse results for mode-specific assertions. Preserve distinct examples' planned case counts, traffic fingerprints, supported actions, preview counts, and generated-guide consistency. Do not remove tests merely because they use similar fixtures.
- [x] Extract report setup to test_support.py and remove direct report-test imports from its consumers. Declare a testonly py_library for support and use it from affected Bazel tests; no production dependency on test support.
- [x] Update affected Bazel targets: remove obsolete targets, union necessary runfiles/deps in destinations; search both repos for references to removed test names. Do not retain duplicate test targets that execute the same merged suite.
- [x] Produce docs/test_consolidation.md with before/after counts, file mapping, exact removed IDs and replacement coverage. Produce a complete baseline-ID-to-new-ID mapping JSON in the task workspace for review, with an explicit reason for removed cases. Prove no unexplained loss or duplicate discovered IDs. Keeping class/method names should make mechanical correspondence simple.
- [x] Run public suite using /opt/conda310/bin/python -m unittest discover -s rtp_llm/test/perf_test -p '*_test.py' -q. Run internal suite from parent with PYTHONPATH pointing at this checkout. Root will execute changed Bazel targets after reviewing the resulting declarations; do not run Bazel yourself.
- [x] Self-review, ensure git diff --check passes, and commit only changed tests/BUILD/test support/mapping documentation and this plan. Record exact commands, outcomes, count changes, mapping audit and changed Bazel targets in report. Root performs independent review and final delivery ref synchronization.
