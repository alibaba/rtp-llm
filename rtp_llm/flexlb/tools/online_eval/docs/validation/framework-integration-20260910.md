# Case framework integration — 2026-09-10

The delivery branch is `codex/ft-case-framework`, checked out at
`/Users/wangziyi/code/rtp-llm-case-framework`. Other directories are working or
validation snapshots, not separate delivery branches.

## Source accounting

| Source | Disposition in the integrated tree |
| --- | --- |
| `6e7da12566` and its predecessors `5d2743a721`, `96dc80af71`, `ca128d5677`, `168ad89170` | Preserved through the previously squashed/rebased framework commit `3c91371e47`; includes YAML-only configuration, mock P→D stream failure propagation, design documents and live probe choreography. |
| `3c91371e47` on configuration base `6b96f5e540` | Retained in full: schema 3 adaptation, retired configuration contracts, bounded elastic convergence, capacity trailers and reserved-live repair. This integration does not edit production code beyond that already-reviewed base. |
| `0dd7f05a9d` | Retained in full: real request pressure instead of runtime speed changes, obsolete park sampling removal, pinned KV evidence, diagnostic SKIP semantics. |
| 18 uncommitted files in `ft-config-20260909` | Integrated: cancellation contract and direct ABSENT_FENCE coverage, restart channel invalidation, kill accounting bounds, execution-partial evidence and wraparound execution-window checks, plus their YAML and tests. |
| Canonical leader spill program/YAML and test fixture changes | Integrated as the dependency of the filler race repair; existing saturation construction retained. Added YAML `filler_sync.seconds: 0.3` before the first filler. |
| Canonical isolation filler construction | Superseded by `0dd7f05a9d`'s already-tested implementation: YAML 1.5-second index wait before first filler, explicit holder checks and terminal drains. Retained that stronger implementation instead of restoring the older construction. |
| Canonical tombstone reference edits | Covered by the integrated exact 8431 rejection, per-engine cancellation census and 8211 post-restart contract, with separate direct absent-fence probes. |
| R2.1 priority expectation changes | YAML selects strict priority for queued rounds, priority comparator and observability. Legacy default remains for compatibility; FIFO unchanged. Observability expects 90 then 70a, and reads `[request-scheduler]`. |
| `error_code_family` from the R2.1 document | Already retired during schema 3 adaptation. Not resurrected with obsolete production configuration. |
| `/tmp/rtp-yaml-case-preserve` | 61 files match `ca128d5677` exactly; remaining leader-spill files are superseded by the newer canonical changes. |
| Canonical `elastic_lifecycle.py` program/action and `scenarios/elastic/lifecycle.yaml` temporary relaxation | Not a validated fix: lowers removal-window success thresholds, including to zero, and tolerates residual accounting. Preserved separately in the original worktree and evidence backup; deliberately excluded from the integrated commit. |

The older `rtp-case-fixes-20260909`, `rtp-case-config-rebase`,
`rtp-mock-pd-link-break`, `rtp-priority-live-design` and
`rtp-yaml-case-configuration` worktrees have no additional tracked modifications.
Historical investigation branches outside this case-framework task are not
silently merged into the delivery branch.

## Verification and evidence

The integrated inventory contains 394 instances. Source accounting and backups:
`/Users/wangziyi/code/case-refactor-reports/2026-09-10/r21-filler/`.
The new full matrix uses an immutable export of `0dd7f05a9d` plus the integrated
Python/YAML/test files, with SHA-256 hashes in `SOURCE.json`:
`/Users/wangziyi/code/case-refactor-reports/2026-09-10/full394-integration/`.

The previous full matrix remains separately available at `full394-contracts`;
its results are not replaced by this run. Passing framework unit tests do not
imply that all scheduler behavior probes pass.

Framework verification: **863 Python tests passed** in 468.87 seconds. Three
remote Maven builds succeeded. All 745 runtime source files match the immutable
export on each of the three hosts (documentation is excluded by MCP sync).
Full matrix results are recorded in the report directory above.
