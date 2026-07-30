---
name: rtp-remote-dev-environment
description: "Design, build, and operate the RTP-LLM remote development environment in which a local Mac is the code and orchestration control plane, the b300 host is the containerized compilation hub, and built wheels are validated locally or distributed to compatible remote GPU nodes. Use when discussing or implementing Mac-to-b300 source synchronization, remote container builds, reusable build caches, wheel packaging, build-once/test-many workflows, multi-machine GPU validation, test result collection, or the overall remote development environment. Triggers include: 远程开发环境, Mac 开发中枢, b300 编译中枢, 同步代码到 115, 远程容器编译, 分发 wheel, 多机测试, build once test many."
---

# RTP Remote Development Environment

## Objective

Build a reproducible workflow with three planes:

- Use the local Mac as the control plane and the only source of truth for code and Git state.
- Use `b300` as the build plane and compile RTP-LLM only inside a controlled container.
- Use `b300` and other compatible GPU nodes as the test plane, distributing wheels instead of rebuilding source on every node.

Keep remote workspaces disposable and rebuildable. Return artifacts, logs, environment metadata, and test results to the Mac; do not allow untracked remote source changes to become authoritative.

## Known Topology

- Local repository: `/Users/huidong/repo/rtp-llm`
- Build host SSH alias: `b300`
- Build host address from local SSH configuration: `11.163.39.115`
- Remote repository: `/data1/zhanghuidong.zhd/workspace/RTP-LLM`
- Remote build environment: container only
- Default source branch for the current K3 workflow: `feat/k3_decode_opt`
- Clean b300 source worktree: use a dedicated task worktree; never repurpose a
  dirty worktree by resetting or overwriting user changes.
- Test-node inventory and observed connectivity: read [references/test-environments.md](references/test-environments.md) when selecting or diagnosing a remote node.
- Concrete cache layout, remote task isolation layout, and test matrix: not yet finalized

Verify topology before using it. Treat values that may change as discovered configuration, not permanent facts.

## Operating Invariants

1. Keep editing, Git branches, worktrees, commits, review, and task orchestration on the Mac.
2. Never compile RTP-LLM on the Mac.
3. Compile on `b300` only inside the selected container image.
4. Synchronize committed source Mac -> `origin` -> b300. Treat patch transfer as
   an explicit fallback only when Git transport is unavailable.
5. Reuse dependency and compiler caches without sharing mutable source or output directories between concurrent tasks.
6. Package build results as traceable wheels and test bundles.
7. Test directly on `b300` when it is the target platform. For other machines, transfer the wheel and test payload into the same validated container environment.
8. Collect logs and structured results back at the control plane.
9. Preview destructive synchronization or cleanup operations before execution. Never delete a broad remote workspace based on an unresolved path.
10. Never store sudo passwords, tokens, or other credentials in this skill, the repository, command lines, logs, or build manifests.
11. After every source operation, require the Mac `HEAD`, Mac
    `origin/<branch>`, origin branch, and clean b300 worktree `HEAD` to have the
    same commit and tree IDs. Require no tracked Mac changes; record but do not
    automatically add or delete unrelated untracked local orchestration files.
12. Keep operation records under `.tmp/rtp-remote-dev-operations/<operation-id>/`.
    Redact credentials and record commands, before/after state, verification,
    and test results.

## Workflow

### 1. Inspect

- Inspect the local branch, commit, worktree state, submodules, and uncommitted diff.
- Verify SSH access to `b300`, available disk space, GPU visibility, container runtime access, and the selected image.
- If Docker reports permission denied on `/var/run/docker.sock`, use the `remote-docker-access` skill. Diagnose `id`, socket ownership, and `docker ps` before requesting any change.
- Discover the target test nodes and required GPU architectures.
- Distinguish a design request from an execution request. Keep design and diagnosis read-only unless the user asks for implementation or operation.

### 2. Synchronize

- Use the Mac checkout as the only writable source checkout.
- Commit locally and push to `origin` before b300 consumes a change.
- Assign a clean b300 worktree per branch or task. If an existing worktree is
  dirty, preserve it and create another worktree; do not reset, checkout over,
  stash, or clean changes of unknown ownership.
- For ordinary updates, fetch and fast-forward only. Stop if fast-forward is
  impossible; do not create an integration merge on b300.
- Keep b300 detached at `origin/<branch>` when the worktree is test-only. This
  makes its exact input identity explicit and avoids accidental remote commits.
- Reuse Bazel and dependency caches outside the source worktree. Never clean a
  shared cache to solve synchronization problems.
- After synchronization, compare commit IDs and tree IDs across Mac, origin,
  and b300. Require a tracked-clean Mac checkout and a completely clean b300
  test worktree before build or test.
- Use `scripts/capture_sync_state.py` before and after every Git operation to
  write structured state into the operation ledger.

### 2.1 Compact Feature History

Rewrite feature history only when the user explicitly requests it and the
target is not a protected/shared main branch.

1. Fetch origin and record the exact origin branch SHA.
2. Create a timestamped backup branch and a Git bundle before rewriting.
3. If the user names an anchor commit and requests all later work folded into
   it, reset the index softly to the anchor's parent and recommit the complete
   tree with the anchor message. Never use a hard reset.
4. Verify the compacted `HEAD` is byte-identical to the backup with both tree
   ID equality and an empty `git diff <backup>..HEAD`.
5. Push with an explicit lease containing the recorded old origin SHA:

   ```bash
   git push \
     --force-with-lease=refs/heads/<branch>:<old-origin-sha> \
     origin HEAD:refs/heads/<branch>
   ```

6. Fetch again and require local `HEAD`, `origin/<branch>`, and `ls-remote`
   results to match.
7. Create or refresh a clean b300 test worktree at the rewritten origin commit.
   Preserve any older dirty worktree as a separate WIP workspace.
8. Re-run the relevant tests because commit identity and test workspace changed,
   even when tree equivalence has already been proven.

Do not force-push when the explicit lease fails. Reinspect the remote history
and ask the user before incorporating another contributor's update.

### 2.2 Operation Ledger

Use an operation ID such as `YYYYMMDD_HHMMSS_<action>`. Store at least:

- `before.json` and `after.json` from `scripts/capture_sync_state.py`.
- `commands.log` with credential-free commands and exit codes.
- `tests.json` with target, configuration, result, signal, and log path.
- `backup.bundle` and backup ref identity for history rewrites.
- A final summary stating whether commit IDs, tree IDs, and worktree cleanliness
  satisfy the synchronization contract.

Never put sudo input, tokens, credential-bearing URLs, or remote-cache headers
in the ledger.

### 3. Build

- Mount the synchronized task workspace and persistent caches into the build container.
- Build and package the wheel inside the container.
- Capture the image reference or digest, Python version, PyTorch version, CUDA toolkit, compiler, Bazel configuration, GPU architecture list, commit, dirty-diff digest, and build command.
- Produce a wheel, checksum, build manifest, and build log as one artifact set.
- Fail clearly when compilation succeeds but packaging or artifact metadata generation fails.

### 4. Route Tests

- For `b300`, install the wheel into a clean test environment in the validated container and run the requested tests directly.
- For another node, transfer only the wheel, checksum, manifest, and required test/model payloads. Avoid transferring the full source tree unless a source-based test explicitly requires it.
- Start the same container image or a proven ABI-compatible image, install the wheel, run environment probes, and then run tests.
- Separate correctness, smoke, performance, and stress-test results. Do not compare performance numbers collected under materially different environments without labeling the difference.

### 5. Report

- Return a concise status summary to the Mac with artifact identity, build status, per-node test status, failed commands, log locations, and environment differences.
- Keep enough metadata to rerun any failed test from the same wheel.
- Do not report success from process exit status alone; verify expected artifacts and test assertions.

## Compatibility Gate

Do not assume that using the same container makes a wheel portable. Before distributing a wheel, verify:

- The wheel contains code for the target GPU compute capability, or uses runtime compilation intentionally.
- Python, PyTorch, CUDA user-space libraries, C++ ABI, and required operator dependencies match.
- The target NVIDIA driver supports the selected CUDA runtime.
- CPU architecture matches, especially for x86_64 versus ARM64 hosts.
- Any host-mounted libraries, model paths, caches, and network dependencies are declared.

Build separate wheels or a deliberate multi-architecture wheel when the compatibility gate cannot prove portability.

## Coordinate Existing Skills

Use specialized project skills instead of duplicating their detailed procedures:

- Use `precheck-commit-push` for local submission and Git workflow.
- Use `dev-container` for RTP-LLM development container creation and management.
- Use `test-execution` for Bazel build and test conventions.
- Use `remote-wheel-test` for isolated wheel transfer and remote validation.
- Use `rtp-worktree-test` for isolated local and remote worktree testing patterns.
- Use `remote-docker-access` when the SSH user cannot access the remote Docker socket. Supply sudo credentials interactively or through an approved secret store; never persist or print them.
- Use architecture- or test-specific skills for kernel benchmarks, performance reports, smoke diagnosis, or multi-node tests.

Keep this skill responsible for end-to-end orchestration, environment boundaries, artifact traceability, and routing decisions.

## Evolve Safely

Treat this as an evolving operational contract. When a workflow is validated in real use:

1. Record stable configuration and decision rules here.
2. Add deterministic repeated operations under `scripts/`.
3. Put detailed host, image, compatibility, and artifact schemas under `references/`.
4. Avoid recording credentials, transient tokens, or machine-specific secrets.
5. Revalidate the skill after every structural change.
