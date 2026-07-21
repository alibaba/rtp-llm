# FlashInfer FMHA v2 Workspace Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase the FlashInfer TRT-LLM FMHA v2 workspace to 128 MiB on `features/amd_qwen`, publish an sm9x image from that commit, and clone `FT_TEST:5713` with the new image and an auditable description.

**Architecture:** Keep the runtime change deliberately minimal: retain the existing pooled workspace lifecycle and change only its fixed capacity and explanatory comment. Build the internal sm9x production image from the unchanged internal `features/amd_qwen` branch plus the updated GitHub `features/amd_qwen` branch, then clone the exact Whale template version while replacing only its two sm9x image references.

**Tech Stack:** Python, PyTorch CUDA workspace allocation, Git, Aone `a1` CI, Whale CLI, `jq`.

---

### Task 1: Prove the current workspace is undersized

**Files:**
- Test: `rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py`

- [ ] **Step 1: Confirm branch isolation and a clean open-source worktree**

Run:

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-list --left-right --count HEAD...origin/features/amd_qwen
```

Expected: branch is `features/amd_qwen`, ahead/behind is `0 0`, and there are no pre-existing source changes.

- [ ] **Step 2: Run the desired-value assertion before implementation**

Run:

```bash
python3 -c 'from pathlib import Path; source = Path("rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py").read_text(); assert "_TRTLLM_FMHA_V2_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024" in source'
```

Expected: FAIL with `AssertionError`, because the source still uses 1024 bytes.

### Task 2: Apply and verify the minimal 128 MiB fix

**Files:**
- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py:28-32`

- [ ] **Step 1: Update the workspace constant and comment**

Replace the old comment and constant with:

```python
# Match FlashInfer v0.6.9 FMHA v2 tests: runtime softmax statistics and
# conditional kernel scratch are allocated from this pooled workspace.
_TRTLLM_FMHA_V2_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
```

- [ ] **Step 2: Re-run the desired-value assertion**

Run the Task 1 assertion again.

Expected: exit code 0.

- [ ] **Step 3: Run lightweight source verification**

Run:

```bash
python3 -m py_compile rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py
git diff --check
git diff -- rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py
```

Expected: compilation and diff checks pass; the scoped diff shows only the comment and constant change.

### Task 3: Commit and publish only the GitHub feature branch

**Files:**
- Commit: `rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py`
- Commit: `docs/superpowers/plans/2026-07-21-flashinfer-fmha-v2-workspace-release.md`

- [ ] **Step 1: Verify branch alignment immediately before commit**

Run:

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
```

Expected: branch remains `features/amd_qwen`, with only the two task files changed.

- [ ] **Step 2: Stage only the two task files and commit**

Run:

```bash
git add rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py docs/superpowers/plans/2026-07-21-flashinfer-fmha-v2-workspace-release.md
git commit -m "fix: increase FlashInfer FMHA v2 workspace"
```

Expected: one conventional commit containing only the approved fix and its plan.

- [ ] **Step 3: Rebase, re-verify, push, and verify branch isolation**

Run:

```bash
git fetch origin main features/amd_qwen
git rebase origin/main
python3 -c 'from pathlib import Path; source = Path("rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py").read_text(); assert "_TRTLLM_FMHA_V2_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024" in source'
python3 -m py_compile rtp_llm/models_py/modules/factory/attention/cuda_impl/trt.py
git diff --check HEAD~1..HEAD
git push origin HEAD:features/amd_qwen
git status --short --branch
git branch --contains HEAD
```

Expected: rebase and verification succeed, local and remote `features/amd_qwen` match, and no main branch is modified.

### Task 4: Build the internal sm9x image

**Files:**
- No repository file changes.

- [ ] **Step 1: Trigger only the sm9x pre-job**

Use pipeline `18410` with:

```text
branch=features/amd_qwen
github_commit=origin/features/amd_qwen
github_source_repo=git@github.com:alibaba/rtp-llm.git
wheel-tag=daily
pre-jobs=[sm9x]
```

Expected: the API returns a pipeline run ID.

- [ ] **Step 2: Poll the run and image-sm9x job to a terminal state**

Run `a1 ci run get` and `a1 ci job list` at bounded intervals.

Expected: `image-sm9x` reaches `SUCCESS`. DADI or regional synchronization failures do not invalidate a successful core image job.

- [ ] **Step 3: Extract and verify the exact image path**

Expected image namespace:

```text
hub.docker.alibaba-inc.com/isearch/rtp_llm_sm9x:<new-tag>
```

The tag must contain the new GitHub commit hash and be newer than the image currently stored in `FT_TEST:5713`.

### Task 5: Clone FT_TEST:5713 with the built image

**Files:**
- No repository file changes; Whale template version source ID is `6a5f236fd9c1fa39075bf9c6`.

- [ ] **Step 1: Generate the clone payload and replace only sm9x references**

Starting from `whale template version clone 6a5f236fd9c1fa39075bf9c6 --dry-run -o json`, use `jq` to update:

```text
.content.image_infos[] | select(.card_type == "NV" and .sm_version == "sm9x") | .image_path
.content.image_packages.nv_image
.description
```

The description must include the base template/version ID, internal and GitHub commits, pipeline/run/job IDs, the 128 MiB workspace change, build date, and exact sm9x image.

- [ ] **Step 2: Submit the modified payload to Whale dry-run**

Run the generated payload through:

```bash
whale template version clone 6a5f236fd9c1fa39075bf9c6 -f - --dry-run -o json
```

Expected: exactly two image string changes plus the description; all other template content remains inherited from version 5713.

- [ ] **Step 3: Create the new template version**

Submit the same reviewed payload without `--dry-run`.

Expected: Whale returns success with a new `FT_TEST` version ID and version number.

- [ ] **Step 4: Read back and verify the new version**

Run `whale template version get <new-version-id> -o json` and assert both sm9x references equal the built image, the description contains the release evidence, and non-sm9x image references match version 5713.

### Task 6: Final state verification

**Files:**
- No additional changes.

- [ ] **Step 1: Re-run code and Git verification**

Run the desired-value assertion, `py_compile`, `git diff --check`, `git status --short --branch`, and `git log -1 --oneline` in the open-source worktree.

- [ ] **Step 2: Report durable identifiers**

Report the code commit, image pipeline/run/job, exact sm9x image, cloned Whale template version/ID, and the unresolved pre-existing merge state in `ai-search-workspace`.
