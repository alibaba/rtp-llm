# Topology KV Candidate Schedule Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate PR #1160 as a screening benchmark, then define the next production gate for runtime sparse MLA/indexer integration and model-quality validation.

**Architecture:** Keep the current benchmark isolated under `benchmark/` so it does not affect RTP-LLM serving paths. Use WSL CUDA validation as the near-term end-to-end signal, then move to runtime integration only after reviewers accept the screening benchmark.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, pytest, flake8, WSL2, NVIDIA GeForce RTX 4060 Laptop GPU.

## Global Constraints

- Do not change RTP-LLM production attention, KV cache, sparse MLA, or indexer behavior in PR #1160.
- Treat the benchmark as a screening signal, not as a serving speedup claim.
- Keep CUDA speed validation opt-in and skip the speed gate when CUDA is unavailable.
- Link the PR to a tracking issue for runtime integration and model-quality validation.

---

### Task 1: Reproduce the WSL CUDA Screening Benchmark

**Files:**
- Modify: `benchmark/README.md`
- Test: `benchmark/test_topology_kv_candidate_schedule.py`

**Interfaces:**
- Consumes: `benchmark_decode_attention(seq_len, selected_tokens, heads, head_dim, rounds, warmup, dtype, device)`
- Produces: Markdown benchmark table for `seq_len=16384`, `selected_tokens=128 256 512 1024`, `rounds=60`, `warmup=20`

- [ ] **Step 1: Create the WSL benchmark environment**

```bash
python3 -m venv /tmp/rtp-llm-bench-venv
source /tmp/rtp-llm-bench-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install pytest flake8
```

- [ ] **Step 2: Verify CUDA is visible from WSL**

Run:

```bash
source /tmp/rtp-llm-bench-venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Expected output includes:

```text
2.5.1+cu121
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

- [ ] **Step 3: Run the benchmark tests**

Run:

```bash
cd /mnt/c/Users/seal/Documents/GitHub/rtp-llm
source /tmp/rtp-llm-bench-venv/bin/activate
time python -m pytest benchmark/test_topology_kv_candidate_schedule.py -q
```

Expected output:

```text
8 passed
```

- [ ] **Step 4: Run compile and style checks**

Run:

```bash
cd /mnt/c/Users/seal/Documents/GitHub/rtp-llm
source /tmp/rtp-llm-bench-venv/bin/activate
time python -m py_compile benchmark/topology_kv_candidate_schedule.py benchmark/test_topology_kv_candidate_schedule.py
time python -m flake8 benchmark/topology_kv_candidate_schedule.py benchmark/test_topology_kv_candidate_schedule.py
```

Expected output: both commands exit with status 0.

- [ ] **Step 5: Run the real-time CUDA benchmark**

Run:

```bash
cd /mnt/c/Users/seal/Documents/GitHub/rtp-llm
source /tmp/rtp-llm-bench-venv/bin/activate
time python benchmark/topology_kv_candidate_schedule.py \
  --seq-len 16384 \
  --selected-tokens 128 256 512 1024 \
  --heads 16 \
  --head-dim 64 \
  --rounds 60 \
  --warmup 20 \
  --device cuda
```

Expected output from the current WSL run:

```text
| seq_len | selected_tokens | dense_sdpa_ms | sparse_selected_ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 128 | 0.2867 | 0.1166 | 2.46x |
| 16384 | 256 | 0.2857 | 0.1106 | 2.58x |
| 16384 | 512 | 0.2853 | 0.1165 | 2.45x |
| 16384 | 1024 | 0.2912 | 0.1346 | 2.16x |
```

### Task 2: Publish the Draft PR and Tracking Issue

**Files:**
- Modify: `benchmark/README.md`
- Create: `docs/superpowers/plans/2026-07-07-topology-kv-candidate-schedule-validation.md`

**Interfaces:**
- Consumes: WSL validation output from Task 1
- Produces: GitHub issue linked from draft PR #1160

- [ ] **Step 1: Commit the validation documentation**

Run:

```bash
git status --short
git add benchmark/README.md benchmark/test_topology_kv_candidate_schedule.py docs/superpowers/plans/2026-07-07-topology-kv-candidate-schedule-validation.md
git commit -m "bench: document topology kv wsl validation"
```

Expected output: a new commit on `codex/topology-kv-candidate-schedule`.

- [ ] **Step 2: Push the branch**

Run:

```bash
git push -u origin codex/topology-kv-candidate-schedule
```

Expected output: branch is up to date on `teerthsharma/rtp-llm`.

- [ ] **Step 3: Create the tracking issue**

Run:

```bash
gh issue create \
  --repo alibaba/rtp-llm \
  --title "Track production validation for topology KV candidate schedules" \
  --body-file /tmp/topology-kv-candidate-schedule-issue.md
```

Expected output: a new issue URL in `alibaba/rtp-llm`.

- [ ] **Step 4: Update draft PR #1160**

Run:

```bash
gh pr edit 1160 \
  --repo alibaba/rtp-llm \
  --body-file /tmp/topology-kv-candidate-schedule-pr.md
```

Expected output: PR #1160 body references the tracking issue and the WSL validation output.

### Task 3: Production Gate After Reviewer Acceptance

**Files:**
- Modify: `rtp_llm/models_py/modules/hybrid/mla_attention.py`
- Modify: `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py`
- Test: `rtp_llm/models_py/modules/hybrid/test/indexer_test.py`
- Test: `rtp_llm/models_py/modules/hybrid/test/mla_reuse_cache_test.py`
- Test: `rtp_llm/test/perf_test/batch_decode_test.py`
- Test: `benchmark/benchmark_serving.py`
- Test: `rtp_llm/test/smoke/tau2_bench_comparer.py`

**Interfaces:**
- Consumes: Candidate rows from `build_block_candidate_schedule(key_block_centroids, config)`
- Produces: Runtime sparse attention path that can consume topology-derived candidate rows without changing default serving behavior

- [ ] **Step 1: Add runtime ingestion behind an opt-in flag**

Implementation requirement:

```text
Default serving must remain dense/current behavior. The topology schedule path must be activated only by an explicit sparse-attention experiment flag or runtime config that is disabled by default.
```

- [ ] **Step 2: Add indexer-level regression coverage**

Run:

```bash
bazelisk test //rtp_llm/models_py/modules/hybrid/test:indexer_test --config=cuda12_9 --config=sm9x
```

Expected output:

```text
PASS
```

- [ ] **Step 3: Add sparse MLA reuse-cache regression coverage**

Run:

```bash
bazelisk test //rtp_llm/models_py/modules/hybrid/test:mla_reuse_cache_test --config=cuda12_9 --config=sm9x
```

Expected output:

```text
PASS
```

- [ ] **Step 4: Benchmark end-to-end RTP-LLM serving**

Run the existing single-node decode benchmark after wiring the opt-in runtime path:

```bash
bazelisk test //rtp_llm/test/perf_test:grid_perf_test \
  --config=cuda12_9 --config=sm9x \
  --test_arg=--batch_size=1 \
  --test_arg=--input_len=16384 \
  --test_arg=--partial=1
```

Expected decision threshold:

```text
Proceed only if the opt-in runtime path shows a useful latency or memory tradeoff against the current baseline on representative hardware.
```

- [ ] **Step 5: Run model-quality validation**

Run the existing tau2 smoke comparer through the RTP-LLM smoke harness after enabling the opt-in runtime path for the smoke server:

```bash
python rtp_llm/test/smoke/tau2_bench_comparer.py
```

Expected decision threshold:

```text
Proceed only if tau2-bench OVERALL score stays at or above the comparer threshold of 0.76.
```
