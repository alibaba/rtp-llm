---
name: rtp-vllm-precision-bisect
description: RTP-LLM vs vLLM precision/output alignment workflow for DeepSeek-V4 and similar inference regressions. Use when debugging RTP-vLLM token divergence, tensor/logit mismatch, teacher-forced decode comparisons, stable oracle setup, FP8 KV/cache/block-id precision issues, or when deciding which precision fixes are production changes versus debug instrumentation.
---

# RTP-vLLM Precision Bisect

## Purpose

Use this skill to turn an RTP-vLLM mismatch into an explainable, reproducible result. The required standard is not "close enough": either outputs match under a stable oracle, or the first divergence is localized to a specific tensor/operator/config difference with evidence.

## Ground Rules

- Start from a stable oracle. Do not bisect two unstable systems.
- Keep production fixes staged separately from debug instrumentation.
- Use teacher forcing only to localize a specific decode step. The final proof must be RTP self-roll natural generation without per-step vLLM inputs.
- Compare exact token ids before interpreting text.
- Record every run path, flags, first-diff index, and confirmed/excluded hypothesis in the active project memory/debug document.
- If a user reports a token such as "RTP has 303 but vLLM has 478", verify absolute generated indices before accepting it as a divergence.

## Stable Baseline

Use vLLM as an oracle only after making it deterministic enough for the target case:

```bash
export VLLM_DSV4_TORCH_TOPK=1
export VLLM_DSV4_DISABLE_AUX_STREAMS=1
# start vLLM with:
#   --enforce-eager --no-enable-prefix-caching
```

For RTP precision comparison, enable deterministic/stable paths appropriate to the investigation:

```bash
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON
export DSV4_TORCH_TOPK=1
export DSV4_INDEXER_TOPK_BACKEND=torch
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_GATE_FP32=1
```

These are precision-debug switches. Keep production defaults off unless the feature is independently approved for normal serving. Precision-debug code paths should be opt-in by environment variable or CLI flag, and final production fixes must not depend on debug-only defaults.

Do not enable these by default unless the investigation specifically needs them:

```bash
export DSV4_TORCH_ATTN=1                    # has caused service load/exit in prior DSV4 tests
export DSV4_MHC_PRE_NUM_SPLIT=<global>      # global override can change long-prefill behavior
```

For PD repros, preserve the intended production topology:

- Prefill: required CP/TP/EP settings from the user; FP8 KV cache if the issue is FP8-KV related.
- Decode: required DP setting; FP8 KV cache; CUDA graph on if the production issue happens with graph on.
- Disable unrelated prompt/prefix cache only when it creates a path mismatch for the experiment.

## Deployment Recipes

### vLLM Stable Oracle

Use vLLM only after disabling known unstable/path-changing behavior. Adapt model paths, GPU id, port, and any local patched vLLM environment to the case:

```bash
tmux new-session -d -s dsv4_precision_vllm_stable '
export CUDA_VISIBLE_DEVICES=<vllm_gpu>
export VLLM_DSV4_TORCH_TOPK=1
export VLLM_DSV4_DISABLE_AUX_STREAMS=1
export VLLM_USE_V1=1
<vllm-python> -m vllm.entrypoints.openai.api_server \
  --model <model_path_or_name> \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port <vllm_port> \
  --max-model-len <max_len> \
  --enforce-eager \
  --no-enable-prefix-caching
'
```

vLLM oracle stabilization used in prior DSV4 precision work consisted of:

- `VLLM_DSV4_TORCH_TOPK=1`: local vLLM debug patch that replaces DSV4 sparse indexer custom top-k kernels with a slow `torch.topk(..., sorted=True)` reference path for both prefill and decode. This is for oracle stability, not production performance.
- `VLLM_DSV4_DISABLE_AUX_STREAMS=1`: local vLLM debug patch that runs DSV4 attention auxiliary work sequentially instead of on overlapping aux streams, reducing schedule-sensitive nondeterminism during comparison.
- `--no-enable-prefix-caching`: CLI setting, not a code patch. Prevents first request and later requests from taking different prefix-cache paths.
- `--enforce-eager`: CLI setting, not a code patch. Avoids graph/capture path differences while constructing the oracle.

Then prove oracle repeat stability before comparing RTP:

```bash
python /path/to/repeat_stability_from_q_ignore_eos.py \
  --q-path /data3/q \
  --backend vllm \
  --record-index <N> \
  --repeats 2 \
  --max-new-tokens <LEN> \
  --stop-repeat-run 10000 \
  --top-k 1 \
  --vllm-url http://127.0.0.1:<vllm_port> \
  --timeout 3600 \
  --out-root <run_root>/outputs \
  --name vllm_stable_oracle_<case>
```

### RTP PD Prefill

For prefill/decode disaggregation, start prefill separately and preserve the topology under test. Example for CP=2 with FP8 KV enabled:

```bash
tmux new-session -d -s dsv4_precision_rtp_pd_prefill '
cd <rtp_worktree>
export CUDA_VISIBLE_DEVICES=<prefill_gpu0>,<prefill_gpu1>
export START_PORT=<prefill_port>
export ROLE_TYPE=PREFILL
export LOAD_PYTHON_MODEL=1
export LOAD_METHOD=fastsafetensors
export MODEL_TYPE=deepseek_v4
export ACT_TYPE=BF16
export FP8_KV_CACHE=1
export ENABLE_CUDA_GRAPH=0
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON
export ENABLE_COMM_OVERLAP=0
export DSV4_TORCH_TOPK=1
export DSV4_INDEXER_TOPK_BACKEND=torch
export DSV4_INDEXER_TOPK_BACKEND_OVERRIDE=torch
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_GATE_FP32=1
/opt/conda310/bin/python -m rtp_llm.start_server \
  --model_type deepseek_v4 \
  --checkpoint_path <model_path> \
  --tokenizer_path <model_path> \
  --load_method fastsafetensors \
  --act_type BF16 \
  --tp_size 2 --ep_size 2 --world_size 2 \
  --role_type PREFILL \
  --fp8_kv_cache 1 \
  --cp_rotate_method ALL_GATHER \
  --use_local 1 \
  --reuse_cache 0 \
  --enable_device_cache 1 \
  --concurrency_limit 1 \
  --frontend_server_count 1
'
```

### RTP PD Decode

For decode, keep DP and CUDA graph settings aligned with the production issue. Example for DP=1, FP8 KV, CUDA graph on:

```bash
tmux new-session -d -s dsv4_precision_rtp_pd_decode '
cd <rtp_worktree>
unset RTP_TEACHER_FORCE_TOKENS RTP_TEACHER_FORCE_OFFSET
export CUDA_VISIBLE_DEVICES=<decode_gpu>
export START_PORT=<decode_port>
export ROLE_TYPE=DECODE
export LOAD_PYTHON_MODEL=1
export LOAD_METHOD=fastsafetensors
export MODEL_TYPE=deepseek_v4
export ACT_TYPE=BF16
export FP8_KV_CACHE=1
export ENABLE_CUDA_GRAPH=1
export ENABLE_CUDA_GRAPH_OVERRIDE=1
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON
export ENABLE_COMM_OVERLAP=0
export DSV4_TORCH_TOPK=1
export DSV4_INDEXER_TOPK_BACKEND=torch
export DSV4_INDEXER_TOPK_BACKEND_OVERRIDE=torch
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_GATE_FP32=1
export DSV4_TORCH_ATTN=0
export MODEL_SERVICE_CONFIG='\''{"service_id":"dsv4-precision","role_endpoints":[{"group":"default","prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:<prefill_port>","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:<decode_port>","protocol":"http","path":"/"}}],"use_local":true}'\''
/opt/conda310/bin/python -m rtp_llm.start_server \
  --model_type deepseek_v4 \
  --checkpoint_path <model_path> \
  --tokenizer_path <model_path> \
  --load_method fastsafetensors \
  --act_type BF16 \
  --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 \
  --role_type DECODE \
  --enable_cuda_graph 1 \
  --fp8_kv_cache 1 \
  --cp_rotate_method PREFILL_CP \
  --use_local 1 \
  --reuse_cache 0 \
  --enable_device_cache 1 \
  --concurrency_limit 1 \
  --frontend_server_count 1
'
```

`ENABLE_CUDA_GRAPH_OVERRIDE` is not read by RTP runtime directly. It is a launcher-script override used by some legacy scripts to set `ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH_OVERRIDE:-1}"`. The runtime-effective setting is `ENABLE_CUDA_GRAPH` or `--enable_cuda_graph`.

### Health and Log Checks

Before running comparisons:

```bash
for p in <prefill_port> <decode_port> <vllm_port>; do
  curl -sS -o /dev/null -w "health $p %{http_code}\n" \
    --connect-timeout 0.5 --max-time 2 http://127.0.0.1:$p/health || true
done
```

Check launch logs for the effective flags:

- RTP decode log must show `enable_cuda_graph: 1` when CUDA graph is part of the repro.
- vLLM log must show prefix caching disabled when `--no-enable-prefix-caching` is intended.
- Teacher forcing env vars must be unset for final self-roll gates.

## Workflow

### CLI Helper

Use the bundled CLI for repeatable precision gates after RTP/vLLM services are deployed:

```bash
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py --help
```

Common commands:

```bash
# Check service health.
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py health \
  http://127.0.0.1:<prefill_port> \
  http://127.0.0.1:<decode_port> \
  http://127.0.0.1:<vllm_port>

# Generate start scripts for vLLM oracle + RTP PD prefill/decode + the
# known-good precision gate. Review the generated scripts, then run
# start_all_tmux.sh and run_known_good_gate.sh.
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py \
  write-launch-scripts \
  --output-dir /tmp/dsv4_precision_gate_scripts \
  --rtp-worktree <rtp_worktree> \
  --model-path /data3/DeepSeekV4-Flash \
  --vllm-python /data3/vllm-dsv4-env/bin/python \
  --prefill-gpus 0,3 \
  --decode-gpu 7 \
  --vllm-gpu 4 \
  --prefill-port 18800 \
  --decode-port 18880 \
  --vllm-port 18000

# Verify the saved known-good record89 artifacts only.
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py \
  known-good-record89 --compare-only

# Run the known-good record89 RTP natural self-roll gate, then compare with
# the saved stable vLLM oracle. This assumes services are already started with
# the required switches in this document.
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py \
  known-good-record89 \
  --rtp-url http://127.0.0.1:<decode_port> \
  --prefill-url http://127.0.0.1:<prefill_port>

# Compare arbitrary generated_ids.json files.
docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py compare \
  --rtp-ids <rtp_run>/generated_ids.json \
  --vllm-ids <vllm_run>/generated_ids.json \
  --prefix-len 1000 \
  --fail-on-diff
```

The CLI prints JSON with `first_diff`, prefix hashes, longest same-token run, and tail-period repetition. It exits nonzero when `--fail-on-diff` or the known-good hash gate fails.

`write-launch-scripts` creates:

- `start_vllm_stable.sh`: starts the stable vLLM oracle.
- `start_rtp_prefill.sh`: starts RTP PD prefill with CP=2 and FP8 KV.
- `start_rtp_decode.sh`: starts RTP PD decode with DP=1, FP8 KV, CUDA graph, and precision switches.
- `start_all_tmux.sh`: starts the three services in tmux sessions.
- `run_known_good_gate.sh`: runs the RTP natural self-roll record89 gate and compares against the saved oracle.

### 1. Pin the Case

Capture these before changing code or restarting services:

- input source and record index, for example `/data3/q`, `record_index=89`
- prompt/input length
- sampling config: `top_k`, `temperature`, `top_p`, `max_new_tokens`, `ignore_eos`
- RTP/vLLM service ports, tmux session names, GPU assignments
- git branch, staged files, unstaged debug files
- exact launch flags and environment variables

### 2. Prove Oracle Stability

Run vLLM repeats first. If vLLM differs from itself, do not compare RTP against it yet. Stabilize top-k, aux streams, prefix cache, and eager/graph behavior until repeated vLLM token ids are identical for the needed length.

Prefer a saved oracle JSON such as:

```text
outputs/<vllm-stable-run>/vllm_run01/generated_ids.json
```

### 3. Run RTP Natural Self-Roll

Run RTP without teacher forcing:

```bash
unset RTP_TEACHER_FORCE_TOKENS RTP_TEACHER_FORCE_OFFSET
python /path/to/repeat_stability_from_q_ignore_eos.py \
  --q-path /data3/q \
  --backend rtp \
  --record-index <N> \
  --repeats 1 \
  --max-new-tokens <LEN> \
  --stop-repeat-run 10000 \
  --top-k 1 \
  --rtp-mode http \
  --rtp-url http://127.0.0.1:<decode_port> \
  --timeout 1800 \
  --out-root <run_root>/outputs \
  --name rtp_selfroll_<case>
```

Compare token ids offline against the saved vLLM oracle:

```python
import hashlib, json

rtp = json.load(open(".../rtp_run01/generated_ids.json"))
vllm = json.load(open(".../vllm_run01/generated_ids.json"))
first = next((i for i, (a, b) in enumerate(zip(rtp, vllm)) if a != b), None)
print("first_diff", first)
print("equal_prefix", rtp[:len(vllm)] == vllm)
print("rtp_hash", hashlib.sha256(str(rtp[:len(vllm)]).encode()).hexdigest()[:16])
print("vllm_hash", hashlib.sha256(str(vllm).encode()).hexdigest()[:16])
```

If the script records an extra RTP token, compare the common prefix and inspect the extra token separately. Do not treat length accounting as a model divergence without checking ids.

### 4. Use Teacher Forcing Only for Localization

When natural generation diverges at generated index `i`, use vLLM's token sequence as forced RTP input around that step so both systems consume the same prefix. This isolates whether the mismatch is caused by:

- earlier generated tokens changing the state, or
- same input state producing different logits/tensors.

Always label teacher-forced artifacts clearly and keep related code/debug flags unstaged unless they become production fixes.

### 5. Bisect From Output Backward

Use a narrow tensor-dump ladder. Start from the failing step and compare:

1. token ids / final logits / top-k ids and values
2. final hidden
3. target layer output
4. attention input, q/kv linear, q/kv norm, RoPE output
5. sparse indexer scores/topk/block ids/combined indices
6. selected KV/cache rows
7. attention output before projection
8. MoE/router/shared/routed outputs only after attention input is proven aligned

For each comparison, record:

- tensor name and layer
- aligned position or generated step
- shape
- `num_diff`, `max_abs_diff`, and first differing value
- whether the tensor is exact, BF16-rounding-only, or semantically different

Do not compare tensors with different semantics. Example: vLLM `kv_norm` may be pre-RoPE while RTP `kv` may be post-RoPE; recompute or dump the matching semantic point.

### 6. Diagnose Common RTP-vLLM DSV4 Root Causes

Check these before deeper kernel work:

- **RoPE/YaRN semantics**: SWA-only layers may still need YaRN when `original_seq_len`/rope scaling exists. vLLM DeepSeek-V4 uses base `rope_theta` for SWA but still applies YaRN scaling.
- **Invalid block ids**: SWA/state/cache block tables can contain invalid `-1` entries. Slot mapping must preserve invalid entries instead of converting them into readable cache slots.
- **Top-k kernel stability**: custom top-k kernels may return a stable set but unstable order. Sparse attention may consume order and amplify this.
- **Prefix cache/path mismatch**: first request and later requests can differ if vLLM prefix cache is enabled.
- **CUDA graph capture**: Python debug hooks may not execute on replay. For final parity, test the same graph setting as production.
- **BF16/FP8 cast points**: small ULP differences are not production bugs unless they cause token divergence or a known semantic mismatch.

### 7. Stage Only Deterministic Production Fixes

Before staging or reporting:

```bash
git status --short --branch
git diff --cached --name-only
git diff --cached --check
git diff --check
```

Stage only changes that are valid without debug flags. Keep these unstaged:

- teacher forcing hooks
- tensor dump and score dump plumbing
- experimental alternate kernels or roundtrip flags
- one-off scripts under `/tmp`

If a file is `MM`, inspect both staged and unstaged diffs:

```bash
git diff --cached -- path/to/file
git diff -- path/to/file
```

## Completion Criteria

A precision alignment task is complete only when one of these is true:

- RTP natural self-roll output matches the stable vLLM oracle for the requested length, and RTP repeats are stable if repeat stability was requested.
- Or, the first remaining mismatch is localized to a specific semantic or kernel difference with saved artifacts and exact first-diff evidence.

Final reports should state whether the final gate used teacher forcing. If it did, it is not a final output-alignment proof.

## Known-Good DSV4 Record89 Gate

When reproducing the 2026-05 DSV4 `/data3/q` precision case, the following gate is the expected known-good result. If this exact case does not match, do not proceed to new experiments until the mismatch is explained.

Case:

- input: `/data3/q`
- `record_index=89`
- prompt/input length observed in the run: `19587` tokens, a 19k long-context input. Do not replace it with a short prompt when reproducing this gate.
- sampling: `top_k=1`, `temperature=0.0`, `top_p=1.0`, `ignore_eos` path, `max_new_tokens=1000`
- RTP topology:
  - prefill CP=2, FP8 KV on
  - decode DP=1, FP8 KV on, CUDA graph on
  - natural self-roll; `RTP_TEACHER_FORCE_TOKENS` and `RTP_TEACHER_FORCE_OFFSET` unset

Required RTP precision switches for this gate:

```bash
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON
export DSV4_TORCH_TOPK=1
export DSV4_INDEXER_TOPK_BACKEND=torch
export DSV4_INDEXER_TOPK_BACKEND_OVERRIDE=torch
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_GATE_FP32=1
export DSV4_TORCH_ATTN=0
export FP8_KV_CACHE=1
export ENABLE_COMM_OVERLAP=0
# decode only:
export ENABLE_CUDA_GRAPH=1
```

Required vLLM oracle switches for this gate:

```bash
export VLLM_DSV4_TORCH_TOPK=1
export VLLM_DSV4_DISABLE_AUX_STREAMS=1
# launch with:
#   --enforce-eager --no-enable-prefix-caching
```

Known-good artifacts from the original alignment:

- vLLM oracle ids:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/vllm_stable_nodump_ignoreeos_len1000_oracle_20260517_220909_record89_20260517_220916/vllm_run01/generated_ids.json`
- RTP final self-roll ids:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_final_novllm_selfroll_len1000_20260518_101414_record89_20260518_101422/rtp_run01/generated_ids.json`

Expected diff result:

```text
rtp_len 1001
vllm_len 1000
first_diff None
equal_prefix1000 True
rtp_hash1000  986b77c92c844fc6
vllm_hash1000 986b77c92c844fc6
```

The known-good output still contains repetition, but the repetition is identical in RTP and vLLM:

```text
longest same-token run: start=263, len=4, token=24180
tail 9-token period repeats 54 times, total 486 tokens
period = [11454, 24, 62, 11030, 65, 939, 20996, 779, 65]
```

Do not use "the 1000-token output repeats" as evidence of RTP mismatch for this case. It is a matched oracle behavior. A real mismatch must be an exact token-id diff, a self-repeat instability between RTP runs, or a localized tensor/logit divergence under the same forced input.
