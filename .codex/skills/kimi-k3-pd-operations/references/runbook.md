# Kimi K3 P/D runbook

## Hosts and paths

| Role | Host | Enter container | Repository | HTTP endpoint |
|---|---|---|---|---|
| Prefill | `11.163.39.114` | `cd /data0/xinfei.sxf/work && sh enter.sh` | `/home/xinfei.sxf/work/kimi/RTP-LLM/github-opensource` | `11.163.39.114:27188` |
| Decode | `11.163.39.115` | `cd /data1/xinfei.sxf/work && sh enter.sh` | `/home/xinfei.sxf/work/kimi_ft/RTP-LLM/github-opensource` | `11.163.39.115:28188` |

Connect with `ssh -tt xinfei.sxf@HOST`. Do not store passwords in this Skill.

## Canonical common configuration

```bash
CHECKPOINT_PATH=/data3/kimi-k3
TOKENIZER_PATH=/data3/kimi-k3
PREFILL_ENDPOINT=11.163.39.114:27188
DECODE_ENDPOINT=11.163.39.115:28188
KIMI_K3_EXECUTION_MODE=optimized
SP_TYPE=eagle3
SP_MODEL_TYPE=kimi_k3_mla_swa_eagle3
SP_CHECKPOINT_PATH=/mnt/nas1/hf/kimi3_eagle
GEN_NUM_PER_CIRCLE=3
KIMI_K3_EAGLE3_AUX_LAYER_IDS=0,44,88
```

Never set `KIMI_K3_SKIP_BUILD=1`. Every launch must run the startup script's
build and validation path.

## Prefill launch on 114

Use a `/data0` run root when `/tmp` is inode-exhausted.

```bash
nohup env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY='*' no_proxy='*' \
  CHECKPOINT_PATH=/data3/kimi-k3 \
  TOKENIZER_PATH=/data3/kimi-k3 \
  PREFILL_ENDPOINT=11.163.39.114:27188 \
  DECODE_ENDPOINT=11.163.39.115:28188 \
  KIMI_K3_EXECUTION_MODE=optimized \
  KIMI_K3_KDA_BACKEND=cula \
  KIMI_K3_KV_CACHE_MEM_MB=4096 \
  KIMI_K3_RUN_ROOT=/data0/xinfei.sxf/k3-pd-prefill \
  KIMI_K3_TMPDIR=/data0/xinfei.sxf/k3-pd-prefill-tmp \
  KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS=1 \
  SP_TYPE=eagle3 \
  SP_MODEL_TYPE=kimi_k3_mla_swa_eagle3 \
  SP_CHECKPOINT_PATH=/mnt/nas1/hf/kimi3_eagle \
  GEN_NUM_PER_CIRCLE=3 \
  KIMI_K3_EAGLE3_AUX_LAYER_IDS=0,44,88 \
  bash example/start_kimi_k3_pd.sh prefill \
  > /data0/xinfei.sxf/k3-prefill-launch.log 2>&1 &
```

The launcher must reject every Prefill backend except `cula`:

```text
error: Kimi K3 Prefill requires KIMI_K3_KDA_BACKEND=cula; got kernel
```

## Decode launch on 115

```bash
nohup env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY='*' no_proxy='*' \
  CHECKPOINT_PATH=/data3/kimi-k3 \
  TOKENIZER_PATH=/data3/kimi-k3 \
  PREFILL_ENDPOINT=11.163.39.114:27188 \
  DECODE_ENDPOINT=11.163.39.115:28188 \
  KIMI_K3_DECODE_TOPOLOGY=tp8_ep8 \
  KIMI_K3_EXECUTION_MODE=optimized \
  KIMI_K3_KDA_BACKEND=kernel \
  KIMI_K3_TARGET_VERIFY_KDA_BACKEND=kernel \
  KIMI_K3_KV_CACHE_MEM_MB=2048 \
  KIMI_K3_RUN_ROOT=/data1/xinfei.sxf/k3-pd-decode \
  KIMI_K3_TMPDIR=/data1/xinfei.sxf/k3-pd-decode-tmp \
  KIMI_K3_FLASHINFER_WORKSPACE_BASE=/data1/xinfei.sxf/k3-pd-flashinfer \
  SP_TYPE=eagle3 \
  SP_MODEL_TYPE=kimi_k3_mla_swa_eagle3 \
  SP_CHECKPOINT_PATH=/mnt/nas1/hf/kimi3_eagle \
  GEN_NUM_PER_CIRCLE=3 \
  KIMI_K3_EAGLE3_AUX_LAYER_IDS=0,44,88 \
  ENABLE_CUDA_GRAPH=1 \
  DECODE_CAPTURE_CONFIG=1,2,3,4,5,6,7,8 \
  RTP_MLA_DECODE_KERNEL=flashinfer \
  LOAD_METHOD=fastsafetensors \
  bash example/start_kimi_k3_pd.sh decode \
  > /data1/xinfei.sxf/k3-pd-logs/decode-cudagraph-fastsafetensors.log 2>&1 &
```

`fastsafetensors` is the normal launch mode and loaded the main weights in
about 135-153 seconds during the validated run. Use `LOAD_METHOD=scratch` only
as a load-time memory diagnostic; it performs CPU conversion for an estimated
1.28 TB model and is much slower.

## Readiness and proxy diagnosis

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
pgrep -af 'server_main.py|kimi_k3_prefill_server'
curl --noproxy '*' -sS -m 5 -w ' HTTP=%{http_code}\n' http://127.0.0.1:PORT/health
```

Both endpoints must return `"ok" HTTP=200`. Tinyproxy HTML errors mean the client inherited proxy variables; for Python urllib use:

```python
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
```

If disk capacity is available but installation reports `No space left on device`, check `df -i`. On 114, `/tmp` inode exhaustion previously required relocating the run root and temporary directory to `/data0`.

## HumanEval request and acceptance calculation

Input collection:

```text
/mnt/nas1/hf/MAL_test_codes/samples/humaneval_0000_turn0.pt
...
/mnt/nas1/hf/MAL_test_codes/samples/humaneval_0004_turn0.pt
```

Each file contains `input_ids` for prompt plus reference response and an
`input_len` boundary. Slice `dump["input_ids"][:dump["input_len"]]`, decode it
with `/data3/kimi-k3`, encode the resulting prompt again, and require exact ID
equality before sending it. The old `kimi_k3_accuracy_input_ids` request field
is no longer supported. Run `scripts/run_humaneval.py`; it performs the strict
round trip and writes complete responses and output IDs to JSON.

Use top-level request fields at `/`, not a `query` wrapper:

```json
{
  "prompt": "<decoded prompt whose re-encoded IDs exactly match>",
  "generate_config": {
    "max_new_tokens": 1000,
    "top_k": 1,
    "top_p": 0,
    "temperature": 0.0,
    "ignore_eos": false,
    "skip_special_tokens": false,
    "return_input_ids": true,
    "return_output_ids": true,
    "can_use_pd_separation": true,
    "force_disable_sp_run": false
  }
}
```

For `GEN_NUM_PER_CIRCLE=3`:

```text
accepted = output_len - iter_count
proposed = (iter_count - 1) * 3
acceptance_rate = accepted / proposed
```

Known-good five-query results with Prefill CULA:

| Query | Output | Accepted/proposed | Rate | Last token |
|---|---:|---:|---:|---:|
| 0000 | 547 | 396/450 | 88.00% | 163585 |
| 0001 | 742 | 492/747 | 65.86% | 163585 |
| 0002 | 232 | 159/216 | 73.61% | 163585 |
| 0003 | 331 | 234/288 | 81.25% | 163585 |
| 0004 | 318 | 224/279 | 80.29% | 163585 |

Weighted rate: `1505/1980 = 76.01%`. All five end at EOS token `163585`.

Latest validated result with Draft `/mnt/nas1/hf/kimi3_eagle`, MAL samples
`/mnt/nas1/hf/MAL_test_codes`, and kernel target verify:

| Query | Input | Output | Accepted/proposed | Rate | Last token |
|---|---:|---:|---:|---:|---:|
| 0000 | 227 | 568 | 408/477 | 85.53% | 163585 |
| 0001 | 217 | 655 | 442/636 | 69.50% | 163585 |
| 0002 | 187 | 232 | 160/213 | 75.12% | 163585 |
| 0003 | 220 | 331 | 235/285 | 82.46% | 163585 |
| 0004 | 220 | 303 | 217/255 | 85.10% | 163585 |

Weighted rate: `1462/1866 = 78.35%`. All five outputs are sane and end at EOS.

### Validated CUDA Graph result

Validated on commits `3917913e9` (target-verify CUDA Graph) and `96f85a339`
(target-verify Decode state fix), with `fastsafetensors`, Decode capture sizes
`1..8`, KV cache 2048 MB, and kernel target verify:

| Query | Output | Accepted/proposed | Rate | Termination |
|---|---:|---:|---:|---|
| 0000 | 1000 | 644/1065 | 60.47% | max tokens |
| 0001 | 1000 | 595/1212 | 49.09% | max tokens |
| 0002 | 428 | 300/381 | 78.74% | EOS 163585 |
| 0003 | 1000 | 727/816 | 89.09% | max tokens |
| 0004 | 1000 | 713/858 | 83.10% | max tokens |

Weighted rate: `2979/4332 = 68.77%`, versus the immediately preceding
non-CUDA-Graph baseline `72.68%`. All responses matched their HumanEval task;
maximum consecutive identical-token runs were 1, except query 1 at 2. The
complete result is
`/data1/xinfei.sxf/k3-pd-logs/humaneval-modefix-cudagraph.json`.

Confirm the live Decode command includes `--enable_cuda_graph 1` and
`--decode_capture_config 1,2,3,4,5,6,7,8`; do not infer current graph state
from older log entries in reused log directories.

## GPU contention and automatic retry

Before launch, require zero compute processes on all eight GPUs. Check on the
host as well as inside the container because another container's PIDs may be
invisible inside the target container:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

If external processes occupy any card, identify them with `ps` on the host and
never signal them. Poll about once per minute; launch immediately only after
all eight cards report no compute process and negligible used memory. A rank
that exits with code `-9` while another user's jobs appear at the same time is
GPU scheduler contention, not evidence of a model/configuration failure.

Prefill `kernel` produced misleading rates near 99% while outputting repeated token `0`, `!`, or a fixed phrase. Always inspect output IDs and decoded text.

Before launch, inspect both `git status --short` and `git diff --cached`. A
staged diagnostic rollback can silently undo the CUDA Graph implementation in
the current HEAD. Preserve the fused D2D copies, immutable host capture
descriptors, target-verify Decode MLA selection, and separate draft-prefill
graph gate introduced by `3917913e9` or its rebased equivalent.

## Hidden-state comparison

MAL references:

```text
/mnt/nas1/hf/MAL_test_codes/samples/humaneval_0000_turn0.pt
...
/mnt/nas1/hf/MAL_test_codes/samples/humaneval_0004_turn0.pt
```

Each MAL file contains `hidden: [3,total_seq_len,7168]`, `input_len`, and `aux_layers: [1,45,89]`. Compare only `hidden[:, :input_len, :]` against RTP layers `[0,44,88]`.

Latest stable RTP dump paths:

```text
/data0/xinfei.sxf/k3hidden-current-combined/
/data0/xinfei.sxf/k3hidden-current-compare.json
```

The latest CULA rerun is bitwise identical to the earlier RTP CULA dump (`max_abs=0`, exact ratio `1.0`) across all five prompts and three layers. RTP versus MAL is not numerically aligned:

| RTP/MAL layers | Cosine range | RMSE range | Interpretation |
|---|---:|---:|---|
| 0 / 1 | 0.8274–0.8340 | 0.00724–0.00757 | Some similarity, not equivalent |
| 44 / 45 | 0.6297–0.6626 | 0.21691–0.22581 | Not similar |
| 88 / 89 | 0.6132–0.6485 | 0.21716–0.22427 | Not similar |

Conclude that RTP CULA is stable and reproducible, but its hidden states are not aligned with MAL, especially in the middle and late layers.
