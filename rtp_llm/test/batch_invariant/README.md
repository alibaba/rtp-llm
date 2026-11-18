# Qwen BF16 batch-invariant inference

The startup switch is `RTP_BATCH_INVARIANT=1`. An unset variable or `0`
selects the regular engine paths. Use NVIDIA L20D (SM103),
`DP_SIZE=EP_SIZE=PP_SIZE=1`, the official BF16 checkpoint, BF16 KV cache,
64-token pages, CUDA eager and ordinary text generation. Dynamic batching
continues to run with the switch enabled.

| Checkpoint | RTP model type | TP |
| --- | --- | --- |
| Qwen/Qwen3-8B | `qwen_3` | 1 |
| Qwen/Qwen3-30B-A3B | `qwen_3_moe` | 1 |
| Qwen/Qwen3.6-35B-A3B | `qwen35_moe` | 1 |
| Qwen/Qwen3.5-397B-A17B | `qwen35_moe` | 4 |

Set `WORLD_SIZE=LOCAL_WORLD_SIZE=TP_SIZE` for the selected topology.
The hybrid models use FP32 SSM state and load the text weights from their
multimodal checkpoints. The standard Qwen3 models use the top-level checkpoint
dtype and full GQA attention throughout.

GDN resumes from the existing FP32 cache at the scheduler's aligned prefill
boundaries. Its Q/K normalization uses a fixed row kernel across the 128-row
dispatch boundary when this mode is enabled. Cross-request prefix reuse is
rejected because a decode-generated SSM state can differ from the state
produced by prefill for the same prefix.
[SGLang prefill alignment](https://github.com/sgl-project/sglang/pull/45545) ·
[SGLang checkpoint provenance](https://github.com/sgl-project/sglang/pull/47580)

The two expert GEMMs in `TritonFusedMoeExecutor` use fixed tiles. The other
BF16 projections use a fixed K traversal and `computeLogits()` emits FP32.
Full attention uses RTP's RoPE and paged KV writers and one online softmax per
query and KV head. [vLLM MoE tiles](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/vllm/model_executor/determinism/batch_invariant_configs.py#L1449) ·
[vLLM 2D attention](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/vllm/v1/attention/ops/triton_unified_attention.py#L1155)

On TP4 the attention and expert outputs need a cross-rank sum. The batch-invariant
path gathers the rank contributions using NCCL and accumulates each element in
FP32 in ascending rank order. RTP continues to use its existing rank-ordered
logit gather after the FP32 LM-head projection on each vocabulary shard.
[vLLM fixed-order reduction](https://github.com/vllm-project/vllm/blob/748f1ab353d77900e1b55961b331d4cc1cd72edc/vllm/model_executor/determinism/batch_invariant.py#L1130)

## Build and operator checks

Run in the CUDA13 development container with Python 3.10 and Bazel 6.4. The
commands reuse the locked Python dependencies Bazel fetched for the wheel.
Run the scripts from outside the checkout so imports resolve to the built package.

```bash
repo="$(git rev-parse --show-toplevel)"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
export PIP_EXTRA_INDEX_URL=https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/
bazel build --config=cuda13 \
    --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3 \
    --action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3 \
    --host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3 \
    //rtp_llm:rtp_llm_whl

external="$(bazel info output_base)/external"
work="$(mktemp -d)"
python3.10 -m venv --system-site-packages "$work/venv"
source "$work/venv/bin/activate"
python3 -m pip install --no-deps "$repo"/bazel-bin/rtp_llm/rtp_llm-*.whl
export PYTHONPATH="$(printf '%s:' "$external"/pip_gpu_cuda13_torch_*/site-packages)"
cuda_libs="$(find "$external"/pip_gpu_cuda13_torch_* -path '*/site-packages/nvidia/*/lib' -type d -printf '%p:')"
python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="${cuda_libs}/usr/local/cuda/lib64:${python_lib}:${LD_LIBRARY_PATH:-}"
cd "$work"
export CUDA_VISIBLE_DEVICES=0
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
tests="$repo/rtp_llm/test/batch_invariant"
python3 "$tests/config_test.py"
python3 "$tests/gdn_test.py"
python3 "$tests/kernel_test.py" --trtllm-baseline
python3 "$tests/kernel_test.py" --qwen3-8b --trtllm-baseline
python3 "$tests/moe_test.py" --legacy
python3 "$tests/moe_test.py"
python3 "$tests/moe_test.py" --qwen3-30b --legacy
python3 "$tests/moe_test.py" --qwen3-30b
```

## Serving comparison

Use the standard RTP launcher and the existing model RPC. The acceptance test
checks the fixed continuation against the official tokenizer, counts all
returned FP32 logit bits, applies FP32 log-softmax to the full returned
vocabulary, and verifies the selected token's FP32 probability from RTP.
The checkpoint config provides the vocabulary dimension for each RPC row.
The batch log must prove co-batching, the target's row, and the target's prefill
grants before a case passes. [vLLM serving checks](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/tests/v1/determinism/test_batch_invariance.py)

```bash
checkpoint=/mnt/fuse/Qwen/Qwen3.6-35B-A3B
model_type=qwen35_moe
start_port=19480
export WORLD_SIZE=1 TP_SIZE=1 DP_SIZE=1 EP_SIZE=1 PP_SIZE=1
export LOCAL_WORLD_SIZE=1 WORLD_RANK=0 LOCAL_RANK=0 ROLE_TYPE=PDFUSION
export LOG_PATH=logs RTP_LLM_RECORD_DECODE_SAMPLE_RATE=1
export FRONTEND_PRE_STOP_DRAIN_SECONDS=1 DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS=1

run_case() (
    set -eo pipefail
    mode="$1"
    budget="$2"
    shift 2
    mkdir -p "$work/mode-${mode}-budget-${budget}"
    cd "$work/mode-${mode}-budget-${budget}"
    unset ACT_TYPE
    act=()
    if [ "$mode" = 0 ] && [ "$model_type" = qwen35_moe ]; then act=(--act_type bf16); fi
    RTP_BATCH_INVARIANT="$mode" python3 -m rtp_llm.start_server \
        --checkpoint_path "$checkpoint" --tokenizer_path "$checkpoint" \
        --model_type "$model_type" "${act[@]}" --max_seq_len 1024 --kv_cache_mem_mb 4096 \
        --start_port "$start_port" --prefill_chunk_size "$budget" > server.log 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null' EXIT
    if [ "$mode" = 1 ] && [ "$budget" = 0 ]; then
        python3 "$tests/service_test.py" --checkpoint "$checkpoint" \
            --address "127.0.0.1:$((start_port + 1))" --greedy-smoke | tee greedy.log
    fi
    python3 "$tests/service_test.py" --checkpoint "$checkpoint" \
        --address "127.0.0.1:$((start_port + 1))" "$@" 2>&1 | tee rpc.log
)

run_case 1 0 --save "$work/solo.npz"
run_case 1 192 --chunked --reference "$work/solo.npz"
run_case 0 0 --greedy-smoke
```

To reproduce the TP1 comparison on both standard Qwen3 checkpoints, use
the same `run_case` function with a fresh directory for each model.

```bash
export CUDA_VISIBLE_DEVICES=0
checkpoint=/mnt/fuse/Qwen/Qwen3-8B
model_type=qwen_3
work="$(mktemp -d)"
run_case 1 0 --save "$work/solo.npz"
run_case 1 192 --chunked --reference "$work/solo.npz"
run_case 0 0 --greedy-smoke

export CUDA_VISIBLE_DEVICES=1
checkpoint=/mnt/fuse/Qwen/Qwen3-30B-A3B
model_type=qwen_3_moe
work="$(mktemp -d)"
run_case 1 0 --save "$work/solo.npz"
run_case 1 192 --chunked --reference "$work/solo.npz"
run_case 0 0 --greedy-smoke
```

## Standard Qwen3 checkpoint revisions and regular-path comparison

The service read these weights from `/mnt/fuse/Qwen` with `ACT_TYPE` unset.
The revision comes from the mount metadata. Every tensor in the indexed
safetensors shards was BF16.

| Checkpoint | Revision | Vocabulary | GPU for serving |
| --- | --- | --- | --- |
| Qwen3-8B | [`b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | 151936 | L20D 0, TP1 |
| Qwen3-30B-A3B | [`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39) | 151936 | L20D 1, TP1 |

The regular path was also sampled against the solo baseline with prefill
budget 0 and the target first in the `193, 11, 321` batch. All four token IDs
matched for both models. The first raw logit difference was at output step 0,
vocabulary index 0. Indices in the RPC report are zero-based.

| Checkpoint | Raw logit differences | Full-vocab logprob differences | Selected-token logprob differences | First raw logit bits, solo to mixed |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 433271 | 444474 | 1 | `0x41133d58` → `0x41133d57` |
| Qwen3-30B-A3B | 352864 | 70831 | 0 | `0x413f21fa` → `0x413f21fb` |

Set `work="$(mktemp -d)"` and run `run_case 0 0 --save "$work/regular-solo.npz"`
with either checkpoint above to reproduce the regular-path difference.
The test verifies real prefill and decode co-batching and prints the first
different index and bit patterns before exiting with an assertion.

## Standard Qwen3 results with the switch enabled

Both services ran CUDA eager on NVIDIA L20D (SM103, 275040 MiB) with CUDA
13.2, PyTorch 2.11.0+cu130 and Triton 3.6.0. The tests used a 1024-token
sequence limit and a 4096 MiB KV pool. Each comparison uses the unchunked solo
request as its reference, with fixed output `[2518, 6303, 6176, 13753]`.
Each of the four steps returned 151936 raw FP32 logits and one selected-token
FP32 probability. The test computed FP32 logprobs over the full vocabulary
and the logprob of the selected token from the returned probability.

| Model | Prefill budget | Case | Target grants | Steps | Token IDs match | Logit diffs | Full logprob diffs | Selected logprob diffs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 0 | Target first, `193, 11, 321` | 193 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-8B | 0 | Target last, `321, 11, 193` | 193 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-8B | 192 | Solo | 192 + 1 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-8B | 192 | Target first, `193, 11` | 192 + 1 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-8B | 192 | Target last, `11, 193` | 128 + 65 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-30B-A3B | 0 | Target first, `193, 11, 321` | 193 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-30B-A3B | 0 | Target last, `321, 11, 193` | 193 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-30B-A3B | 192 | Solo | 192 + 1 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-30B-A3B | 192 | Target first, `193, 11` | 192 + 1 | 4 | Yes | 0 | 0 | 0 |
| Qwen3-30B-A3B | 192 | Target last, `11, 193` | 128 + 65 | 4 | Yes | 0 | 0 | 0 |

The execution recorder verified the actual target row and phase in each
mixed prefill and decode case. Each model generated eight ordinary greedy
tokens with the switch on and off. Qwen3-8B produced the same eight tokens
in both states. On Qwen3-30B-A3B, the seventh token was `9934` with the
switch off and `1467` with the switch on.

The 8B operator checks found 8 BF16 differences in the regular attention
output projection, up to 27 in the regular dense MLP and 150159 FP32
differences in the regular LM head. The existing invariant kernels matched
bitwise for these shapes. GQA prefill and decode, Q/K normalization, and
the 30B router logits, top-k IDs, weights and expert output also matched
bitwise across their batch and row changes. The configuration suites ran
78 tests. A real 8B launch with `--act_type fp16` failed its startup
BF16 validation.

The container logs are under `/home/admin/rtp-bi-qwen3-1118`. The `runs`
subdirectories for each model contain `off`, `on-budget0` and `on-budget192`
runs with `server.log`, `rpc.log` and `logs/batch_schedule.*.log`.
`operators` and `config-regression` contain the operator and config outputs.
The Qwen3.6-35B-A3B TP1 comparison also passed with `[4, 248320]` logits.
Qwen3.5-397B-A17B TP4 was checked in configuration and rank-local operator
tests. Its service comparison is from the earlier TP4 run below.

## Qwen3.6-35B-A3B results

The CUDA13 wheel ran on one NVIDIA L20D (SM103, 275040 MiB) with CUDA
13.2, PyTorch 2.11.0+cu130 and Triton 3.6.0. The service used the mounted
official checkpoint, a 1024-token sequence limit and a 4096 MiB KV pool.
The enabled service selected BF16 with `ACT_TYPE` unset.

| Prefill budget | Target prefill grants | Same-phase batch prompt lengths | Target row (zero-based) | FP32 mismatches |
| --- | --- | --- | --- | --- |
| 0 | 193 | `193, 11, 321` | 0 | 0 |
| 0 | 193 | `321, 11, 193` | 2 | 0 |
| 192 | 192 + 1 | Solo, compared with budget 0 | 0 | 0 |
| 192 | 192 + 1 | `193, 11` | 0 | 0 |
| 192 | 128 + 65 | `11, 193` | 1 | 0 |

For each comparison, the four continuation positions contributed 993,280
logits and 993,280 full-vocabulary FP32 logprobs. The selected-token
logprobs from RTP's returned FP32 probabilities also matched. The execution
recorder showed both prefill and decode co-batching in each mixed case.
With the switch off, greedy generation returned eight tokens
`[760, 1414, 369, 25, 271, 248068, 271, 248069]`.

The 22 GDN comparisons matched across prefill output, convolution state,
FP32 SSM state and three decode steps, including recovery at 64, 128 and 192
tokens. The MoE top-k IDs, weights and combined expert outputs matched
across the 64/65-token boundary and row swaps. With long decode peers, the
regular TRT-LLM attention path differed by 1,053 BF16 elements at KV length
193 and 2,150 at length 4097. The invariant path matched in both cases.

The longest service test prompt was 321 tokens. Longer service contexts
and performance remain unmeasured.

## Qwen3.5-397B-A17B verification

After the build and the `run_case` definition above, select the TP4 checkpoint
and a separate output directory. Each service invocation loads the checkpoint
from the existing mount.

```bash
checkpoint=/mnt/fuse/Qwen/Qwen3.5-397B-A17B
model_type=qwen35_moe
work="$(mktemp -d)"
cd "$work"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4 LOCAL_WORLD_SIZE=4 TP_SIZE=4
export DP_SIZE=1 EP_SIZE=1 PP_SIZE=1
python3 -m torch.distributed.run --standalone --nproc-per-node=4 "$tests/tp_reduce_test.py" --legacy
python3 -m torch.distributed.run --standalone --nproc-per-node=4 "$tests/tp_reduce_test.py"
python3 "$tests/gdn_test.py" --tp4-397b
python3 "$tests/moe_test.py" --tp4-397b

run_case 1 0 --save "$work/solo.npz"
run_case 1 192 --chunked --reference "$work/solo.npz"
run_case 0 0 --greedy-smoke
```

The 397B checkpoint's index contains 806,795,875,168 bytes of BF16 weights.
The tested topology uses four L20D GPUs with 275040 MiB per GPU, CUDA 13.2,
PyTorch 2.11.0+cu130 and Triton 3.6.0. The service configuration uses
`--max_seq_len 1024 --kv_cache_mem_mb 4096` and leaves `ACT_TYPE` unset when
the mode is enabled.

The baseline used the regular engine with the same TP4 topology. The
execution recorder confirmed a mixed prefill and mixed decode batch with
prompt lengths `193, 11, 321`. All 993,280 returned FP32 logits differed
from the target's solo output. In the isolated TP4 sum test, the first
4096-dimensional hidden row had 1,309 differing BF16 elements when another
row joined it. The fixed-order path passed 32 batch-size and placement
comparisons on all four ranks with zero differing elements.

The first TP4 implementation also exposed a short-chunk Q/K normalization
difference. The real service agreed through layer 7, then differed in
layer 8's GDN output when the final prefill grant shrank to one token.
The standalone Q/K test reproduced two BF16 differences across the
128-row kernel dispatch. Selecting the existing fixed row kernel eliminated
those differences. The FP32 SSM state at token 192 matched between the
unsplit run, the saved page and the restored page in the service trace.

The final 397B comparisons use the same target prompt and fixed continuation
`[2438, 6105, 5983, 13358]` as the 35B test. Each comparison checks 993,280
raw FP32 logits, 993,280 full-vocabulary FP32 logprobs computed from those
returned logits, and the selected-token logprobs derived from RTP's returned
FP32 probabilities. The execution records prove the prefill grants, the target
row and same-phase co-batching independently of the client request order.

| Prefill budget | Target grants | Mixed prompt lengths | Target row (zero-based) | Logit mismatches | FP32 logprob mismatches |
| --- | --- | --- | --- | --- | --- |
| 0 | 193 | `193, 11, 321` | 0 | 0 | 0 |
| 0 | 193 | `321, 11, 193` | 2 | 0 | 0 |
| 192 | 192 + 1 | Solo, compared with budget 0 | 0 | 0 | 0 |
| 192 | 192 + 1 | `193, 11` | 0 | 0 | 0 |
| 192 | 128 + 65 | `11, 193` | 1 | 0 | 0 |

Restarting the final wheel with budget 0 and comparing its solo result against
the budget-192 result also passed. Its mixed execution IDs were
`1794936223835181166` (prefill, target row 0), `1794936223835181167`
(decode, target row 0), `1794936223835181238` (prefill, target row 2), and
`1794936223835181239` (decode, target row 2). The budget-192 mixed execution
IDs were `1794935961931444038` / `1794935961931444039` for target row 0 and
`1794935961931444109` / `1794935961931444111` for target row 1.

The 22 rank-local GDN comparisons passed, including the FP32 SSM state at
64, 128 and 192 tokens and three decode steps. MoE routing IDs, weights and
the combined expert output passed the 64/65-token and row-swap checks at
the TP4-local dimensions. Existing collective and configuration tests passed
(38 + 10 + 8 + 9 cases). Qwen3.6-35B-A3B's single-GPU service still passed
the mixed short/long request and row-swap comparisons after the GDN change.
With `RTP_BATCH_INVARIANT=0`, the final TP4 service generated eight greedy
tokens `[248046, 198, 248045, 74455, 198, 248068, 271, 248069]`.

The container's run directory is
`/home/admin/rtp-bi-01a1e936/397b-invariance.QbDotw`. It contains the
`verified0.log`, `verified192.log`, `final-off.log`, `tp-final.log`,
`trace-diff.log`, and the per-service `logs/batch_schedule.*.log` files.

The longest service test prompt was 321 tokens. Long-context service tests,
throughput and latency remain unmeasured. Other GPU architectures, parallel
topologies, quantization, CUDA graphs and cross-request prefix reuse remain
outside the supported configuration and fail their startup checks.
