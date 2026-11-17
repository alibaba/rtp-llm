# Qwen3.6-35B-A3B batch-invariant inference

The startup switch is `RTP_BATCH_INVARIANT=1`. An unset variable or `0`
selects the regular engine paths. The initial configuration uses an NVIDIA
SM103 GPU with `WORLD_SIZE=TP_SIZE=DP_SIZE=EP_SIZE=PP_SIZE=1`,
the official BF16 checkpoint, BF16 KV cache, 64-token pages, FP32 SSM state,
CUDA eager and ordinary text generation. The engine loads the text weights
from the multimodal checkpoint. It keeps the normal scheduler and dynamic
batching.

GDN resumes from the existing FP32 cache at the scheduler's aligned prefill
boundaries. Cross-request prefix reuse is rejected because a decode-generated
SSM state can differ from the state produced by prefill for the same prefix.
[SGLang prefill alignment](https://github.com/sgl-project/sglang/pull/45545) ·
[SGLang checkpoint provenance](https://github.com/sgl-project/sglang/pull/47580)

The two expert GEMMs in `TritonFusedMoeExecutor` use fixed tiles. The other
BF16 projections use a fixed K traversal and `computeLogits()` emits FP32.
Full attention uses RTP's RoPE and paged KV writers and one online softmax per
query and KV head. [vLLM MoE tiles](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/vllm/model_executor/determinism/batch_invariant_configs.py#L1449) ·
[vLLM 2D attention](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/vllm/v1/attention/ops/triton_unified_attention.py#L1155)

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
python3 "$tests/moe_test.py" --legacy
python3 "$tests/moe_test.py"
```

## Serving comparison

Use the standard RTP launcher and the existing model RPC. The acceptance test
checks the fixed continuation against the official tokenizer, counts all
returned FP32 logit bits, applies FP32 log-softmax to the full returned
vocabulary, and verifies the selected token's FP32 probability from RTP.
The batch log must prove co-batching, the target's row, and the target's prefill
grants before a case passes. [vLLM serving checks](https://github.com/vllm-project/vllm/blob/4bd291dc3c3771af5a81a4b822c29391d5bf2ae7/tests/v1/determinism/test_batch_invariance.py)

```bash
checkpoint=/mnt/fuse/Qwen/Qwen3.6-35B-A3B
export WORLD_SIZE=1 TP_SIZE=1 DP_SIZE=1 EP_SIZE=1 PP_SIZE=1
export LOCAL_WORLD_SIZE=1 WORLD_RANK=0 LOCAL_RANK=0 ROLE_TYPE=PDFUSION
export LOG_PATH=logs RTP_LLM_RECORD_DECODE_SAMPLE_RATE=1
export FRONTEND_PRE_STOP_DRAIN_SECONDS=1 DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS=1

run_case() (
    mode="$1"
    budget="$2"
    shift 2
    mkdir -p "$work/mode-${mode}-budget-${budget}"
    cd "$work/mode-${mode}-budget-${budget}"
    unset ACT_TYPE
    act=()
    if [ "$mode" = 0 ]; then act=(--act_type bf16); fi
    RTP_BATCH_INVARIANT="$mode" python3 -m rtp_llm.start_server \
        --checkpoint_path "$checkpoint" --tokenizer_path "$checkpoint" \
        --model_type qwen35_moe "${act[@]}" --max_seq_len 1024 --kv_cache_mem_mb 4096 \
        --start_port 19480 --prefill_chunk_size "$budget" > server.log 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null' EXIT
    python3 "$tests/service_test.py" --checkpoint "$checkpoint" "$@"
)

run_case 1 0 --save "$work/solo.npz"
run_case 1 192 --chunked --reference "$work/solo.npz"
run_case 0 0 --off-smoke
```

## Measured results

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

The 21 GDN comparisons matched across prefill output, convolution state,
FP32 SSM state and three decode steps, including recovery at 64 and 128
tokens. The MoE top-k IDs, weights and combined expert outputs matched
across the 64/65-token boundary and row swaps. With long decode peers, the
regular TRT-LLM attention path differed by 1,053 BF16 elements at KV length
193 and 2,150 at length 4097. The invariant path matched in both cases.

The longest service test prompt was 321 tokens. Longer service contexts
and performance remain unmeasured.
