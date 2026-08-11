## Benchmark
### Bench Server
```bash
# prepare test data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# launch start server
# please use GUARANTE_GENERATE_MEM to avoid lack of memory error when generate
# WEIGHT_TYPE can choose int8, fp16, bf16, fp32
# INT8_KV_CACHE mean use int8 store kv cache, default fp16
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export GUARANTE_GENERATE_MEM=1
export WEIGHT_TYPE=fp16
export INT8_KV_CACHE=0
python3 -m rtp_llm.start_server

# benchmark service
python3 ./benchmark_serving.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /path/to/tokenizer --num-prompts 10000 --trust-remote-code --backend rtp-llm --max-batch-size 64
```

## Performance

### Qwen-7B-Chat
test vllm-0.2.6 whl and rtp-llm
#### A10
<img src=../picture/A10_perf_data.png width="600px">



more test data are on the way!

## Streaming tool-call stress

`benchmark_tool_call.py` sends request-scoped tool arguments through the OpenAI
streaming endpoint. The default profile requires one named tool call per request;
it checks request isolation, stream termination, chunk structure, tool-call
arguments, call ID uniqueness, and model semantics. A deterministic fraction of
requests is cancelled after dispatch, and `/worker_status` is compared before and
after the recovery window.

```bash
python3 benchmark/benchmark_tool_call.py \
    --base-url http://127.0.0.1:30000/v1 \
    --model model-name \
    --requests 1000 \
    --concurrency 32 \
    --cancel-rate 0.1 \
    --output tool_call_stress.json
```

Use a separate run to require two tool calls in one response and validate the
parallel-call protocol:

```bash
python3 benchmark/benchmark_tool_call.py \
    --base-url http://127.0.0.1:30000/v1 \
    --model model-name \
    --requests 1000 \
    --concurrency 32 \
    --tool-choice required \
    --parallel-tool-calls \
    --output parallel_tool_call_stress.json
```

Structural and semantic failures return a non-zero exit code by default. The
reported `throughput_rps` and latency percentiles include only fully successful
non-cancelled requests; `attempted_rps` includes the complete offered workload.
Per-request latency and TTFT start after the local concurrency slot is acquired,
while wall time and throughput include the complete workload and its client queue.
`--allow-semantic-errors` is available only when semantic mismatches should be
reported without failing the process.
