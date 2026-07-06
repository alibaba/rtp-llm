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

### Topology-derived KV candidate schedule

`topology_kv_candidate_schedule.py` is an opt-in CUDA screening benchmark for
long-context decode attention. It does not change RTP-LLM serving behavior.

The benchmark builds rectangular, request-local candidate rows from KV block
metadata:

- sink blocks keep early attention-sink evidence,
- local blocks keep the newest causal context,
- high-drift blocks keep topology-style witness blocks where K centroids move
  sharply between neighboring blocks.

The CUDA benchmark compares PyTorch dense SDPA decode attention against a
selected-token decode path that gathers candidate K/V rows and runs attention
over only those selected tokens. Report the result as a benchmark signal only;
end-to-end model speedup still requires integration with the runtime sparse MLA
or indexer path and model-quality validation.

```bash
python benchmark/topology_kv_candidate_schedule.py \
  --seq-len 8192 \
  --selected-tokens 256 512 1024 \
  --heads 16 \
  --head-dim 64 \
  --rounds 50 \
  --warmup 20 \
  --device cuda
```

Example output:

```text
| seq_len | selected_tokens | dense_sdpa_ms | sparse_selected_ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 256 | 0.3740 | 0.0998 | 3.75x |
```

