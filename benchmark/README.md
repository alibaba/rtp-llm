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

### MiniMax M3VL ViT Request-Concurrency Baseline

`multimodal_m3_vit_concurrency.py` measures the ViT service core path with one
image per request:

```text
decoded CPU RGB -> MMScheduler -> H2D -> GPU transform -> batched ViT
-> image-token assembly -> CUDA completion
```

Download, image decode, RPC transport, cache hits, and LLM prefill are excluded
so the result can be used to compare scheduler, GPU preprocessing, and ViT
kernel changes. The default image cases are 448x448, 1920x1080, and 2560x1440.
Request concurrency is swept over `1,2,4,8,16,32,64` with one image per request
and batch request/media caps of 64. C32 is the fixed high-concurrency comparison
point. "Maximum throughput" means the highest selected result in this sweep
under the configured cap; the JSON records the actual average and maximum formed
batch sizes.

Each point is measured three times for at least 10 seconds per repetition. The
complete repetition whose throughput is the median is used in CSV and charts;
every repetition is retained in JSON. Before every repetition the script waits
for an idle GPU window. A repetition is discarded and retried if another GPU
becomes more than 50% busy during it.
The initial run also requires every GPU to use at most 4 GiB, and later samples
require every non-target GPU to remain below that memory watermark. This avoids
recording an unrelated resident model in the absolute NVML memory result.
GPU memory is reported both as the PyTorch peak allocated-memory increase above
the loaded-model baseline and as the absolute/delta NVML process-device
watermark.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/multimodal_m3_vit_concurrency.py \
    --checkpoint /path/to/MiniMax-M3 \
    --concurrencies 1,2,4,8,16,32,64 \
    --requests-per-point 128 \
    --minimum-point-seconds 10 \
    --repeats 3 \
    --idle-memory-threshold-mib 4096 \
    --batch-wait-ms 10 \
    --max-batch-size 64 \
    --max-batch-images 64
```

The output directory contains:

- CSV with the selected baseline repetition for every image/concurrency point;
- JSON with metadata, all valid repetitions, and discarded repetitions;
- a four-panel PNG covering throughput, P50/P99 latency, GPU utilization, and
  peak allocated-memory growth.
