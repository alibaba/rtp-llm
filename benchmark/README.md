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

