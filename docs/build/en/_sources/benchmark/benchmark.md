# RTP-LLM Performance Benchmark Tool
In this chapter, I will present the performance testing tools developed in RTP-LLM, including standalone measurement of model prefill and decode performance under various batch sizes with single-node and multi-node parallelism, timeline recording, and their usage methods.
## Design Principle
RTP-LLM employs a special batch scheduler that accumulates requests until the specified batch size is reached, then all requests enter the engine simultaneously. The scheduler supports both prefill and decode modes; in decode mode, requests are only allocated KV cache without prefill, enabling accurate and efficient measurement of engine performance.
In detail, the batch-scheduler profiler is executed three times for a single input:
 - Warm-up run, to account for one-time setup such as JIT compilation.
 - Timing run, to measure engine performance.
 - Profiling run, to capture a timeline for subsequent analysis; this step may degrade end-to-end performance.

For every run we use min_new_tokens and max_new_tokens to ensure that all requests perform the same number of decode steps.

Since in decode mode, we not prefill the KVCache, so hidden_states after every forward step is not real. So we also hack moe gate select for moe model and speculative accept func for mtp. In that way, we get stable result for analyse result.
## Single-Node Benchmark
using commands below can start a performance benchmark, mixed prefill and decode
```shell
bazelisk test //rtp_llm/test/perf_test:perf_test \
    --config=cuda12_6 \
    --test_arg=--ckpt_path=${/PATH/TO/CKPT} \
    --test_arg=--tokenizer_path=${/PATH/TO/TOKENIZER} \
    --test_arg=--model_type=${MODEL_TYPE} \
    --test_arg=--dp_size=1 \
    --test_arg=--tp_size=1 \
    --test_arg=--batch_size="1,2,4,8,16,32" \
    --test_arg=--input_len="128,1024,2048,4096" \
    --test_env=INT8_MODE=1 # optionally using --test_env to using custom env for RTP-LLM setup
```
specially `batch_size` states for the batch in single DP node, when `DP_SIZE` param is setted.

also we support test prefill or decode only when prefill and decode not share the same config(such as prefill use deepep normal, and decode use deepep masked), in that case user should also set `--partial={0:all(default), 1:decode, 2:prefill}`, below is an example of testing decode only:
```shell
    --test_arg=partial=1
```
## Multi Node Benchmark
We also provide a Python script to enable multi-node benchmark, but since it requires setting up the environment and starting the script on multiple machines, it still involves more steps than single-node testing.

For each machine to be tested, you need to create an environment in which RTP-LLM can run and support passwordless SSH access from the current machine’s port.
### Benchmark Yaml
Configure the parameters with reference to `rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml`. Below is the detail explaination of yaml structure.

First part is for cloning code to local in ssh machine, and checkout to the branch you need to test
```yaml
benchmarks:
  - name: "H20_Deepseek-R1_Decode_EP32_4K"
    # git config
    git_repo_url: "git@github.com:alibaba/rtp-llm.git"
    git_checkout_ref: "origin/main"
```
Second part describe the machine list, user name and port to ssh
```yaml
    # machine config
    ip_lists:
      - "33.126.67.231"
      - "33.126.67.17"
      - "33.126.51.159"
      - "33.126.83.168"
    run_user: "admin"
    ssh_port: 2222
    # model config
```
Third part describe the model info which should be pre-download to local machine
``` yaml
    tokenizer_path: "/mnt/nas1/hf/deepseek_r1_4layers/"
    checkpoint_path: "/mnt/nas1/hf/deepseek_r1_4layers/"
    model_type: "deepseek3"
```
Fourth part describe the test cases, including prefill/decode, batch_size and input_len(they will be used as Cartesian product).Specially, tp_size and dp_size len should be equal as each tuple of them will be started as a parallel config. For example, in below config, script will start three server with `TP=1 DP=32`, `TP=2 DP=16`, `TP=4 DP=8` and benchmark
```yaml
    # test config
    is_decode: true
    batch_size_list: "[1,2,4,8,16,32,48,64,80]"
    input_len_list: "[4096]"
    tp_size: [1,2,4]
    dp_size: [32,16,8]
```
bazel_build_args is the flag for bazelisk build, if you want to test in AMD card, change `--config=rocm`
```yaml
    # build config
    bazel_build_args: '" --jobs 100 --verbose_failures --config=cuda12_6 "'
    # file dir config
    ft_sub_dir: "rtp_llm_perf_test"
```
last part is the model env config, be careful that all env configs type should in `[int, float, string, bool]`, or there maybe unexpected error
```yaml
    # model config
    start_port: 12333
    concurrency_limit: 80
    accl_dispatch_num_warp_groups: 4
    accl_combine_num_warp_groups: 4
    decode_test_length: 2048
    warm_up: 1
    act_type: "bf16"
    weight_type: "fp16"
    reserver_runtime_mem_mb: 0
    device_reserve_memory_bytes: 0
    load_ckpt_num_process: 96
    max_context_batch_size: 1
    enable_merge_w13: true
    use_deepep_moe: true
    enable_layer_micro_batch: 2
    enable_comm_overlap: true
    redundant_expert: 0
    accl_low_latency_optimize: 1
```
### Run step
```shell
# in root of RTP-LLM
cd rtp_llm/test/perf_test/multi_node
# Configure multi_benchmark_config.yaml
# Firstly run use -s to download wheel deps
/opt/conda310/bin/python3 multi_benchmark.py -m run -s
# Second and after
/opt/conda310/bin/python3 multi_benchmark.py -m run
# Finally clean running dir in machine
/opt/conda310/bin/python3 multi_benchmark.py -m clean
```
Also, multi-node benchmark dumps profile json in rank0, but currently we don't scp dir to local yet. So please go to rank0 and get profile data manually before clean step

## Result Format
Decode result, where batch size stands for per DP Rank
``` shell
+---------------------------------------------------------------------------------------------+
|                                        Decode Result                                        |
+---------+------------+------------------+--------------+------------------+-----------------+
| Seq Len | Batch Size | Sucess/Total Req | Input/Output | Waiting Time(ms) | Decode Time(ms) |
+---------+------------+------------------+--------------+------------------+-----------------+
| 4096    | 1          | 1/1              | 4096/2048    | 0.00             | 1.75            |
| 4096    | 2          | 2/2              | 4096/2048    | 0.00             | 1.72            |
| 4096    | 4          | 4/4              | 4096/2048    | 0.00             | 1.91            |
| 4096    | 8          | 8/8              | 4096/2048    | 0.00             | 1.93            |
| 4096    | 16         | 16/16            | 4096/2048    | 0.00             | 2.00            |
| 4096    | 32         | 32/32            | 4096/2048    | 0.00             | 2.19            |
| 4096    | 48         | 48/48            | 4096/2048    | 0.00             | 2.43            |
| 4096    | 64         | 64/64            | 4096/2048    | 0.00             | 2.60            |
| 4096    | 80         | 80/80            | 4096/2048    | 0.00             | 2.87            |
+---------+------------+------------------+--------------+------------------+-----------------+
```
