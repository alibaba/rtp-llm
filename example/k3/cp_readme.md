# K3 CP Smoke 使用说明

这份文档说明 K3 当前 CP 代码应该如何启动和验证。基础功能继续使用原来的 PD smoke；新增的缓存压力和性能入口是独立工具，不会改变基础 smoke 的请求集合。

## 1. 先理解几个开关

Prefill 和 Decode 的 cache 是否按 CP 分片，由两个环境变量控制：

```bash
PREFILL_CP_KV_CACHE_SHARDED=0|1
DECODE_CP_KV_CACHE_SHARDED=0|1
```

四种布局对应关系如下：

| 布局 | `PREFILL_CP_KV_CACHE_SHARDED` | `DECODE_CP_KV_CACHE_SHARDED` |
|---|---:|---:|
| P1D1 | 0 | 0 |
| P8D1 | 1 | 0 |
| P1D8 | 0 | 1 |
| P8D8 | 1 | 1 |

建议先使用 `SP_TYPE=mtp` 跑完精度和压力，再测试 Eagle3。Prefill CP + Decode KTP8 通过下面的拓扑开关选择：

```bash
KIMI_K3_DECODE_TOPOLOGY=dp8_ktp8_ep8
```

KTP8 不等于 Decode CP；它是 Decode 的 attention/KDA 拓扑配置，cache 分片仍由 `DECODE_CP_KV_CACHE_SHARDED` 单独控制。

## 2. 基础双机 PD smoke

基础 smoke 已经存在，入口是：

```text
example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py
```

它会在两台机器上同时拉起 Prefill 和 Decode，自动申请远程端口，并在 Prefill 完成后通知 Decode。不要手动固定一组端口，也不要先启动 Decode、等待完成后再启动 Prefill；使用 `--parallel-start` 让两端并行加载。

下面是命令模板，路径、机器、容器和 checkpoint 按实际环境替换：

```bash
export PREFILL_CP_KV_CACHE_SHARDED=1
export DECODE_CP_KV_CACHE_SHARDED=1
export SP_TYPE=mtp
export KIMI_K3_DECODE_TOPOLOGY=legacy

/opt/conda310/bin/python -u \
  /path/to/runtime/runfiles/rtp_llm/example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py \
  --prefill-ssh-target <prefill-host> \
  --decode-ssh-target <decode-host> \
  --prefill-repo-root /path/to/runtime/runfiles/rtp_llm \
  --decode-repo-root /path/to/runtime/runfiles/rtp_llm \
  --prefill-checkpoint-path <prefill-checkpoint> \
  --decode-checkpoint-path <decode-checkpoint> \
  --prefill-sp-checkpoint-path <prefill-mtp-checkpoint> \
  --decode-sp-checkpoint-path <decode-mtp-checkpoint> \
  --prefill-endpoint <prefill-host> \
  --decode-endpoint <decode-host> \
  --prefill-container <prefill-container> \
  --decode-container <decode-container> \
  --container-user <uid>:<gid> \
  --suite all \
  --parallel-start \
  --remote-detached \
  --remote-control-root /path/to/remote/control \
  --artifact-root /path/to/controller/artifacts \
  --run-id <unique-run-id>
```

`all` 中已经包含基础 cold/hit/partial/mixed、多请求复用、MTP、chunk prefill、多模态和长历史请求。基础功能不需要再通过新脚本重复执行。

切换四种布局时，只改变两个 CP 环境变量：

```bash
# P1D1
PREFILL_CP_KV_CACHE_SHARDED=0 DECODE_CP_KV_CACHE_SHARDED=0 ...

# P8D1
PREFILL_CP_KV_CACHE_SHARDED=1 DECODE_CP_KV_CACHE_SHARDED=0 ...

# P1D8
PREFILL_CP_KV_CACHE_SHARDED=0 DECODE_CP_KV_CACHE_SHARDED=1 ...

# P8D8
PREFILL_CP_KV_CACHE_SHARDED=1 DECODE_CP_KV_CACHE_SHARDED=1 ...
```

每次运行都保留完整命令、环境、服务日志和 `cache/startup.json`。pool inventory 只说明各 group 的容量和 backing bytes，不等同于某个请求实时占用了多少 block。

## 3. 960K 缓存压力

基础 `all` 不变。缓存压力使用独立入口：

```text
example/k3/kimi_k3_cache_pressure.py
```

它会：

1. 构造多个目标约 960000 tokens 的独立前缀（服务返回的实际长度为准）；
2. 先分别填充这些前缀；
3. 多轮回访相同前缀；
4. 每次检查 PD 是否生效、输入长度是否正确、答案是否正确；当前入口仅串行填充与回访，不保证四个前缀足以淘汰整个 host cache；
5. 保存每次请求的 reuse、耗时、原始请求和响应。

该入口要求已经有一个可用的 PD Prefill endpoint：

```bash
/opt/conda310/bin/python -u \
  /path/to/runtime/runfiles/rtp_llm/example/k3/kimi_k3_cache_pressure.py \
  --base-url http://<prefill-host>:<prefill-port> \
  --output /path/to/evidence/cache-pressure \
  --namespace k3-cp-p8d8-mtp-pressure \
  --checkpoint /path/to/kimi-k3 \
  --prefix-count 4 \
  --rounds 3 \
  --target-tokens 960000 \
  --timeout 1800
```

960K 是默认压力长度，可以通过 `--target-tokens` 改为 600K 等实际可运行的长点，并记录实际输入长度。当前不测试 1M+1 等边界，也不在精度阶段扫描 BS；较长点的失败不覆盖已经完成的较短点结果。

资源不足可以是压力测试中的合法结果，但必须在之后发送一个短请求确认服务恢复。CUDA OOM、未知 HTTP 5xx、半截输出或服务进程退出都算失败。

## 4. 单角色性能 smoke

性能不启动完整双机 PD，使用：

```text
example/k3/kimi_k3_perf.py
```

这个入口复用 RTP-LLM 现有的 `EngineServer` 和 `BatchPerfImpl`，因此仍然使用正常的模型启动、scheduler 和 cache 路径。

### Prefill

Prefill 使用 `--mode prefill`，默认输入长度是 960000 tokens。采用普通 Prefill/FIFO 路径，工作集按 B1、DP1 顺序访问；先完整填充、再完整回访一轮作为预热，然后计时回访 `--rounds` 轮，每批内部不重复预热。

```bash
PYTHONNOUSERSITE=1 /path/to/runtime/bin/kimi_k3_perf \
  --mode prefill \
  --input-len 960000 \
  --batch-size 1 \
  --workset-size 1 \
  --rounds 3 \
  --result-dir /path/to/evidence/prefill-shared \
  --model_type kimi_k3 \
  --checkpoint_path /path/to/kimi-k3 \
  --tokenizer_path /path/to/kimi-k3 \
  --tp_size 8
```

Prefill 至少跑两种工作集：

- `workset-size` 较小，TP 和 CP 都可以重复命中；
- `workset-size` 较大，使用实际 pool 容量推导，观察 CP 和 TP 的 reuse 差异。

两种配置使用同样的请求长度和访问顺序。结果写入 `k3_perf.json` 的每轮 `metrics`：

- `avg_ttft_ms`：包含引擎排队和缓存加载的首 token 时间，单位 ms；
- `avg_wait_time`：引擎报告的等待时间，单位 ms；
- `avg_prefill_time`：扣除上述等待后的时间，单位 ms；
- `avg_reuse_len`：总复用 token 数。单角色响应使用通用 reuse 字段，不能将默认值为零的 `prefill_*` 字段当作真实的分层命中统计。

CPU/GPU 分层复用需结合原始响应的 `local_reuse_len`、`memory_reuse_len` 等字段确认。CP Prefill 的收益由工作集留存与 TTFT 实测说明，不预设冷请求必然更快。

### Decode

Decode 使用 `--mode decode`，固定生成 512 tokens：

```bash
PYTHONNOUSERSITE=1 /path/to/runtime/bin/kimi_k3_perf \
  --mode decode \
  --input-len 960000 \
  --batch-size <Bcommon> \
  --workset-size <Bcommon> \
  --rounds 3 \
  --decode-test-length 512 \
  --result-dir /path/to/evidence/decode-p8d8 \
  --model_type kimi_k3 \
  --checkpoint_path /path/to/kimi-k3 \
  --tokenizer_path /path/to/kimi-k3 \
  --tp_size 8
```

`Bcommon` 不通过服务端反复试出来，而是根据 TP/CP 两侧实际 pool geometry 分别计算可承载 BS 后取：

```text
Bcommon = min(B_tp, B_cp)
```

单角色 Decode 使用 PDFUSION 初始化缓存，CP 对照需同时传入 `--prefill_cp_kv_cache_sharded 1 --decode_cp_kv_cache_sharded 1`，TP 对照两者均为 `0`。真实双机 PD 的两个角色仍分别设置自己的开关。

同一个 `Bcommon` 用于 CP 和非 CP 的公平 TPOT 对比。另外分别运行 TP 和 CP 自己的最大候选 BS 点；候选按实际 pool 几何估算，成功运行才是该点的实测证据，不做逐整数搜索。

perf 框架已经根据 `cost_time`、`first_token_cost_time`、`wait_time` 和 `output_len` 计算 decode time/token。MTP 不能按 SSE 帧数当 token 数；结果中应同时保留 `iter_count` 和 speculative accepted/proposed 统计。

性能入口为投机输出缓冲额外预留 `2 × gen_num_per_cycle + 1` 个 token，避免 `max_seq_len=输入+输出` 导致零输出空间。两种模式均默认 `KIMI_K3_PREFILL_CHUNK_TOKENS=65536`，保留显式环境配置：单角色 Decode 的 PDFUSION 初始化也会预留 Prefill GEMM workspace，按单轮 chunk 限制它可避免随历史 KV 长度分配巨大临时空间，不会缩短请求的 KV。示例的模型参数仅展示调用结构，正式运行仍须核对 FP8、MTP checkpoint、chunk、Graph 和 cache 策略。

## 5. Profile

性能统计运行关闭 profile。每个代表性配置单独增加 `--profile`：

```bash
--profile --profile-trace-name k3-p8d8-960k
```

入口会复用 RTP-LLM 现有的 `profile_step`、`profile_trace_name`、`GEN_TIMELINE_SYNC` 和 `TORCH_CUDA_PROFILER_DIR`，结果目录中应出现对应 timeline/Torch profile 文件。profile 在全部正式回访结束后单独执行，不混入 TTFT/TPOT 统计，也不提前刷新工作集。timeline 是按需分析依据，不是每次排查的强制门槛。

## 6. 如何判断结果

- 基础功能和 960K 压力：看输出是否正确、服务是否持续可用、合法资源不足后短请求能否恢复。
- Prefill：看 TTFT 和 reuse，尤其是共同工作集与较大工作集之间的差异。
- Decode：看 CP/TP 的 `B_tp`、`B_cp`，以及共同 `Bcommon` 下的 TPOT。
- pool inventory：只用来解释容量，不把启动时总 block 数误认为请求实时 batch。
- FP8、MTP、chunk、Graph 和 CP 布局必须记录在结果的环境快照中。

旧的固定 pressure、1M±1 边界和手工固定端口命令已经不再是当前入口。提交代码或切换布局后，先检查 `git status --short`、服务日志和实际 endpoint，再开始 smoke。
