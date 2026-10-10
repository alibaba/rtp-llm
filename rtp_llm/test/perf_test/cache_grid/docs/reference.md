# Cache Grid 格式与执行参考

常用命令和产物索引见 [README](../README.md)。本页集中说明输入契约和执行细节。

## Grid 格式

最小单请求配置：

```json
{
  "schema_version": 2,
  "generator": {"cache_alignment": 4096},
  "cases": [
    {"case_id": 0, "batch_size": 1, "input_len": 8192, "cache_len": 0},
    {"case_id": 1, "batch_size": 1, "input_len": 8192, "cache_len": 4096}
  ]
}
```

- `cases` 必须非空。`case_id` 为非负整数、全文件唯一；重复几何也会被拒绝。
- 单请求的 `case_id/batch_size/input_len/cache_len` 均必填，`batch_size` 必须为 1。
- `0 <= cache_len < input_len`，cold 请求也必须显式写 `cache_len: 0`。
- 必填整数字段不接受数字字符串、浮点数、布尔值或自动补齐。
- 缓存对齐只在 `generator.cache_alignment` 配置，必须为正整数。
- 采样在独立生成器中完成，runner 只消费显式 cases。
- `kind/description/generator` 中的来源信息及 `summary` 用于记录和检查，不能代替实际 cases。
  手动修改 case 后应同步更新 summary。

输入长度包含缓存前缀。cache-hit case 应按实际复用块对齐，并留出
`cache_grid.commit_tail_tokens` 供 seed 提交，例如 `cache_len + 4096 <= input_len`。
输入采样对齐与缓存复用粒度是不同维度；kernel block 也不能直接当成可见复用粒度。
启动时会探测实际 reuse 粒度，失败则拒绝测量。
单请求 case 若落入相同物理缓存桶，runner 会保留一个代表；被去重的 ID 不能用于 retest/profile。

## Batch 调度

batch>1 必须显式写分组和前缀策略：

```json
{
  "schema_version": 2,
  "generator": {"cache_alignment": 4096},
  "cases": [{
    "case_id": 0,
    "batch_size": 4,
    "prefix_policy": "independent",
    "request_groups": [
      {"count": 2, "input_len": 8192, "cache_len": 4096},
      {"count": 1, "input_len": 8192, "cache_len": 0},
      {"count": 1, "input_len": 16384, "cache_len": 12288}
    ]
  }]
}
```

每组的 `count/input_len/cache_len` 都必填，count 为正，各组 count 之和等于 batch_size。
分组 case 不必填写顶层 input_len/cache_len；如果提供，必须与各组最大值一致。
分组形式也支持 batch=1。

| prefix_policy | seed 行为 |
|---|---|
| `independent` | 每个请求使用独立前缀 |
| `shared_by_group` | 同组共享前缀，每个共享前缀只 seed 一次；不同组独立 |

每个请求、每轮测量使用不同后缀，避免前一轮后缀缓存污染。cold 请求无需 seed。
流程为：将专用 BatchDecodeScheduler 设为 seed 数量 S → 并发 seed 并等待完成 →
切换为正式 batch B → barrier 放行 B 个请求 → 等整轮完成 → 下一轮。case 结束恢复 batch=1。
S=0 时跳过 seed。HTTP 每个槽使用独立 session，input_ids/gRPC 也支持并发。

该机制要求 DP=1、独占服务、prefill-only（partial=2、decode=1）。
实际组批由运行时调度器控制；`engine.max_context_batch_size` 和 `engine.concurrency_limit`
始终沿用 profile，启动器不会自动提高或强制设成 1。请按 workload 配置足够的准入并发。
请求失败可能导致 batch 等不齐，因此即使关闭 fail_fast，也会停止后续测量并恢复调度器。
可通过引擎 `BatchDecodeScheduler::schedule` 日志和 forward trace 验证实际组批。

`--skip-reuse-validation` 允许测试保留实测 reuse 不匹配的结果，仍记录期望值及校验状态；
输入/输出形状、请求成功和有效时间仍严格检查，启动前缓存块探针也不跳过。
这些 reuse 不匹配的结果不会进入统一 pipeline 的正式图表或公式。

## 容量与随机采样

随机生成器按剩余总 token 预算采样，允许长短请求混合，不把预算均分给 batch。
设对齐后的最小长度为 m、原始输入总预算为 T，第 i 个请求上限为
`min(max_input_tokens, T - 已采样总量 - 剩余请求数 × m)`；生成后打乱槽位。
命中请求至少留出 commit tail；cold 概率默认 0.2，实际比例受预算影响。

可选 `generator.workspace_tokens` 对 packed token 总量施加固定约束：

```text
sum(input_len) <= max_batch_tokens
sum(align_up(input_len + 1, cache_alignment)) <= workspace_tokens
```

`+1` 为输出预留。正式 batch、并发 seed、缓存探针都需通过预算校验。
不使用 `batch × 最长请求` 矩形预算，也不绑定 CP8 或固定 1M 上限。
启用 CP 时，缓存块须是执行对齐 `2 × TP` 的倍数；CP 禁用时执行对齐为 1。

| 模式 | 启动容量行为 |
|---|---|
| 有 workspace | `max_seq_len=floor(workspace_tokens/cache_alignment)×cache_alignment`；`max_batch_tokens_size=min(workspace_tokens,max_batch_tokens)` |
| 无 workspace、batch>1 | 必要时提高 `max_batch_tokens_size`，容纳最大一批输入总量 |
| 所有模式 | 不改写 profile 的 `max_context_batch_size` 和 `concurrency_limit` |

超出固定预算时拒绝，不自动拆 batch 或扩容。`--kv-budget-tokens` 仅估算活跃请求和 seed 尾部，
不包含旧轮缓存、权重、模型专用 state pool、通信或 JIT；token 预算不保证总显存不 OOM。
旧 workspace policy preset 已移除，需重新生成 grid；预算改变后不能接续原 checkpoint。

## 分析契约

原始结果要求 `schema_version=2`、`mode=prefix_cache_grid` 和 `metrics` 数组。
单请求保存 `runs[]`；分组保存 `request_groups`、`runs[].requests[]` 和调度信息。
正常 metric 明确记录 batch_size、input_len、status、measure_runs、success_runs；
单请求还包含 cache_len_requested。执行异常 `status=error` 可以没有测量轮次。

正式观测要求所有轮次成功，每个请求 input/output 形状正确、output_len=1，
reuse 与请求 cache 精确一致，服务端耗时为有限正数。
时间字段 `prefill_time_ms` 来自服务端 `aux_info.first_token_cost_time`，单位 ms，含引擎等待，
不扣减 wait_time，不含输入分词及客户端通信。HTTP 与 input_ids/gRPC 使用同一口径。
缺少服务端数据不能用 `ttft_ms/client_wall_time_ms/batch_wall_time_ms` 补齐。

每轮取批内请求服务端耗时最大值，跨轮用 median（默认）、min 或 trimmed 聚合。
统一绘图和拟合直接读取原始结果；`report/observations.json` 是审计输出，不是手工准备的前置文件。
CSV 仅用于输出或独立导出，不作为拟合输入。

restricted-symbolic 从受限候选库选择公式，使用相对平方误差，需要 CPU PyTorch。
保留请求列表；`sum(computeTokens²)` 是逐请求平方后求和，不是总 compute 平方。
候选还包括 batchSize、maxComputeTokens 等批量特征。训练按 batch 等权、同 batch 内几何等权，
跳过线性相关候选项；同一分布及其排列、跨文件重复观测进入同一数据分区。
训练/验证/测试使用确定性的 70%/15%/15% 哈希桶，测试集不参与选式或系数重拟合。

每个 batch 必须有训练、验证、测试样本并通过误差门禁；不完整来源不能通过生产验收。
`--symbolic-max-terms`、`--symbolic-complexity-tolerance-pct`、hinge/exp 参数控制搜索。
不再提供 quadratic 算法、`--model-family`、`--objective` 或旧划分参数。
异常分析检查缓存/计算单调性和拟合残差；无法形成训练与验证集时，报告 residual_check 跳过原因。
导出遵循 FlexLB PrefillTimeFormula 语法；上线仍需目标 Java 运行时与适用范围验证。

多运行合并先检查模型、拓扑、精度、缓存和引擎配置一致性，容量差异保留在来源审计中。
每个来源必须有 profile 快照或内嵌 profile，不能猜测不同模型可混合。

```text
单卡输入 TPM = sum(input_len) × 60000 / RT(ms) / cards
单卡计算 TPM = sum(input_len - cache_len) × 60000 / RT(ms) / cards
```

卡数优先 `engine.world_size`，否则 `tp_size × dp_size × pp_size`（缺省维度为 1）；
EP/CP 不额外相乘，pipeline 可用 `--cards` 覆盖。这是服务端口径的 token 速率，不是客户端端到端吞吐。

## Nsight Systems

命令见 [README 的 trace 部分](../README.md#重测与-trace)。在模型服务所在容器及用户环境执行。
启动器自动设置 Bazel `--run_under`，通过 `nsys launch` 采集 CUDA/NVTX/OS runtime、
进程树 CPU 采样和上下文切换；会覆盖其他 run-under 设置，不需要额外改 `.bazelrc`。
`--dry-run` 可查看完整命令；`nsys status --environment` 可检查容器采样权限。

默认生成唯一 session；`--nsys-session` 可显式指定，不得复用活动 session。
安装版本必须支持 `launch/start/stop`，控制命令使用相同 nsys 可执行文件。
每轮顺序为 `seed → nsys start → target request → tail → nsys stop`。
不会同时调用 Kineto `/start_profile`，不启用服务 GEN_TIMELINE_SYNC 或逐层同步。
采集覆盖整个请求和 host 处理；`--nsys-tail-seconds` 默认 0.1、上限 60，
只是返回后的等待窗口，不是跨 rank GPU 同步屏障。

结果写入 `cache_perf_replays/profile_<id>/nsys/*.nsys-rep`；manifest 记录 session、
请求结果、采集范围、报告路径、控制命令输出和退出码。stop/export 完成后才执行下一轮 seed 或退出服务。
报告缺失/为空、控制命令失败都会使补采失败；目标请求异常时仍尝试 stop，并单独记录清理异常。
`--trace-timeout`（默认 180 秒）同时约束控制命令和导出；导出超时后先检查 session 状态。

首次采一轮，确认所有预期 GPU worker 都有 CUDA 活动和 CPU 调度信息；文件存在并不等于覆盖全部 rank。
PyTorch record_function 不自动转为 NVTX，kernel 名称和 launch correlation 仍可查看。
省略 backend 或选择 kineto 则输出各 rank Chrome JSON。补采数据只用于诊断。

## 快照与恢复

统一入口保存 v2 `cache_perf_launch.json`、v4 `test_info.json` 及 profile/grid 快照。
启动清单记录有效参数、环境、Bazel 配置和 SHA256；test_info 保存状态与引用，不重复整份配置。
恢复校验快照、grid/profile/run_config 指纹及测量参数，不读取后来改动的源 profile。
底层直接 Bazel 测试的 v3 test_info 是其当前元数据，不作为统一入口的隐式恢复配置。

## 底层共享 seed 与物化输入

底层 Bazel runner 的 `--materialize_cache_cases` 可预生成文本/token IDs，
`--cache_case_files` 使用对应物化目录；grid 指纹、几何及轮次必须匹配。
物化用例的 store_info/record 必须使用当前 schema，seed 明确写成文本或 marker/filler。

`--cache_shared_seed` 是另一种实验：只支持未分组 batch=1、dashsc_input_ids、明确缓存块，
不支持 HTTP prompt、物化输入、固定 workspace 或 profiler。它与 case 内 shared_by_group 不同。
先 seed `最大 cache_len + commit tail`，命中 case 按 cache 长度降序运行，cold 最后运行。
每轮仍使用独立后缀并精确校验 reuse，测量“共享长前缀已驻留”的请求。

结果标记 seed_mode=shared_prefix_v1，保存 shared_seed ID、耗时和刷新记录，不能混用独立 seed 的目录。
重启续测会重新 seed。实际命中不足时，用当前 cache_len+tail 刷新并以新后缀重试一次；
失效请求不进入正式统计，刷新耗时计入 case elapsed_s。重试仍失败则停止，不无限重试。
