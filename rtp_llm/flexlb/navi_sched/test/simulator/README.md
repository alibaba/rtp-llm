# NaviSched 模拟集群包

这个归档用于搭建无 GPU 的 NaviSched 模拟集群，独立于正式 NaviSched C++ 包和 Python
wheel。正式包不包含模拟执行代码和角色启动器：正式 Agent、Client 作为 Python/C++ 库嵌入
模型框架并接受插桩；Master 始终由 Python `navi_sched.master_process` 入口启动。

## 启动三个角色

```bash
tar -xzf navi_sched_simulate-0.1.0.tar.gz
cd navi_sched_simulate-0.1.0
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p runtime/agent runtime/master runtime/client
bin/navi_sched_simulate_agent \
  --biz agent_biz \
  --grpc-port 19001 \
  --server-port 19001 \
  --runtime-dir "$PWD/runtime/agent"

cat > runtime/master/config.json <<'EOF'
{
  "local_nodes": [
    {"biz_name": "agent_biz", "ip": "127.0.0.1", "grpc_port": 19001,
     "support_heartbeat": true}
  ]
}
EOF
(
  cd runtime/master
  export NAVI_SCHED_MASTER_BIZ_NAME=master_biz
  export NAVI_SCHED_MASTER_CONFIG_FILE=config.json
  export NAVI_SCHED_MASTER_GRPC_PORT=19101
  export NAVI_SCHED_MASTER_LOG_FILE=navi_sched.log
  export NAVI_SCHED_MASTER_SNAPSHOT_DIR=navi_sched
  ../../bin/navi_sched_simulate_master
)

bin/navi_sched_simulate_client \
  --biz client_biz \
  --master-biz master_biz \
  --node master_biz,127.0.0.1,19101 \
  --runtime-dir "$PWD/runtime/client"
```

## 回放线上流量

Client 可通过 `--traffic-log` 启动 C++ 回放图。运行时支持单个 JSON 对象、JSON 数组和 JSONL；
时间字段按 `timestamp_ms`、`timestamp`、`reqTime` 的顺序识别，输入长度按 `input_length`、
`usage.inputTokens`、`prompt_token_count` 的顺序识别，hash 数组按 `input_block_hash_ids`、
`block_cache_keys`、`blockHashes` 的顺序识别。hash 元素可为十六进制字符串、非负 `int64` 或
`uint64`，进入 NaviSched 后保持为 `uint64`；运行时不 tokenize，也不重新计算 hash。

真实 enriched JSONL 应先转换为紧凑回放文件，避免每次启动扫描无关字段。转换器输入
`timestamp`、`input_length`、`input_block_hash_ids`、`service_names`，按一个 service 过滤并输出
`timestamp_ms`、`request_id`、`service_name`、`input_length`、`input_block_hash_ids`：

```bash
bazelisk build --config=clang //aios/navi_sched:traffic_replay_log_converter
./bazel-bin/aios/navi_sched/traffic_replay_log_converter \
  /home/zq103303/deepseek-pro-v4.2026-08-04.h09.enriched.jsonl \
  runtime/traffic_replay.jsonl \
  your_service_name
```

当前模拟 block size 固定为 256。`--traffic-service-name` 指定本次运行的唯一 service；同一回放
运行只处理该 service 的请求。

```bash
bin/navi_sched_simulate_client \
  --biz client_biz \
  --master-biz master_biz \
  --node master_biz,127.0.0.1,19101 \
  --traffic-log /home/zq103303/llm.log \
  --traffic-biz agent_biz \
  --traffic-client-count 8 \
  --time-compression 1.0 \
  --runtime-dir "$PWD/runtime/client"
```

`TrafficReplayR` 在启动时稳定排序请求时间，输入顺序相同的时间戳保持原始日志顺序，并轮转分配给逻辑
Client。`TrafficReplayKernel` 是唯一的计时器和提交者：它以一个 Navi periodic async pipe 控制
到点发送，因此多 Client 不会竞争定时器而改变全局提交顺序。请求一旦提交到 Client -> Master ->
Agent 图即继续回放下一条，不等待模拟执行完成；结果 Kernel 仅用于确认全部 fork 图自然结束后关闭
回放图。`--time-compression=2` 表示以两倍速度回放。

`test/simulator/run_traffic_replay.sh` 的
`NAVI_SCHED_TRAFFIC_CACHE_CAPACITY_BYTES` 表示模拟集群的总 Cache 容量，脚本会按
Agent 数量均分后传给每个 Agent。默认总容量为 300 GiB；按默认 20 Agent 回放时，每个
Agent 得到 15 GiB。`NAVI_SCHED_TRAFFIC_SCHEDULER` 选择 Master 使用的正式 Scheduler，支持
`random`、`max_cache_hit` 和 `cost`。`NAVI_SCHED_TRAFFIC_SCHEDULER_CONFIG` 将 JSON 字符串通过
Master 的 `scheduler_config` 传给所选 Kernel。汇总文件会同时记录策略名、总容量、单 Agent
容量、单 block 字节数、每个 Agent 的请求和 batch 分布，以及逻辑排空时间。CostScheduler
运行还会按 Master 返回的 `schedule_id` 去重每轮 `schedule_loss`，并在
`cost_scheduler_loss` 中保存明细和分布；同一轮多个请求不会重复计数。
CostScheduler 的 `alpha_decay` 默认为 `1.0`，表示固定步长；`min_alpha` 默认为 `0.0`。例如
`{"alpha":128,"alpha_decay":0.5,"min_alpha":8,"max_loop_count":10}` 会依次使用
`128/64/32/16/8/8/8/8/8/8`。
CostScheduler 至少执行 2 轮，剩余调度 window 足够时继续，最多执行
`max_loop_count` 轮。动态预算由 Master 的实际 `max_wait_schedule_time_ms` 决定。

需要同时回放多个 biz 时，将 `NAVI_SCHED_TRAFFIC_CASES_FILE` 指向一个 JSON 数组。每个元素定义
一个目标 biz 的输入流量、Agent 池和 Client 数量：

```json
[
  {
    "traffic_log": "/data/short-prompt.jsonl",
    "service_name": "short-prompt-service",
    "biz_name": "short_prefill",
    "agent_count": 20,
    "client_count": 2
  },
  {
    "traffic_log": "/data/long-prompt.jsonl",
    "service_name": "long-prompt-service",
    "biz_name": "long_prefill",
    "agent_count": 40,
    "client_count": 2
  }
]
```

脚本只启动一个 Master，但为每个 biz 创建独立 Agent 池，并让该 biz 的 Client 使用自己的真实
请求时间线。各 biz 的启动屏障彼此独立，不同请求数不会互相阻塞。全局汇总仍写入
`traffic_replay_summary.json`；其中 `biz_metrics` 按 biz 给出请求数、Agent/Client 数、延迟、
利用率和 CostScheduler loss，顶层字段给出所有 biz 的整体指标。不设置该变量时继续使用
`NAVI_SCHED_TRAFFIC_LOG`、`NAVI_SCHED_TRAFFIC_SERVICE_NAME`、
`NAVI_SCHED_TRAFFIC_AGENT_COUNT` 和 `NAVI_SCHED_TRAFFIC_CLIENT_COUNT` 的单 biz 模式。

要验证共享池，可把 cases 文件改为 `mode: mixed` 的对象。脚本按原始时间戳合并所有来源，保留
每条请求的 `traffic_source`，再让它们访问同一个目标 biz 和 Agent 池：

```json
{
  "mode": "mixed",
  "biz_name": "shared_prefill",
  "agent_count": 60,
  "client_count": 8,
  "sources": [
    {
      "name": "short",
      "traffic_log": "/data/short-prompt.jsonl",
      "service_name": "short-prompt-service"
    },
    {
      "name": "long",
      "traffic_log": "/data/long-prompt.jsonl",
      "service_name": "long-prompt-service"
    }
  ]
}
```

独立池和共享池对比必须使用相同的来源文件、时间线、请求总数、Scheduler 配置和单 Agent Cache
容量。汇总中的 `traffic_mode` 标识运行方式；混合模式在
`biz_metrics[0].traffic_sources` 中按来源给出 prompt、排队、执行和端到端延迟分布，用于检查
长请求是否拖慢短请求。顶层 Cache 命中率和利用率用于判断共享后的 Cache 与集群容量收益。

例如使用固定 2 轮和自动并行度运行 CostScheduler：

```bash
NAVI_SCHED_TRAFFIC_SCHEDULER=cost \
NAVI_SCHED_TRAFFIC_SCHEDULER_CONFIG=\
'{"lambda":0.5,"alpha":512.0,"max_loop_count":2,"parallelism":0}' \
  ./aios/navi_sched/test/simulator/run_traffic_replay.sh
```

回放可用 `--time-compression` 缩短实际运行时间，但汇总中的所有延迟字段始终是未压缩的
真实毫秒值，能直接和线上延迟比较。`traffic_replay_summary.json` 以
`latency_time_basis: "uncompressed_ms"` 标明此口径，并单独记录本次运行的压缩倍率；该倍率
会用于还原模拟等待时间。操作系统毫秒定时器、kernel 再调度和 batch 切换的固定墙钟开销无法
同步压缩，还原时会随倍率放大，因此队列精度结论必须以 1x 为准，压缩运行只用于快速回归。

项目维护的正式矩阵同时覆盖单 biz 和五个来源混合流量。对比运行必须完成全部请求、触发 LRU，
并使用相同的输入、Agent 数、Client 数、Cache 容量和随机种子。

墙钟通信、线程竞争和调度批边界不参与时间压缩；它们会形成随压缩倍率放大的累计非执行间隙。
因此压缩运行适合快速策略回归，绝对排队精度仍应使用 1x 运行确认。

正式结论使用 `TIME_COMPRESSION=1` 的完整矩阵生成：

```bash
./aios/navi_sched/scripts/run_cost_scheduler_simulation_suite.sh
./aios/navi_sched/scripts/run_replay_viewer.sh
```

脚本覆盖 Scheduler 基线、调度窗口、Cost loop 上限、lambda、独立池与共享池，以及共享池容量。
每个 case 的原始结果位于
`aios/navi_sched/tmp/cost_scheduler_suite/<case>/traffic_replay_summary.json`，汇总页面位于
`http://127.0.0.1:18780`。中断后可加 `--resume`；脚本只复用源码指纹一致，并且配置、请求数、
Agent 数和时间口径都通过校验的完整结果。正式矩阵把请求 deadline 和自然排空上限设为 2 小时，
任何图超时都会使 case 失败而不是生成不完整报告。查询跨 case 的具体数值时以结构化报告为准。

节点利用率是单节点所有 batch 的执行延迟之和，除以该节点首批开始至末批完成的逻辑墙钟跨度；
时间压缩只改变运行墙钟，两者均还原为未压缩时间。整体利用率是所有 Agent 利用率的算术平均。

真实运行在全局请求序号 10,000 和 30,000 处收集的两个 replay 样本，均可在不加载 Agent
Cache 的离线进程中使用正式 `max_cache_hit` 策略精确复现原决策。1000 Agent 并发完成上报的
三轮测试中，每轮 1000 条均到达 Master，传输延迟均值为 4.788、4.746 和 3.442 ms，p99 为
12.707、11.781 和 9.630 ms。

汇总中的 `latency_model_fit` 记录每个 Agent 以及全局按执行 batch 数加权的模型误差：
`batch_count` 是完成执行 batch 样本数，`prediction_bias_ms` 是 `predicted - actual` 的有符号
均值，`prediction_mae_ms` 是预测绝对误差均值，
`prediction_rmse_ms` 是均方根误差，`prediction_mean_relative_absolute_error` 是相对绝对误差均值。
模拟执行真值、Agent 预测和训练均以一个执行 batch 聚合为一个样本；白噪声会反映在这些误差中。
这些值用于当前回放配置下的回归观察，不是正式延迟承诺。

设置 `NAVI_SCHED_TRAFFIC_REPLAY_SAMPLE_INTERVAL` 后，脚本按固定请求间隔收集真实调度 replay，
并在每个 Agent 停止时输出 `traffic_queue_trace.csv`。汇总中的
`queue_prediction_accuracy.schedule_to_batch_start` 比较 Master 调度时保存的队列预测与 Agent
实际开始执行时间，才是队列预测精度；数值按 `time_compression` 还原为未压缩时间。
`execution_work_to_batch_start` 只比较队列预测与等待区间内前序 batch 的实际执行时间，
`ready_execution_work_to_batch_start` 进一步只保留调度时已经 ready 的前序 batch；
`future_ready_execution_work_to_batch_start` 统计调度后才 ready、但实际排在当前请求前面的执行
工作，用于识别 Master 当时不可能看到的新增负载。
`non_execution_gap_to_batch_start` 统计前序执行时间之外的墙钟间隙。下一批在前一批逻辑
deadline 前已经 ready 时，不存在请求传输或凑批等待；这部分 deadline 到实际唤醒的间隙记录在
`timer_wakeup_delay_to_batch_start`。若下一批稍后才 ready，deadline 到 ready 的部分记录在
`batch_availability_gap_to_batch_start`；剩余部分记录在 `unexplained_gap_to_batch_start`。
`batch_timer_wakeup_delay` 和 `batch_availability_gap` 直接给出相邻 batch 的单次分布。
`mean_underprediction_decomposition_ms` 按 `actual - predicted` 口径把总低估拆成执行模型、timer
wakeup、batch 可用性和未解释四部分，并提供恒等式最大误差。

同一段 60 秒流量的 30 Agent、20 Client 实测中，1x 的单批连续 timer overshoot 平均为
0.898 ms，采样请求累计 93.665 ms，非执行间隙为 93.667 ms，未解释残差为 0.002 ms；4x
还原后的单批值为 3.538 ms，采样请求累计 340.754 ms，非执行间隙为 340.760 ms，残差为
0.005 ms。说明这项系统性误差来自无法随回放倍率压缩的毫秒 timer 唤醒开销，而不是 Master
队列衰减或 Agent 执行模型。simulate 使用公式真值训练模型以验证拟合能力，因此不把这个运行时
伪影加入正式 Agent 预测；生产 Agent 的实际 `startBatch` 到 `finishBatch` wall latency 本身会
作为执行样本进入模型。

1x 无噪声对照还确认了另一个独立的冷启动误差：Master 的 queued/running 计数与 trace 重建
结果一致，调度后新增的前序执行工作为 0，但 ready-queue 仍平均低估 88.026 ms。Agent 在
`BATCH_READY` 时用当时的模型估计该 batch，并把结果加入 O(1) queue sum；后续 fit 发布新模型
时不会扫描和回算已经排队的 batch。高压冷启动会一次积压大量使用早期参数的估值，即使执行开始
时的最新模型已接近真值，旧估值仍保留到对应 batch 完成。该行为是“不回扫队列”的性能取舍，
不是漏报请求；使用启动历史拟合可以减小冷启动偏差。若未来要消除它，需要设计不扫描队列的模型
版本修正机制，不能把每次 fit 后全队列回算放进正式热路径。
压缩回放中 Agent 上报的是还原后的逻辑队列时间，而 Master 按正式系统语义用实际墙钟年龄衰减
该值；倍率大于 1 时会少减一部分逻辑进度，形成反向高估并部分抵消上述冷启动低估。这也是队列
准确性必须以 1x 为准的原因，不能用 4x 的较小 bias 判断模型更准确。
`queue_timing_consistency` 使用 Agent 返回字段和同一组 ready/start 时间戳复算，只检查序列化和
缩放链路，不是独立预测精度。绝对时间对齐仅用于所有角色运行在同一台机器上的 simulate，正式
系统不依赖 Master 与 Agent 时钟同步。

Agent 收到停止信号后不再接受新请求，但会继续串行排空已经接收的 batch。最后一个请求完成、
Cache 更新和延迟样本进入正式 Agent 汇报链路后，模拟执行图才输出 EOF；simulate stop 会等待
这个自然结束过程，不用固定超时截断已接收流量。

要从真实回放中收集调度上下文，可在拥有目标请求的 Client 上重复指定
`--traffic-collect-replay-sequence N`。该参数只对对应的单个请求设置正式
`collectReplay` 标志；Master 会为其所在的调度批构建一次上下文。轻量调度结果从 `decision`
输出立即进入 Client forward，完整快照随后从独立 `replay` 输出返回，不参与 Agent 凑批时序。
默认输出为运行目录中的 `traffic_schedule_replay.jsonl`，也可通过
`--traffic-replay-output` 指定。不要对每条请求开启：每个样本都包含该批全部候选 Agent 的
命中和负载信息，会明显增加调试链路的序列化和存储开销。

脚本模式下可将零起始的全局请求序号写入
`NAVI_SCHED_TRAFFIC_REPLAY_SAMPLE_SEQUENCES`，多个序号使用逗号分隔。只有命中序号的请求会
设置 `collectReplay`，但输出快照仍包含该请求所在调度轮次的全部请求和候选 Agent。

三个命令需要在不同进程中启动，可用 `SIGINT` 或 `SIGTERM` 停止。归档在 `python/` 目录中
附带匹配版本的正式 `navi_sched` wheel 内容。三个入口优先使用
`NAVI_SCHED_PYTHON_BIN` 指定的 Python 3.10；随后通过 pybind 加载包内 C++
`_navi_sched_core.so`。Python 只解析启动参数并持有角色生命周期。

`navi_sched_simulate_master` 执行正式 `navi_sched.master_process`，实际创建的是正式 C++
`NaviSchedMaster`。归档仅在自身内部用 simulation core 替换正式扩展，因此 Agent 可以启用
模拟执行 Kernel，并通过正式 Agent-Master 流上报 Cache、inflight 和延迟状态。

`bin/navi_sched_simulate_agent` 与 `bin/navi_sched_simulate_client` 只是为了便于直接启动测试
集群；它们不引入正式 Agent/Client 可执行程序，也不会再启动独立 C++ 子进程。Client 入口
创建启用 forward 的角色，测试负载所在进程再通过 NaviSched C++ 或 Python Client API 提交请求。

模拟角色只开放配置的 Gig gRPC 端口。RTP 导流模拟复用 RTP-LLM 自己的 Frontend 健康检查，
这个归档不模拟 Suez Admin HTTP-RPC 协议。

## 构建与验证

在仓库根目录执行：

```bash
./aios/navi_sched/scripts/build_simulate_package.sh
./aios/navi_sched/test/simulator/test_simulate_package.sh
```

归档输出到 `aios/navi_sched/release/`。`build_release.sh` 会自动构建该归档并运行解压后的
三角色回归测试。
