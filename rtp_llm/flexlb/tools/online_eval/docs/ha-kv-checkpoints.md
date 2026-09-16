# HA 切换与 KV 淘汰检查

这两类用例仍由 Python 编排；YAML 只指定 profile、环境规模和参数。
每个 `case::variant::profile` 是一个独立执行实例，独占自己的 Master、Mock 引擎和端口。

## ABA 切换：client_fallback_failback::wraparound

A/B 使用同一组 2P/4D Mock 引擎。持续流量先走 A，A 被杀后切到 B；
A 重启后，原客户端保持 B；B 被杀后再切回 A。

| 阶段 | 检查 |
|---|---|
| A 正常服务 | 建立成功率、P/D 分布、TTFT p50/p95 与各账本峰值基线 |
| 杀 A 后 | 立即检查切换请求，包括故障前已发出但尚未完成的请求；检查重试和 B 路由 |
| B 稳定服务 | 检查 B 路由、成功率、Prefill 分布与账本上界 |
| 重启 A | 检查拓扑恢复、20 个显式 A 探针（并发 10）的成功率与 Prefill 分布，再检查 A 本地持有量归零 |
| A/B 共存 | 检查原流量仍走 B，同时记录相对基线的延迟和分布变化 |
| 杀 B 后 | 立即检查切回 A 的请求与重试，再检查 A 稳态 |
| 流量结束 | 对账全部 issued、实时 terminal 和最终结果，检查全程成功率至少 95%；最后检查 A 的 inflight 清零 |

`checkpoint_window_s` 默认 5 秒，允许 3～10 秒；每窗口至少 30 个请求。
稳态成功率至少 95%，目标 Master 路由占比为 100%，两台 Prefill 最大份额不超过 75%。
故障过渡窗口会包含在故障边界完成的旧路由请求，因此成功率下界为 90%、新目标占比下界为 50%，
并要求至少一次 failover。过渡窗口记录分布，由紧随其后的稳态窗口判定是否存在持续单边调度。
TTFT 是对照证据，不使用未经校准的延迟门限。

`max_owner_load` 默认 128，是该规模下显式配置的积压上界。
Scheduler 请求数、Prefill batch 数和 Decode load 分别判断；它们不是同一个计数单位，
也不直接等于客户端并发数。Decode load 含共享引擎上报负载，因此 A 重启后即使没有客户端流量，
也可能观察到 B 带来的引擎负载。此时只要求 A 自己的 Scheduler、Prefill ownership、
Decode reservation/dispatch permit 为零。全部流量结束后才要求完整 inflight 清零。

实时检查使用可选 `LIVE_CLIENT_EVENTS=true`。Java 客户端逐条刷新 `client_lifecycle.jsonl`，
包括 issued/terminal、序号、发出时间和完成观测时间。Python 增量读取，检测缺失、重复、
序号断裂和半条记录；未完成请求不能因缺少最终结果而从样本中消失。
已有最终 `client_events.jsonl` 格式保持不变。其他用例默认不启用实时日志。

## 冷缓存淘汰：cache_churn::lru_affinity

Prefill 总容量为 4 块，Mock 保留 1 块作为 reserve，因此单次请求最多申请 3 块。
旧 pressure 一次请求 5 个键，超过总容量，本身不可准入，无法验证 LRU。

新流程先写入并回放 `[810001, 810002, 810003]`，确认命中同一 Prefill、无淘汰、
`referenced_blocks=held_blocks=0` 且缓存中有 3 个键。
随后发送 `[810001, 810004, 810005]`：单次 3 块可以准入，前后键集合有 5 个，
超过 4 块容量；共享首键提供同机亲和。检查请求成功、保持亲和、缓存不超容量且淘汰数增加。

## 在途引用保护：cache_churn::referenced_occupancy

这是同一个 Python 程序的新 variant，扩展四个 profile，共增加 4 个实例。
预热 3 个键后，让同一 Prefill 上的请求运行 15 秒，观测其 3 块 KV 被在途请求引用。
再发一个新键请求，要求 3 秒内在另一台 Prefill 完成；原持有者的引用数仍为 3、淘汰增量为零。
最后等原请求成功完成，确认引用数归零。

冷缓存可以被淘汰；被在途请求引用的块不能被当成可回收空间。
因此不能用“缓存有键”直接推出“剩余可用容量不足”。两个 variant 分别覆盖这两种状态。

## 定向执行

在已分配的远端环境、已设置租约端口的情况下，从 `online_eval/` 运行：

```bash
python3 parallel_runner.py --parallel 3 --profile batch-window \
  --instances 'client_fallback_failback::wraparound::batch-window,cache_churn::lru_affinity::batch-window,cache_churn::referenced_occupancy::batch-window' \
  --out-dir /path/to/new-output
```

`parallel_runner.py` 每次选择一个 profile。其余三个 profile 分别运行；同时运行时端口区间必须分离。
每个检查点的 JSON 原始快照、请求明细和最终对账随实例输出保留。
