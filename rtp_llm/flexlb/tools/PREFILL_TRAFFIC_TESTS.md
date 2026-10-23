# Prefill 连续流量回归

这组测试用真实 FlexLB Master 和 Java Mock Engine 检查调度、排队、缓存归属与扩缩容行为。
场景参数维护在 [prefill_traffic_cases.json](prefill_traffic_cases.json)，执行入口为
[run_prefill_traffic_suite.py](run_prefill_traffic_suite.py)。修改场景后可以直接重跑同一套用例。

Mock 的基准为每个 Prefill worker 每秒处理 70,000 个 compute token，输入为 4K 或 64K token。
这套测试不执行模型或 GPU kernel；测得的 TPM、缓存命中率和 TTFT 用于验证所声明的模拟条件，不能代替真实模型压测或线上 SLO 验收。

## Master 固定速率性能回归

调度服务自身的毫秒级性能门槛使用独立入口，从 FlexLB 根目录执行：

```bash
bash tools/run_master_scale_regression.sh /path/to/new/master-scale-output
```

默认串行执行 3 轮，每轮 `BATCH/FIXED_WINDOW` 和 `NON_BATCH/SINGLE` 各 14 项，共 84 项。
每项使用原测试的真实 token payload、固定请求序列、2GiB JVM 堆和原有断言；
P1/D2、P4/D8、P16/D32 分别测试 1000/2000/5000/10000 QPS，另有 P2/D4、P8/D16 的 2000 QPS。
Master P99 必须小于 50ms，交付等待及 BATCH 的 batch wait P99 不超过 50ms，
客户端 P99 不超过 250ms，完成吞吐至少达到目标的 85%。
每项还验证成功、请求覆盖、路由和生命周期排空。BATCH 经真实 gRPC 到 Mock Engine，校验接收请求唯一、输入长度和路由一致；
NON_BATCH 只测 Schedule 返回路由，不执行后续推理 RPC，不能据此证明 Engine 执行性能。
使用 `FLEXLB_PERF_REPEATS=1` 可执行单轮；
减少轮次必须在报告中注明。脚本校验每轮实际执行的 14 项 XML 结果，保留全部失败，并检查测试期间源码是否发生变化。

此入口仅选择固定速率方法，**不执行不限速一次灌入 8192 请求的测试**；该场景暂缓，不算通过。
原 `api-performance-regression` profile 仍保留该测试及其他性能方法。
这些指标是 Schedule RPC 延迟和 QPS，不是模型 TTFT、TPM 或 KV 命中率；后者仍需运行下述连续流量矩阵。

运行期间停止其他压测及并行编译。脚本的互斥锁只阻止本入口同时运行，不约束其他命令。
诊断日志也可能干扰延迟：有 GC 日志/JFR 的定位运行应与无额外分析选项的验收分开保存，
不能把诊断失败删除、降低门槛或用后续成功覆盖。若 GC 总停顿远大于回收阶段之和，应继续核查日志 I/O、
JVM 和主机阻塞，不能直接归因于调度算法；异步 GC 日志同样需要检查缓冲区丢日志提示。

## 运行

需要 Java 21、Maven 构建出的 Master jar、与 online_eval runner 兼容的 Mock jar，以及安装该 runner Python 依赖的环境。
Master、Mock、runner 是三个独立输入，应固定各自版本。不要把不兼容 runner 的启动参数强行传给另一个版本的 Mock。
所有命令从 FlexLB 根目录执行，输出目录必须不存在，避免覆盖失败证据。
重复的 seed、dispatch 或性能模型组合会直接报错。子进程使用本轮保存的场景快照，运行中修改原 catalog 不会改变后续组合；失败处理也不能覆盖已有运行目录。

```bash
python tools/run_prefill_traffic_suite.py --help
python tools/run_prefill_traffic_suite.py --list

python tools/run_prefill_traffic_suite.py \
  --runner /path/to/compatible/online_eval \
  --jar /path/to/master.jar \
  --mock-jar /path/to/compatible-mock.jar \
  --out-dir /path/to/new/quick-output \
  --suite quick

python tools/run_prefill_traffic_suite.py \
  --runner /path/to/compatible/online_eval \
  --jar /path/to/master.jar \
  --mock-jar /path/to/compatible-mock.jar \
  --out-dir /path/to/new/acceptance-output \
  --suite acceptance
```

首次全场景检查可以加 `--seeds 1`，执行 48 个组合；完整验收使用默认三个 seed，或显式加 `--seeds 1 2 3`。
`--jobs 1` 顺序执行；资源充足时可用 `--jobs 2` 或 `--jobs 4` 并行执行独立组合。并发会增加本机资源压力，不能因此放宽发流有效性断言。
可先运行不启动 Java 服务的审计与输入生成回归：

```bash
python -m unittest discover -s tools -p 'test_prefill_*py'
```

`quick` 选择低负载全冷和混合扩缩容两个场景，执行两个 dispatch 模式、恒定和独立波动两种性能模型，使用 seed 1。
`acceptance` 使用 catalog 中的全部场景、性能模型和 seeds。每个组合启动独立环境，不继承上一个组合的缓存或 session 归属。
完整矩阵包含多次启动、预热、发流和排空；缩短运行时优先减少 seeds 或指定场景，不要删掉硬超时后的观察窗口。

可以单独复现一个组合：

```bash
python tools/run_prefill_traffic_suite.py \
  --runner /path/to/compatible/online_eval \
  --jar /path/to/master.jar \
  --mock-jar /path/to/compatible-mock.jar \
  --out-dir /path/to/new/single-output \
  --cases tools/prefill_traffic_cases.json \
  --suite acceptance \
  --case repeated-cold-bursts \
  --dispatchers BATCH \
  --jitters independent \
  --seeds 1
```

同一主机上的多个执行进程需要独立 Master HTTP、management、gRPC 和 Mock worker 端口范围。
端口或启动失败属于环境失败，不能记为调度通过。只清理本次测试拥有的进程和目录。

## 场景与参数

| 场景 | 关注点 |
|---|---|
| `low-load-cold` | 无 session、全冷请求，低负载下连续扩容和缩容，要求没有排队拒绝 |
| `mixed-elastic` | 冷暖混合、部分 session，扩容后缩至较小集群，再恢复容量 |
| `all-session-warm` | 所有请求带 session、可复用前缀，冷 worker 加入与持有缓存的 worker 退出 |
| `short-session-hotspot` | 短请求集中访问热门 session，检查热点和空闲容量同时存在时的行为 |
| `isolated-cold-burst` | 稳态背景流量上叠加一次冷请求突发，再恢复低负载 |
| `repeated-cold-bursts` | 周期性冷突发，持续时间跨越排队硬期限，观察是否积累拥塞 |
| `long-cold-ramp` | 长请求为主、全冷，流量连续上升至超出基准容量，再下降 |
| `overload-recovery` | 自动学习估算器、长短混合，缩容叠加持续过载，再恢复容量与低负载 |

具体 worker 数、负载、持续时间和断言以 JSON 为准。每个场景分别执行：

- `NON_BATCH`：`SINGLE` 决策，前端 `GenerateStreamCall` 交付。
- `BATCH`：`FIXED_WINDOW` 决策，最多 32 个请求、20ms 收集窗口，由 Master `EnqueueBatch`，前端 `FetchResponse`。

这两个组合覆盖不同的交付路径。BATCH 的多请求 batch 门槛按整个运行检查，包含正式发流前的 16 请求 `calibration`。
低 QPS 下，正式测量阶段可能全部执行单请求 batch；这证明 BATCH 交付路径在低流量下可用，不证明该阶段承受了多请求合批压力。
高负载或突发阶段还需单独统计正式阶段 Engine 事件中的多成员执行批次，不能用 calibration 的合批替代这一证据。
`maxInflightPerPrefillWorker` 在 NON_BATCH 中计请求、在 BATCH 中计 batch；同名数值不代表两种模式具有相同的请求并发上限。

主要参数语义：

- `load` 相对于初始三个 worker 的基准计算能力，不会随着扩缩容自动改变发流速率。实际压力还受缓存未命中、worker 波动和热点影响。
- `cold_fraction` 是使用新前缀的概率，`long_fraction` 是 64K 输入的比例；其余输入为 4K。
  这些是输入分布参数，不能当作实测缓存命中率。暖请求也可能因迁移、worker 退出而冷算。
- `session_fraction` 控制可复用会话中携带 session ID 的比例；`session_for_cold` 控制冷请求是否携带独立 session ID。
  最终带 session 的请求比例以实际发出的记录为准。
- `hot_fraction` 控制对热门会话的集中程度；结合冷请求比例解释热点，不能把整个集群的平均空闲等同于热点 worker 空闲。
- `arrival: poisson` 使用 seed 固定的随机到达间隔。突发是额外的冷请求；`burst_period_s` 使其在同一阶段重复。
- `ramp_end_load` 使同一阶段内的到达速率线性变化。阶段之间保持在途请求，不先排空队列，结束整个场景后才收集并排空。
- 扩缩容是受控操作：退出 discovery、等待 Master 观察到拓扑变化，再让对应 worker 优雅退出。它验证计划内缩容，不代表突然宕机无损。
  验收同时覆盖 Master 本地队列和 Engine 已接收请求。仅 Engine drain 成功不足以证明无损；本地队列在 worker 退出时被关闭并返回 8510 也会判失败。

## Worker 性能波动

三种性能模型共用相同的基准计算公式：

| 模型 | 计算吞吐系数 |
|---|---|
| `stable` | 恒定为 1 |
| `independent` | 每个 worker 按自身 seed 和时间 slot 独立变化 |
| `correlated` | 所有 worker 在同一时间 slot 使用相同系数 |

波动每秒更新一次，每五个 slot 将 `0.9、0.95、1、1.05、1.1` 做一次确定性打乱。
因此吞吐目标为 63K–77K compute token/s；对应执行时间按系数的倒数变化，不能称为执行时间也恰好上下 10%。
已开始执行的 batch 保持启动时的参数。新 worker 加入后到首次控制更新之间使用默认系数 1.0，不能把这段时间算作已注入波动；此后参与更新。
Mock 将执行时间按 `max(1, int(compute_tokens / 70 / factor))` 取整数毫秒。
因此，特别短的请求用 token 数除以执行时间反算的吞吐可能略出 63K–77K；审计核对整数执行公式，不能要求每个短请求的实测 TPS 严格落在 ±10% 内。
同一 seed 的流量输入和性能序列可复现；操作系统调度、RPC 时序和实际 batch 分组仍可能不同。

性能控制写入实际 Mock 执行公式，并读回确认。验收还从 `prefill_done` 的整批 compute token 和执行时间反算实际系数；只看到 HTTP 设置成功不足以证明注入生效。
性能更新边界的请求单独标记为时序歧义，仍需落在允许的离散系数内。波动覆盖不足、控制线程落后或事件无法解释都应使测试失败。
Mock 的 sleep 模型简化了真实 GPU 并发、通信和 batch 效率，±10% 的结论仅针对这套模型。

## 结果审查

先检查输入和事件证据是否成立，再解释性能。每次运行保留配置、输入参数、客户端逐请求记录、worker 事件和结果文件；审查失败时保留原目录。
至少核对以下事项：

1. 请求 ID 唯一，所有发出的请求均有终态；成功请求与 Engine 完成事件一一对应，输入 token 数一致，没有重复执行或孤立成功。
   实际执行地址必须与调度结果一致，Prefill 时间必须落在客户端请求至首响应区间内。
   Mock 的 `evict_chain` 是不带 request ID 的缓存生命周期事件，按固定 schema、Engine 身份、时间范围和释放链单独审计并计数；未知事件仍失败。
2. 收到成功调度后发生的 Engine 或传输失败单独算失败。允许排队拒绝的阶段也只能接受有证据支持的拒绝类型；错误码相同不代表原因相同。
3. 计划到达时间与实际发起时间的偏差没有超过有效性门槛。Schedule 直接异步提交，等待排队结果不占用发流线程；实际 RPC 发出和收到调度结果后的 stream 创建偏差也分别检查。发流器自身积压导致的降载不能解释为服务能力提高。
4. 拓扑确实达到目标 worker 数，缩容 worker 的已接收工作完成排空；实际 Engine 证据确认 dispatch 模式、batch 大小和性能波动。
5. 全冷低负载的强制零拒绝断言成立。其他场景即使允许过载拒绝，也必须展示拒绝数量、比例和恢复过程，不把“错误分类符合预期”写成“服务无损”。

每个阶段的 `min_success_fraction` 是执行前固定的最低完成比例，过载阶段也不能只完成一个请求就通过。
它用于发现明显退化，不是线上 SLO；阶段是否达标必须和实际比例一起报告。
末阶段还要求经过 `settle_seconds` 后发出的请求全部成功，检验积压是否在低负载下消退。
恢复初期仍可能受上一阶段在途工作影响，不能仅用全阶段平均值代替这个检查。
当前场景没有预设 TPM 的绝对下界或 TTFT 的 SLO 上界。`PASS` 表示证据、发流有效性和该场景声明的行为门槛通过，
不表示 TPM 或 TTFT 已达到线上 SLO；这些指标仍需结合具体部署目标逐项判断。

指标需要使用一致的请求集合和时间窗口：

- **Input TPM** 统计成功请求的输入 token；**compute TPM** 统计这些请求的 `input_tokens - cache_hit_tokens`。
  包含排空时间的 cohort 吞吐与固定墙钟窗口吞吐有不同含义，应分别标注。跨阶段在途请求不能重复计入。
- **KV cache 命中率** 按成功请求的 token 加权。拒绝掉大量冷请求会提高这个比值，所以必须与成功率、拒绝率一起看。
- **TTFT** 从客户端实际发起开始，到首个有效响应；发流偏差独立报告。P50/P95/P99 使用成功请求，并同时给出样本数。
  小样本 P99 主要反映最慢的几个样本，不能据此声称线上尾延迟稳定。
  兼容 Mock 的两种路径均发送 first-token 和 terminal 两帧；此处测的是客户端首个响应时间，不是请求完成时间。
  两种模式的调度、RPC 和取流路径不同，不能将延迟差值全部归因于调度，也不能当作真实模型首 token 的延迟。
- **吞吐负载** 要结合实际 worker 数、已验证性能系数与真实 compute token 计算。名义 `load=0.8` 不证明每个 worker 都处于 80% 利用率。

比较稳定性能和波动性能时，使用相同 case、dispatch、seed 的输入；比较 NON_BATCH 和 BATCH 时同时查看实际 batch 分布。
逐个 seed 展示结果，再汇总差值和范围。不要只取最好的一次，不要用多次运行的平均值掩盖单次失败，也不要在看到结果后降低门槛。

## 覆盖边界

本矩阵覆盖计划内扩缩容、连续积压、热点、长短请求、缓存复用机会、session 比例，以及独立和同时发生的轻度性能变化。
它没有证明真实 GPU 吞吐、KV 显存不足和驱逐、网络分区、突然宕机、Master 切换、同地址 worker 重启、客户端取消或超时竞态均正常。
这些故障需要相应协议回归或 online_eval 故障用例提供独立证据；不能用正常路径的吞吐结果替代。

[run_prefill_elastic.py](run_prefill_elastic.py) 提供更大规模的快速离散模拟，适合筛查规模和策略组合。
它不执行完整 Master/Engine 协议，也不能替代上述 NON_BATCH/BATCH 实际交付测试。
所有运行结论和测量表放在运行产物中；本 README 只维护可复用的操作方式与解释规则。

## 汇总多个运行目录

[summarize_prefill_traffic.py](summarize_prefill_traffic.py) 只读取已有证据，不重跑或放宽门槛：

```bash
python tools/summarize_prefill_traffic.py \
  /path/to/matrix-seed1 /path/to/matrix-seed2 \
  --out-dir /path/to/new/summary-output
```

输出 `summary.md`、`summary.json`、`phases.csv` 和 `comparisons.csv`。
完整矩阵读取 `results.json`；正在运行的矩阵读取 `results-slot-*.json` 并明确标为 `INCOMPLETE`。
失败运行保留原始失败状态和原因，缺少官方报告时从原始事件计算的指标标为 `RAW_UNVALIDATED`，缺少唯一 Engine 证据的 KV/compute 指标留空。
报告生成命令成功仅说明文件已写出，不代表流量测试通过。

每个 seed 单独展示阶段数据。仅在同一矩阵、同一 case/dispatch/seed 的 stable 对照具有相同完整 trace，且逐请求实发字段、代码版本与配置匹配时计算差值。
失败对照仍明确标为未经验证的描述性比较；缺失对照或输入不完整时不计算差值。
这种配对不能证明统计显著性，也不能把过载下接受请求集合不同造成的命中率差异解释为单一调度机制的收益。

## 缩容、自动合批与突发保护回归

修复这些问题后必须重跑完整 acceptance 矩阵，保持原 catalog、seed、Mock 和 runner 不变，只替换 Master 制品。
记录两版 jar SHA256；逐组合核对完整 trace、配置和 worker 波动序列。原失败运行仍参与对比，不能只挑两版都通过的组合。

- **计划缩容**：BATCH 本地尚未交付的请求应释放旧路由预留，回到全局队列重新选择 worker，保留原请求 ID、future 和截止时间。
  8510 不属于正常过载拒绝；成功请求必须恰好执行一次。8431 仍需要队列过载或原截止时间的证据。
  代码回归 `QueuedDeliveryWithdrawalRaceTest` 覆盖重复退休通知、已有 placement claim、取消以及已提交交付不能重放；
  `WorkerBatcherQueueTest` / `WorkerBatcherSchedulingTest` 检查正常退休与意外关闭的区别。
- **自动学习合批**：`overload-recovery` 的每个 BATCH/seed/jitter 组合都要统计正式发流阶段的多成员 batch 数、最大 batch 与 singleton 数。
  `PrefillAdaptiveBudgetTest` 覆盖预测固定开销超过自动服务窗的情况；token 收集预算按增量计算成本求解，完整延迟预测仍用于排队估算。
  calibration 合批只能证明交付路径，不能代替正式阶段合批证据。NON_BATCH 指交付路径，Mock 内部仍可能把请求合批。
- **突发背景流量**：`isolated-cold-burst` 和 `repeated-cold-bursts` 分开统计额外注入请求与同期背景请求的成功率、失败码、拒绝前等待和成功请求 P99。
  自动模式下，未达到硬过载水位的年轻请求保留 soft wait 预算；年龄从原请求创建时起算，重排队不续期。
  `PrefillQueueSheddingTest` 检查年轻请求保护不会转而淘汰更高优先级/更有缓存收益的请求，并检查硬过载与显式手动恢复配置仍生效。

对比报告应同时给出成功数/总数、固定窗口完成 TPM、含排空时间的成功 cohort TPM、成功请求 token 加权 KV hit、TTFT P99，以及恢复早期和硬期限后的尾段损失。
cohort TPM 的分子只计成功请求 token，分母包含该到达阶段所有请求达到终态所需的排空时间，也包括最后才超时的失败请求。
固定窗口完成 TPM 按 Engine 完成时间归窗，可能计入前一阶段遗留的请求；两种口径需要同时查看。
成功率提高可能伴随更长等待或更多冷算，需保留这些代价。至少按场景、dispatch 和 jitter 展示结果，不能只给全矩阵平均值。
排队截止时间不等于端到端 TTFT 上界：截止前已交付的请求还要执行 Prefill，成功请求的 TTFT 可能超过排队硬期限。
