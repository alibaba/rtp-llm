# Schema 3 配置与 case 对齐（2026-09-09）

状态：测试侧修复完成；BATCH 摘机快速失败仍有运行时缺口，保持 FAIL。本文是定向回归报告，不是 392 实例全量通过报告。

## 提交整理

- 原分支 `codex/ft-case-framework`：`6e7da12566`。
- 与 `feature/flexlb_config` 的交叉点：`d1a5b4f10665d82b1987280b852c1a2537a702cc`。
- 交叉点之后 5 个提交均为王子毅；压缩为一个提交，压缩前后树完全相同。
- 目标分支新增提交 `6b96f5e540` 为洛离，保持独立。压缩后的提交 rebase 到它之上。
- 使用 `codex/ft-config-alignment-20260909` 交付，保留旧分支及其未提交修改。没有强推。

## 配置变化

`FLEXLB_CONFIG.schemaVersion` 升到 3；scenario YAML 的 `schema_version: 2` 是另一套版本，保持不变。

| 旧配置 | 当前处理 |
| --- | --- |
| 两种 dispatcher inflight 上限 | 统一 `max_inflight_per_prefill_worker`；BATCH 计批次，NON_BATCH 计请求 |
| `scheduler.capacity` 的全局 outstanding / 等待位上限 | 产品删除，case 不再传这些字段；依赖其拒绝行为的探针退役 |
| `staleInflightTimeoutMs` | 显式请求预算改写为 `requestLifecycle.request.timeoutMs`，语义是匹配状态上报的非活动超时，不再宣称旧 fence TTL |
| delivered-not-accepted 配额、超时与 enqueue RPC 超时 | 产品删除，不保留静默兼容别名 |
| `engineCancellation.ackTimeoutMs` | 删除；完成等待预算使用 `preemption.timeoutMs` |
| `candidateChoice` / decode decay、outlier、输出截断估算 | 删除；采用上游确定性 Prefill 选择和 Decode 轮转、完整输入输出记账 |
| discovery ENV | 使用 `MODEL_SERVICE_CONFIG` 的 hosts / discovery_file |
| gRPC executor ENV | 压测生成器将其显式投影到 JSON 的 `grpcServer`，文件与进程读同一份配置 |

PRIORITY 省略 preemption 会启用默认策略，不能再解释成关闭抢占。Python 编译器的能力清单也按这个行为计算。

## 退役的实例契约

| 场景 / 变体 | 原因 |
| --- | --- |
| admission_queue / master_capacity（2 个 profile） | 全局 outstanding 准入上限已删除 |
| priority_preemption / disabled_zero_eviction | PRIORITY 的“省略即关闭”开关已删除 |
| priority_preemption / error_code_family | 其中全局容量 8431 分支已删除；其他错误族保留独立用例 |
| priority_preemption / prefill_queued_live_single、prefill_queued_live_window | 新源码 `WorkerBatcher` 对 BATCH 不做等待请求数量限制，旧等待位抢占舞台不成立；不能把旧 window PASS 当成新版本保障 |
| priority_preemption / decode_reserved_live_single | SINGLE 的本地窗口在固定间隔之前关闭，无法稳定构造 never_seen 的本地驱逐 |

Prefill 随机分布公平性检查保留取样，明确标为 SKIP 诊断项。没有放宽旧阈值。Decode 轮转分布仍使用 YAML 中已有的显式 band；完整性检查仍生效。

## decode_reserved_live_window

保持 placeholder drain 前移、确认物理池空闲。victim 改回 512/2，incoming 3500/2，D 池 5 块，window 400ms，提交间隔 0.15s。间隔包含 debug 取样耗时，不再在取样之后额外等待完整间隔。观察 victim 为 queued、无 dispatch permit、引擎未见过，才发 incoming；观测竞态不能判产品失败。

同窗本身不足以触发驱逐。保留原有、只改变上报可用量的 1000 token 压力控制，并检查它没有改变物理块数：

- Master 预算：`5120 × 90% = 4608`。
- 未驱逐：`1000 + (512+2) + (3500+2) = 5016 > 4608`，缺口 408。
- 驱逐 victim：`1000 + (3500+2) = 4502 <= 4608`。
- 引擎物理池：incoming 4 块，水位 1 块，5 块池可容纳。victim 在本地队列中，没有物理租约。

若去掉压力，即使同窗也只有 4016 token，不存在 Master 缺口；此时不驱逐不能证明 Master 缺陷。PR10 / PR5 / PR6 的裁决表达式及阈值不改。

## 弹性并发收敛

新增四个 convergence 变体，原四个变体保留原成功率断言。复用已有 Python tail 观测，未加入 Java 插桩或 HTTP 轮询。

- 从 resolved_config 派生 `statusStaleAfterMs + cleanupIntervalMs + margin`；超过 cap 直接报配置漂移。
- 成功 add 后从 discovery 文件读取真实 IP，按已知 HTTP→gRPC `+1` 协议转换；不从 engine 名字猜 IP。请求落点与 remove/add 事件按完整地址及发生顺序匹配，处理地址复用。
- 尾段延长到最后一次成功 remove 返回后的 bound，加 YAML 指定观测窗口；要求足够的后界样本，防止空样本通过。
- NON_BATCH 死引擎命中要求 2 秒内 UNAVAILABLE；BATCH Schedule 失败要求 5 秒内明确返回。BATCH 没有返回地址的失败也统计，并标注无法归因，30 秒 deadline 不会漏掉。
- 没有失败样本时 failfast 检查显示 SKIP，不冒充 finding resolved。
- 新变体显式 `unique_ports: true`：mock 的 gRPC 服务绑定 wildcard IP，复用端口会让旧 IP 的请求进入新引擎。客户端在已分配端口窗口内给每次 add 分配唯一端口，避免将此别名行为误判为摘机失败；原四变体不变。解析器仍按完整地址及时间处理重建代次。
- 尾窗结束时若尚未发够 YAML 声明的最少样本，继续采样，仍受总 stage deadline 限制；不会靠空样本或缩短 RPC 超时通过。

## 摘机窗口与 R1.6

摘机窗口文档提出的 SUSPECTED 路由过滤和继续探测属于产品策略，本次不实现。当前 `ExpirationCleaner` 仍按最后成功 poll 加 stale 阈值，在 cleanup 周期内摘除。测试量化其有界收敛及失败响应，保留真实产品失败。

R1.6 trailer 缺口属于 mock：仅为 generateStreamCall 的 P 容量 602、D 分配 8211 出口添加原始 ErrorDetailsPB 的 `grpc-status-details-bin`，保持 RESOURCE_EXHAUSTED 和原消息。Netty 传输测试校验两个 trailer；stopped UNAVAILABLE 不增加该 trailer。

rebase 中保留的只读 debug 方法同步到新字段和注册 API；未修改 Master 的调度、驱逐、取消逻辑。上游启动器设置 `IGNORE_GETENV_PROPERTY_NAME`，测试启动参数显式传 `--flexlb.debug.enabled=true`，不再依赖 Spring 读取环境变量。

API 单测中 `nonPreemptiveQueueWaitsForDecodeAtDelivery` 的 PRIORITY 两行与新默认抢占策略矛盾，限定到 FIFO 两种 dispatcher；没有更改调度逻辑。

## 额外发现的容量 case 编排问题

`pool_saturation_evict_reject_recover` 需要三个同时占用 8 块的请求，默认 inflight=2 只能占住 16 块。YAML 显式配置 inflight=4，恢复原有 24 块饱和前提，所有块数、错误和恢复断言不变。

两个容量变体在 NON_BATCH 引擎拒绝后仍可能保留 Master 记账，不能将物理引擎释放等同于 Master 已清空。YAML 明示 request inactivity=60000ms、最终清场预算=90s；Decode 拒绝后的恢复请求前增加 Master inflight 清空检查。清场失败仍然是 TIMEOUT，未改变产品释放逻辑。

## 验证记录

构建、单测和真实 case 运行产物位于：
`/Users/wangziyi/code/case-refactor-reports/2026-09-09/config-alignment/`。

### 单测与静态检查

- Java：共 1432 项（Mock 366；common/cache/grpc/sync/api 合计 1066），0 failure、0 error，API 保留原有 1 项 skip。
- Python：最终 848 项全部通过，耗时 403.862 秒；结果见 `python-v3-final.json`。
- scenario 编译：32 个逻辑场景、187 个变体、392 个实例、3976 项检查。
- 修改的 Python 已通过语法、Black 24.8.0、isort 5.13.2 检查；压测脚本通过 bash 语法检查；git diff 无空白错误。

### 真实 case 结果

运行在独立 MCP 租约中，最高 8 条 lane 并行。初次失败证据完整保留，没有覆盖成通过结果。

| 用例 | 修复后结果 | 证据 |
| --- | --- | --- |
| decode_reserved_live_window / BW | 两轮 PASS，PR10/PR5/PR6 原断言均成立 | cases-bw-r2、cases-bw-r3 |
| decode_pool_exhaustion_terminal / 四种 profile | 4/4 PASS | cases-*-r2 |
| pool_saturation_evict_reject_recover / 四种 profile | 4/4 PASS | BATCH cases-*-r2；NON_BATCH cases-*-r3 |
| single_nonbatch_convergence、window_nonbatch_convergence | 各两轮 PASS | cases-sn/wn-r3、r4 |
| batch_convergence、single_batch_convergence | 各两轮 FAIL，仅 failfast 不满足 | cases-bw/sb-r3、r4 |
| 原四个 elastic_concurrent_mutation | BATCH 两个 PASS，NON_BATCH 两个原成功率断言 FAIL（0.4 < 0.5） | cases-*-r1；按需求保留旧口径 |

四个新增变体在最终两轮均有足够后界样本，15 秒边界后 dead hits 均为 0。BATCH 仍有约 30 秒 Schedule DEADLINE_EXCEEDED，违反 5 秒快速失败要求，且无返回落点：只能确认快速失败缺口，不能仅凭此把失败归因到某台已摘除引擎。未改 Master 使其通过。

### R1.6 FR 状态

- FR-1 完成：只给两个容量出口添加 ErrorDetailsPB trailer；状态码、description 与拒绝逻辑不变。
- FR-2 完成：真实 Netty Java 回归验证 602/8211 以及逐字相同的 message；stopped 没有 trailer。
- FR-3 完成：四个 NON_BATCH 实例及四个 BATCH 对照通过；`r16-trailer-evidence.json` 中四份错误均为 parsed，代码分别为 8211、602、8211、602，message 含 LACK_MEM。
- FR-4 完成：原工作目录的 failure-todo.md R1.6 状态及 r16-requirements.md 实施注记引用本分支最终提交；历史需求文档不混入本次代码提交。

本轮没有运行全部 392 个真实实例；不能把定向用例和单测通过表述为全量矩阵通过。SUSPECTED 过滤、继续探测等产品策略不在本次变更内。
