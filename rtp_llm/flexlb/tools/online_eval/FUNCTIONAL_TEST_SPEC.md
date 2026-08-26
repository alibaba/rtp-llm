# FlexLB Mock 框架非压测测试需求规格 v2

> 适用：flexlb_functional_tests.py 统一框架（实施中）。本文档定义 v2 用例结构——基于 v1 盘点的四点结构修订：
> ① anomaly 与 cancel 去重并重定义（取消失败模式）；② behavior 组重组；③ disconnect 与 kill 合并为参数化引擎故障组 + 新增断连抖动；④ 补空白用例 + 新增 master HA 三段式场景。

## 1. 设计原则

- **单一职责分组**：一个用例只验证一个维度；同一故障的不同注入方式（HTTP 停启 vs 进程 kill）是同一用例的参数，不是两组用例。
- **anomaly = 异常路径**：正常取消路径归 cancel 组；anomaly 只测"取消失败/请求未送达/流错误"等异常语义。
- **断言内聚**：恢复率/清理时间等被动行为断言并入对应故障用例，不单独立场景（避免 behavior 式大杂烩）。

## 2. 套件结构 v2

### 2.1 smoke · 功能正确性（24 用例）

| 子组 | 数量 | 变化 |
|---|---|---|
| cancel（T1-T6） | 6 | 不变 |
| scheduling（S1-S12） | 12 | 不变（S4/S9 断言按 Java 真实拒绝语义） |
| anomaly | 6 | **重构**：删 E1（与 cancel 重复）；原 E2/E3 更名；新增 4 个 |

**anomaly 重构明细**（重定义：取消失败 / 未送达 / 流错误）：

| ID | 场景 | 来源 |
|---|---|---|
| smoke_anomaly_prefill_no_respond | 全 prefill no_respond → 请求超时 → clear 恢复 + inflight 清零 | 原 E2 |
| smoke_anomaly_prefill_enqueue_error | 全 prefill enqueue_error → 请求报错 → 恢复 | 原 E3 |
| smoke_anomaly_decode_no_respond | **decode 侧** no_respond → 输出流悬挂/超时 → 恢复（补 decode 注入空白） | 新增 |
| smoke_anomaly_decode_enqueue_error | **decode 侧** enqueue_error → startDecode 降级路径（prefill 本地完成）→ 断言请求仍成功完成（背压降级语义） | 新增 |
| smoke_anomaly_cancel_failed | **取消失败模式**：目标引擎 no_respond 下发 cancel → cancel 无响应 → 请求超时兜底 → 恢复后状态一致、无泄漏（用户定义的 anomaly 核心） | 新增 |
| smoke_anomaly_fetch_error | fetch_error 注入 → client fetchResponse 流收到错误 → 恢复（补 fetch_error e2e 空白） | 新增 |

（generate_error master 侧归 chaos 吗？不——它也是异常路径，若实现时用例数膨胀可将其并 入本组第 7 条 smoke_anomaly_generate_error，优先级中。）

### 2.2 chaos · 混沌与恢复（参数化引擎故障 + 抖动 + master + 弹性）

#### A. engine_down 统一组（disconnect + kill 合并，参数化）

用例：`chaos_engine_down__{down_mode}__{target}__{topology}`
- down_mode：`http_stop`（/stop_engine→/start_engine）| `process_kill`（kill -9 victim 进程→重启）
- target：`prefill` | `decode`
- topology：`multi`（2P+2D）| `single`（1P+1D）

矩阵 8 变体（v1 的 disconnect 2 + kill 4 合并，补齐 http_stop×single 2 个）。**统一断言集**（吸收原 behavior S2/S4）：
1. 故障期 master HTTP 200 不崩、请求持续可发（multi）或优雅降级（single）
2. 存活引擎接管（multi：accepted 持续增长）
3. 恢复后目标引擎 inflight=0（排空等待）
4. 恢复成功率 ≥95%
5. 无异常 cancelled
6. **cleanup_time < TTL**（原 S4 calibrate 快清断言并入）
7. **channel 恢复**（原 S2 断言并入：宕机期 accepted 不变，重启后增长）

#### B. flap 断连抖动组（新增，补空白）

| ID | 场景 | 断言 |
|---|---|---|
| chaos_flap_short | 短时停启循环 N≥5 轮（每轮停 500ms~2s，**短于 3 连败窗口**） | master 不误摘（endpoint 保持）· 请求成功率不低于阈值 · 无泄漏 |
| chaos_flap_mixed | 混合长短间隔（短于/长于 3 连败窗口交替，N≥6 轮） | 长窗口被摘除后短窗口快速恢复 · 路由候选集最终一致 · 3 连败计数不导致永久摘除 |

#### C. inflight 清理专项（原 behavior S1 保留独立——它验证的是"客户端死亡 + 引擎停止"的复合清理）

| ID | 场景 | 断言 |
|---|---|---|
| chaos_inflight_ttl_cleanup | stop 引擎 + kill 客户端 → stuck inflight | 90s 内归零（计时对比 TTL 30s） |

#### D. master 满载行为（原 behavior S3 保留独立——测的是 master 配额满时对新请求的行为，非引擎故障）

| ID | 场景 | 断言 |
|---|---|---|
| chaos_master_quota_block | 1P+1D，4 批填满 MAX_INFLIGHT_BATCHES → stop 引擎 → 20 新请求 | 阻塞期失败率 ≥50%（错误类型分布）· 恢复后 ≥90% |

#### E. master 进程组（不变）

| ID | 场景 | 断言 |
|---|---|---|
| chaos_master_kill | kill -9 master（稳态/decode 中两时机） | 重启后 inflight 全 0 · 恢复 ≥95% · TTFT 退化 ≤50% · batchId 无冲突 |
| chaos_master_recovery | master 宕机 → client fallback 直连 → 重启 | fallback 成功率 · TTFT 恢复时间线 |

#### F. master_ha 三段式（新增，需求详见 §3）

| ID | 场景 | 断言 |
|---|---|---|
| chaos_master_failover | 双 master：kill 主 → 备升主 | 备接管服务 · client 无需重启继续成功 · 切换时间记录 |
| chaos_master_fallback_return | 主 kill → client fallback 直连 → **新主恢复 → client 回切 master** | 三段各段成功率 · 回切时间（依赖 §3 调查结论，功能缺口则标注） |

#### G. elastic 弹性组（6，已另行规格，此处引用）

chaos_elastic_add_flow / remove_flow / add_remove_cycle / rebalance / stop_after_add / concurrent_ops

## 3. master HA / frontend-engine 切换（DEFERRED · 整块延后）

> **2026-08-26 决定**：fallback 的真实语义是"master（含 slave）连接级全死时，frontend 直接把请求发给 engine，engine 以单请求模式接收并自行组批"——不是 client 层 per-request 失败重发。该方案涉及 frontend/master/engine 切换链路，**改动大，待其余全部完成后再考虑**。本节保留调查结论（已核实，见下），全部用例与功能缺口标记 deferred，不进实施队列。

### 3.1 已核实的真实 fallback 语义（RTP 仓库代码级）

| 事实 | 证据 |
|---|---|
| frontend 每请求路由：master 队列超限 → 主动拒绝（不 fallback）；正常 → Schedule gRPC | backend_rpc_server_visitor.py:332-360 |
| **fallback 触发 = master+slave 连接级全失败**（业务错误 code≠200 直接报错不 fallback）；探活全挂则直接走 domain 路由 | master_client.py:85-109,295-316；host_service.py:333-350 |
| frontend 后台 1s 探活 /rtp_llm/master/info（连续 2 败标记 UNHEALTHY） | host_service.py:246-251,412-455 |
| fallback 动作：VIPServer 域名解析 engine 地址 → GenerateStreamCall 单请求直发 engine | backend_rpc_server_visitor.py:306-330；model_rpc_client.py:448-493 |
| engine 双入口：EnqueueBatch（master 组批）+ GenerateStreamCall（单请求）；单请求进引擎由 FIFOScheduler 与其它并发单请求**合并组批** | model_rpc_service.proto:707,729-731；LocalRpcServer.cc:153-195；FIFOScheduler.h:73-84 |
| engine RPC 失败先回 master 重路由（≤3 次）再 fallback | backend_rpc_server_visitor.py:518-566 |
| 真实引擎 e2e 已有：M7 基线 / M8 kill master fallback / M10 恢复（flexlb_domain_fallback_smoke.py） | internal_source smoke |

### 3.2 mock 框架现状与真实语义的差距（deferred 后的备忘）

1. JavaLoadClient 触发面偏宽：schedule_error（业务错误）与 fetch 失败也触发 fallback——真实仅连接级失败才切（压测 fallback 比率会高估）
2. mock 的 generateStreamCall 恒 batch=1，**缺 engine 单请求侧自组批**（真实引擎 FIFOScheduler 会合并并发单请求）——fallback 期间吞吐被系统性低估
3. mock client 替代 frontend 做路由决策：结构上合理（框架无独立 frontend 进程），per-request 粒度与真实一致
4. 主备：选主/备转发完整，但状态交接缺失、client 跨地址回切缺失、双 master 需双 IP 环境 + ZK

### 3.3 deferred 用例与缺口（不实施，仅记录）

- chaos_master_failover / standby_forward / fallback_return / ha_full 四用例 → deferred
- 功能缺口：client(frontend) 跨地址回切、fallback 熔断、主备状态交接、框架 ZK+双 master 编排、mock 单请求自组批 → 全部随"master/frontend 切换方案"一起定案

## 4. 验收标准

1. 全部用例经 `flexlb_functional_tests.py --suite all` 可执行，统一 JSON 结果（suite/name/mode/status/duration_ms/detail）
2. 任一 FAIL 总 exit=1；`--filter`/`--mode`/`--list` 生效
3. 每用例结束无 java 进程残留（pgrep 校验）
4. 本文档每个用例 ID 在框架中可寻址（命名一致）

## 5. 实施跟踪

| 项 | 状态 |
|---|---|
| 主框架移植（cancel/scheduling/anomaly v1/behavior/disconnect/kill/master 旧映射） | Hank 进行中（按本 v2 修订执行） |
| anomaly 重构 + 空白补充（decode 注入/cancel_failed/fetch_error） | 排队 |
| engine_down 合并 + flap 组 | 排队 |
| elastic 组 6 用例 | 排队（规格已发） |
| master_ha 组 | **DEFERRED**（§3：待 master/frontend 切换方案定案） |
