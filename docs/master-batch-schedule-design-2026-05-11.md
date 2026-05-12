# Master `/batch_schedule` Protocol Redesign — Design Doc

**日期**：2026-05-11（v2 修订 2026-05-11，audit 后大改）
**作者**：weike.chw
**状态**：Phase 1 ready for implementation
**关联分支**：`feature/master-batch-schedule`（commit 4548c8923 已实现 RR batch 基础设施）
**对应 API 文档**：`docs/master-batch-schedule-api.md`

---

## 0. Phase 划分

| Phase | 范围 | 状态 |
|---|---|---|
| **Phase 1（本次 PR）**| RR batch 性能优化 + wire 协议加 `sub_requests` 字段口子（DTO 接收 + 长度校验，但策略层不读）| **本文重点** |
| **Phase 2（后续 PR）**| TTFT batch 真正落地：`ShortestTTFTStrategy implements BatchLoadBalancer`，复用单选逻辑 | 本文给出方案大纲，留作 phase 2 设计起点 |

**为什么这么拆**：
- RR batch 已落地，本次只需做"原子 op 合并"这一项性能优化（5 行核心改动）
- TTFT batch 是 caller 对工作量分布有诉求时才用得上，今天没 caller 强需求 → 延后
- wire 协议上预留 `sub_requests` 可选字段，phase 2 落地时无需 caller 配合升级（向前兼容）

## 1. 背景

flexlb master 已有 `/schedule` 接口（单请求多阶段路由），但对"caller 一次性需要 N 个 worker 派发独立工作"场景（典型：prompt_generator 一次散发 N 条独立推理），caller 当前只能并发 N 次 `/schedule`。痛点：
- master 内部 N 次 cursor / 锁串行 + N 倍 SSL/HTTP parsing CPU
- 缺少批量便利语义

`/batch_schedule` 设计为 `/schedule` 的批量便利版，**单 role 部署**专用（多 role 走 `/schedule` per request）。

> 注：从 wall-time 角度看，N 次并发 `/schedule` 已经能 multiplex，"省 N 次 RTT" 的 framing 不准；真正省的是 master 端 CPU 串行 + 网络层的 N 倍 parsing 开销。

## 2. Goals / Non-goals

### Phase 1 Goals
- **RR batch 路径性能优化**：N 次原子 op → 1 次原子 op，降低高 QPS 集群下的 cache line 弹射量
- **Wire 协议加 `sub_requests` 字段口子**：DTO 接收 + 长度一致性校验，策略层暂不读 → phase 2 启用时无 wire 改动

### Phase 2 Goals（非本次）
- TTFT batch 实现，让长短 prompt 在 batch 内自动散开
- 复用 `select()` 完整机制（`localTaskMap` / 心跳 reconcile / `ResourceMeasure` 过滤 / cache-aware / CAS fairness）

### Non-goals（永久）
- **不覆盖多阶段批量**（解耦 PD 集群每份请求要 PREFILL + DECODE 配对）。这类场景 caller 走 `/schedule` 并发即可
- **不做 batch atomic rollback**：fire-and-forget 派发；TTFT 模式 phase 2 落地时复用 `select()` 的 `localTaskMap` 自然有 rollback 接口（partial failure 处理见 §6）
- **不做 cache-aware batch 调度**：caller 场景 KV cache 复用率低，价值不显著

## 3. Scope

| 部署形态 | 是否支持 |
|---|---|
| 单 role（PDFUSION / PREFILL / DECODE / VIT 任一） | ✅ 支持 |
| 多 role（disaggregated PD / VL）| ❌ 拒绝，返回 INVALID_REQUEST，引导走 `/schedule` |
| 0 role（master 未就绪）| ❌ 返回 NO_AVAILABLE_WORKER |

## 4. 关键设计决策

### D1: Wire 协议形状

**Request — Level 0（最简，phase 1 RR 唯一形态）**:

```json
{"batch_count": 10}
```

**Request — Level 1+（phase 1 接收但忽略，phase 2 TTFT 启用）**:

```json
{
  "batch_count": 10,
  "sub_requests": [
    {"request_id": 1001, "seq_len": 1280},
    {"request_id": 1002, "seq_len": 980},
    ...
  ]
}
```

| 字段 | 必填 | 类型 | Phase 1 行为 | Phase 2 行为（TTFT）|
|---|---|---|---|---|
| `batch_count` | ✅ | int | 范围 `[1, BATCH_SCHEDULE_MAX_COUNT]`（默认 1000，env 可配）| 同 |
| `sub_requests` | ❌ | array | 接收，校验 length == batch_count（若提供），策略层不读 | TTFT 模式下必填，每元素必须有 `request_id` |

**`sub_requests[]` 单元素字段**（沿用 `/schedule` 的 `Request` DTO 形状）：

| 字段 | Phase 1 | Phase 2（TTFT）|
|---|---|---|
| `request_id` | 不读 | **必填**（写入 `localTaskMap` 用作 key，重复会被校验拒）|
| `seq_len` | 不读 | 可选；不传按 0 算，等价于"工作量未知" |
| `block_cache_keys` | 不读 | 不读（cache-aware batch 是更晚的演进）|
| `generate_timeout` | 不读 | 不读 |
| `request_time_ms` | 不读 | 不读 |

**Response（成功）**:

```json
{
  "success": true,
  "code": 200,
  "server_status": [
    {"server_ip": "10.1.2.2", "http_port": 28100, "grpc_port": 28101},
    ...
  ],
  "real_master_host": "10.1.0.1:7001"
}
```

**Response（失败）**:

```json
{"success": false, "code": 1001, "error_message": "..."}
```

**理由**：
- caller 起步零认知成本，Level 0 单字段 `batch_count` 即可
- Level 1+ 字段 phase 1 就预留进 wire，phase 2 启用 TTFT 时 caller 端无需配合升级
- 不引入 `protocol` 选择字段：response 同时返 `http_port` + `grpc_port`，caller 按需挑
- 不 echo `request_id`：caller 按下标对齐 `server_status[i]` ↔ `sub_requests[i]`
- 响应字段命名沿用 master 现有 `Response` DTO 风格，降低多接口认知负担

### D2: Role 由 master 推断，caller 不传

**理由**：
- role 信息部署侧 `MODEL_SERVICE_CONFIG` 已显式声明
- caller 再传一遍会和部署声明产生**双源**风险
- 本接口 scope 限定单 role 部署，**推断在合法 scope 下天然无歧义**

**实现**（已实现，不动）：

```java
List<RoleType> roleTypes = MODEL_ROLE_WORKER_STATUS.getRoleTypeList();
if (roleTypes.isEmpty()) return NO_AVAILABLE_WORKER;
if (roleTypes.size() > 1) return INVALID_REQUEST;  // 多 role 不支持
RoleType roleType = roleTypes.get(0);
```

### D3: 端口返双 port，caller 自选

不加 `protocol: HTTP|ARPC` 之类的 enum 字段，response 同时返 `http_port` + `grpc_port`。已实现，不动。

### D4: Phase 1 只支持 ROUND_ROBIN，Phase 2 加 SHORTEST_TTFT

**Phase 1**：唯一支持的策略是 RR，因为：
- RR 已实现（commit 4548c8923）
- RR 行为简单，本次只优化原子 op 数（D8）
- 其他策略（SHORTEST_TTFT / WEIGHTED_CACHE / RANDOM）调用 `/batch_schedule` → DefaultRouter 返 `"strategy for role X does not support batch_schedule"`

**Phase 2**：加 SHORTEST_TTFT。方案见 D5。

**永久不引入**：LOAD_AWARE_RR（SHORTEST_TTFT 已等价）、RANDOM_BATCH（统计上等价 RR）、WEIGHTED_CACHE_BATCH（cache 复用率低，退化为 random）。

### D5: TTFT batch 实现方案（Phase 2，本节是大纲）

**核心一句话**：`selectBatch` = `for (req : subs) { select(ctx, role, group); collect targets; }`，**N 次单选循环**，复用 `select()` 完整机制。

**为什么这么做**：
- 单选已有 race-free 机制：`putLocalTask` 内部先 `localTaskMap.put` 后 `addRunningQueueTime`，下一轮 `scoreWorkers` 自动看到刚选 worker 的 queue 升高 → batch 内分流自动有
- 心跳 `updateRunningQueueTime` 重算依据 `localTaskMap`，batch 写真账本 → 心跳行为完全等同于单选场景（无需特殊"虚拟账"机制）
- `getAvailableWorkers` 自带 `ResourceMeasure` 过滤 → batch 跟单选同样跳过过载 worker
- CAS `lastSelectedTime` fairness 跟单选完全等价

**Phase 2 接口签名变更**：

```java
public interface BatchLoadBalancer extends LoadBalancer {
    // Phase 1（保持）：RR 用
    List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group);

    // Phase 2 新增：TTFT 用，可选实现，默认 throw UnsupportedOperationException
    default List<BatchScheduleTarget> selectBatch(List<Request> subs, RoleType roleType, String group) {
        throw new UnsupportedOperationException();
    }
}
```

DefaultRouter 在 phase 2 根据策略类型分发到对应签名，不影响 phase 1 RR 路径。

**性能账**：
- 复杂度 O(N × W log W)，每次 select 约 10-50μs
- N=50, W=10：~1ms master CPU
- N=1000, W=100：~70ms（极端值，BATCH_SCHEDULE_MAX_COUNT 上限可调小）
- 优化空间：把 `getAvailableWorkers` hoist 出 loop，省 30%。**不在 phase 2 起步做**，profiler 数据指向再加。

**Partial failure 处理**（phase 2 决策项）：N 次 select 跑到第 i 次失败时，**rollback 前 i-1 个 worker 的 `localTaskMap` entry**，返空 list + 错误码。理由：TTFT 既然写了真账本，partial fail 显式清理是干净的，不留靠心跳擦的悬挂状态。

### D6: BatchLoadBalancer 接口签名（Phase 1 不动）

Phase 1 保持现有 `selectBatch(int count, RoleType, String)`。Phase 2 时**通过 default 方法**新增 `selectBatch(List<Request>, ...)` 签名，不破坏 phase 1 RR 实现。

**理由**：
- Phase 1 RR 路径不需要 sub_requests，强行改签名是无意义的 churn
- 接口扩展通过 default method + 策略层 instanceof 分发，向后兼容

### D7: RR batch 性能优化（Phase 1 核心改动）

**当前 RR `selectBatch`**（commit 4548c8923）：

```java
for (int i = 0; i < count; i++) {
    WorkerStatus selected = alive.get(nextIndex(roleType, alive.size()));  // ← N 次原子 op
    targets.add(buildTarget(selected));
}

private int nextIndex(RoleType roleType, int size) {
    return Math.floorMod(cursor.getAndIncrement(), size);
}
```

**问题**：N 次 `cursor.getAndIncrement()`：
- 无竞争：~5ns/op，N=50 = 250ns
- 有竞争（并发 `/schedule` 在抢 cursor）：~25-100ns/op，cache line 在 CPU 间反复弹射，N=50 = 1.25-5μs，p99 更差

**优化版本**：

```java
@Override
public List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group) {
    List<WorkerStatus> alive = aliveWorkers(roleType, group);
    if (alive.isEmpty()) return new ArrayList<>();

    int aliveSize = alive.size();
    // ★ 一次原子 op 占连续 N 个 cursor 号
    int start = cursors.get(roleType).getAndAdd(count);

    List<BatchScheduleTarget> targets = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
        int idx = Math.floorMod(start + i, aliveSize);  // 纯本地算术，无内存屏障
        targets.add(buildTarget(alive.get(idx)));
    }
    return targets;
}
```

**收益**：

| 维度 | 当前 | 优化后 | 收益 |
|---|---|---|---|
| 原子 op 次数 | N（50）| 1 | -98% |
| Cache line 弹射 | N 次 | 1 次 | -98% |
| 与并发 `/schedule` 的竞争点 | N 次抢 cursor | 1 次抢 | 降低尾延迟 |
| 纯 CPU（无竞争）| ~10μs | ~6μs | -40% |
| 纯 CPU（高 QPS 竞争）| 50-500μs | ~10μs | -90%+ |
| 算法复杂度 | O(N) | O(N) | 不变 |
| Wrap-around 语义 | floorMod 兜底 | 同 | 兼容 |

**整数溢出**：`getAndAdd(count)` 接近 `Integer.MAX_VALUE` 时 `start` 会溢出到负数，`start + i` 可能继续溢出，但 `Math.floorMod(负数, size)` 按 Java spec 返正确正数。**与现有 `getAndIncrement` 同样的兜底，无需特殊处理**。

**不做的优化**（明确放弃，避免雕花）：

| 候选 | 收益 | 决策 |
|---|---|---|
| 缓存 `BatchScheduleTarget` 实例（worker 属性不变就复用）| 微 | ❌ 不做 |
| 缓存 `aliveWorkers` 列表，心跳时刷新 | 小 | ❌ 不做 |
| `ArrayList` 换数组 | 微 | ❌ 不做 |
| 跳过 `Math.floorMod`（count<=size 时直接 subList）| 微（mod 1ns）| ❌ 不做 |
| 用 `LongAdder` 替 `AtomicInteger` | 负收益（cursor 是写多读多）| ❌ 不做 |

**RR 的核心价值是简单**（CLAUDE.md 注明 "50-200x cheaper than SHORTEST_TTFT"），保持简单本身是设计目标，0.1% CPU 优化不值得换复杂度。

### D8: 错误码 / 错误信息

| 场景 | code | error_message | Phase |
|---|---|---|---|
| `batch_count` 缺失 / 0 / 负 / 超上限 | INVALID_REQUEST (1001) | `batch_count must be in [1, N]` | P1（已实现）|
| 提供 `sub_requests` 但 `size != batch_count` | INVALID_REQUEST (1001) | `sub_requests length M != batch_count N` | **P1 新增** |
| master 未就绪 | NO_AVAILABLE_WORKER (1002) | `master not ready or MODEL_SERVICE_CONFIG missing` | P1（已实现）|
| 多 role 部署 | INVALID_REQUEST (1001) | 引导走 `/schedule` | P1（已实现）|
| 该 role 无 alive worker | role-specific（`NO_PDFUSION_WORKER` 等）| role 标准信息 | P1（已实现）|
| 该 role 配置的策略不实现 `BatchLoadBalancer` | INVALID_REQUEST (1001) | `strategy for role X does not support batch_schedule` | P1（已实现）|
| Master 不可达（slave 收到时）| NO_AVAILABLE_WORKER (1002) | slave 直接返 500，**不 fallback** | P1（已实现）|
| TTFT 模式但缺 `sub_requests` / 缺 `request_id` / 重复 `request_id` | INVALID_REQUEST (1001) | TTFT 专属错误信息 | **P2** |

**Slave 不 fallback 的理由**：master 维护全局 RR cursor / queueTime 状态，slave fallback 会造成账本分裂。caller 应客户端重试拿新 master。

## 5. Phase 1 实现盘点

| 组件 | 状态 | Phase 1 动作 |
|---|---|---|
| `BatchScheduleRequest` DTO | ✅ 已有 `batch_count` | **改**：加 optional `sub_requests: List<Request>` 字段 + `@JsonIgnoreProperties(ignoreUnknown = true)` |
| `BatchScheduleResponse` / `BatchScheduleTarget` DTO | ✅ 已实现 | 不动 |
| `BatchLoadBalancer` 接口签名 | ✅ `selectBatch(int count, ...)` | 不动 |
| `RoundRobinLoadBalancer.selectBatch` | ✅ 已实现 | **改**：N 次 `getAndIncrement` → 1 次 `getAndAdd(count)` + 本地 floorMod 循环（D7）|
| `DefaultRouter.batchSchedule` | ✅ 已实现 | **改**：加 `sub_requests` 长度校验（若提供）|
| `HttpLoadBalanceServer` 路由 + slave→master 转发 | ✅ 已实现 | 不动 |
| RR batch 测试（单元 + 并发 cursor 测）| ✅ 已实现 9 case | **加 1 case**：sub_requests 提供时长度校验、不提供时不影响 RR；**加 1 case**：高竞争下 batch + 单调用 cursor 不重叠 |
| `docs/master-batch-schedule-api.md` | ⏳ 旧版本 | **重写**：Phase 1 用法（Level 0 RR）+ 注明 Level 1+ 字段 phase 2 启用 |

**Phase 1 不动的**：`ShortestTTFTStrategy`、`BatchBalanceContext`（不创建）、所有 TTFT batch 相关代码、`BATCH_PREFILL_INCREMENT_MS` env（不引入）。

**Phase 1 代码改动量估算**：~30 LOC 主代码 + ~80 LOC 测试。

## 6. 局限 / 已知不足

### Phase 1 局限
1. **TTFT 不支持 batch**：caller 调 `/batch_schedule` 但 role 配 SHORTEST_TTFT → 返 INVALID_REQUEST。需要分流的 caller 等 phase 2，或暂走 RR
2. **Wire 字段 `sub_requests` 接收但不读**：phase 1 caller 传了也是空跑。文档需明确"phase 1 仅 batch_count 生效"
3. **不支持多阶段批量**：解耦 PD 集群完整推理 → caller 走 `/schedule` 并发
4. **RR cursor 共享**：单调用 `select()` 和 `selectBatch()` 共享 cursor，并发会交错推进。预期但需 caller 知晓

### Phase 2 必决决策（本文档不做，留 phase 2 设计起点）
1. **Partial failure 行为**：N 次 select 中途失败 → rollback 已选 / 返 partial / 不管？建议 rollback（D5 已说理由）
2. **重复 `request_id` 校验**：DefaultRouter 加 `Set` 去重，5 行。否则 caller 隐式 bug 会污染 `localTaskMap`
3. **`WorkerStatus.putLocalTask` 顺序约定**（先 put 后 add）是 batch 安全性的隐式依赖。phase 2 落地时给该方法加注释固化约定

### Phase 1 不做但需要 caller 知晓
- **caller 撒谎/不发请求的容错**：RR 不入 `localTaskMap` → 没有清账负担，caller 拿了不用也不影响 master。Phase 2 TTFT 落地时这个保证不再成立（写真账本，依赖 worker 心跳的 `markTasksAsLost` 自愈，~20-40ms）

## 7. 未来演进路径（非破坏性）

| 触发场景 | 扩展方式 | Phase |
|---|---|---|
| 让 TTFT 也支持 batch | `ShortestTTFTStrategy implements BatchLoadBalancer` 的 List<Request> 重载，内部 `for (req : subs) { select(...); }` | **P2** |
| caller 想 atomic rollback | TTFT 路径在 partial failure 时 rollback 已选 worker（D5 已规划）| P2 |
| caller 想 cache locality | TTFT 路径读 `sub_requests[].block_cache_keys`，调 `cacheAwareService.findMatchingEngines`。**前提**是 caller 和 worker tokenizer / blockSize / hash bit-exact 对齐 | P3+ |
| 多阶段批量（解耦 PD 完整推理）| response shape 升级为嵌套，**独立 PR**，本接口不动 | 远期 |
| 新策略接入 batch | 该策略 `implements BatchLoadBalancer`，注册到 `LoadBalanceStrategyEnum` | 按需 |

## 8. 决策点

### Phase 1（本次确认）
- [x] D1 Wire 形状：`sub_requests` 加进 DTO 但策略层不读
- [x] D2 Role 推断（已实现）
- [x] D3 端口双 port（已实现）
- [x] D4 Phase 1 仅 RR
- [x] D6 接口签名 phase 1 不动
- [x] D7 RR `getAndAdd(count)` 优化
- [x] D8 错误码（phase 1 加 sub_requests 长度校验）

### Phase 2（推迟决策）
- [ ] D5 TTFT batch = N 次 select 循环，partial failure 时 rollback
- [ ] D8 P2 部分：TTFT 缺字段 / 重复 ID 校验

## 9. References

- 现有 commit：`4548c8923 feat(flexlb): add /batch_schedule endpoint with RoundRobin strategy`
- API 文档：`docs/master-batch-schedule-api.md`
- 早期讨论：`docs/dispatcher-batch-schedule-design-2026-05-08.md`
- 对比报告：`docs/dispatcher-batch-schedule-comparison-report-2026-05-09.md`
- 相关代码：
  - `rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DefaultRouter.java`
  - `rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/strategy/RoundRobinLoadBalancer.java`
  - `rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/strategy/ShortestTTFTStrategy.java`（phase 2 扩展）
  - `rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/dao/loadbalance/BatchScheduleRequest.java`
  - `rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/dao/loadbalance/BatchScheduleResponse.java`
  - `rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/dao/loadbalance/BatchScheduleTarget.java`
  - `rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/strategy/BatchLoadBalancer.java`
- Audit 历史（2026-05-11 conversation）：识别 D5 原方案的 7 个问题（虚拟账机制描述错、IdentityHashMap snapshot 没解决真 race、`BatchBalanceContext` 不存在、`sub_requests` 5 字段只读 1 等），并通过"复用 `select()` N 次循环 + RR 用 `getAndAdd` 合并"两个决定全部消除
