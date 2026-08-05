# Resource Management

FlexLB 依据后端资源水位动态调整**路由并发许可数**，把请求"卡"在调度层而不是打爆引擎；
同时在策略层按角色维度过滤资源不可用的 worker。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/resource/`。

## DynamicWorkerManager

- `@PostConstruct` 启动单线程 `ScheduledExecutorService`（daemon，线程名
  `worker-capacity-scheduler`），`scheduleWithFixedDelay` 每 `resourceCheckIntervalMs`
  （默认 **10ms**）执行一次 `recalculateWorkerCapacity()`。`@PreDestroy` 等待 5s 后强停。
- 初始容量：`maxTotalWorkers = scheduleWorkerSize`（默认 CPU 核数），`ReducibleSemaphore`、
  `totalPermits`、`allowedWorkers` 都初始化为该值。

### 每 tick 的重算流程

1. 遍历 `EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleTypeList()`（非空角色）；
2. 每个角色：按 `FlexlbConfig.getResourceMeasureIndicator(role)` 取 `ResourceMeasure`，
   计算该角色所有 worker 的**平均水位**（算术平均，注意：不过滤 dead worker）；
3. 取所有角色的**最大水位** `maxWaterLevel`；
4. `allowedWorkers = ⌊maxTotalWorkers × (1 − maxWaterLevel/100)⌋`，下限 0、上限
   `maxTotalWorkers`；
5. `adjustPermitCapacity(desired)`（synchronized）：`desired > totalPermits` → `release(1)`；
   `<` → `reducePermits(1)`——**每 tick 只动 1 个许可**（`ADJUSTMENT_STEP=1`），步进收敛避免震荡。

调试日志限流：水位 debug 日志每 **10s** 一条（`WATER_LEVEL_LOG_INTERVAL_MS=10000`），
`allowedWorkers` 变化时立即打印。

## 水位公式

### PrefillResourceMeasure（指标 `WAIT_TIME`）

实际度量是**有效待处理任务数**：取引擎上报的 `waitingTaskList.size()` 与 FlexLB
即时维护的 `localTaskMap.size()` 两者中的较大值。两份数据存在重叠，因此不相加；本地计数用于
弥补引擎 step 执行期间 WorkerStatus 快照更新不及时的问题。

- 单 worker 水位：`queueSize ≤ 0` → 0；`≥ prefillQueueSizeThreshold(3)` → 100；
  否则 `queueSize × 100 / prefillQueueSizeThreshold`。
- `isResourceAvailable()`（策略层候选过滤用）：alive 且队列长度对
  `prefillQueueSizeThreshold(3)` 做即时判断：`queueSize < 3` 可用，`queueSize ≥ 3` 不可用。

### DecodeResourceMeasure（指标 `REMAINING_KV_CACHE`）

度量是 KV cache 使用率 `usedPct = used × 100 / (used + available)`：

- 单 worker 水位：`usedPct ≤ decodeFullSpeedThreshold(40)` → 0；
  `≥ decodeStopThreshold(80)` → 100；之间线性插值。
- `isResourceAvailable()`：alive 且使用率对 `decodeAvailableMemoryThreshold(90)` 做滞回判断。

### 滞回（防抖）

Decode 使用 `WorkerStatus.updateResourceAvailabilityWithHysteresis`（flexlb-common）：
`lower = upper − upper × hysteresisBiasPercent(15)/100`；指标 ≥ upper → 不可用；
≤ lower → 恢复可用；带内保持原状态。用 `AtomicBoolean` CAS 切换，避免边界震荡。

## ReducibleSemaphore 与许可门控

`ReducibleSemaphore` 继承 JDK `Semaphore`，仅把 protected `reducePermits(int)` 暴露为
public。按 JDK 语义，reduce 可使可用许可为负——在途请求释放后才能再获取。

**唯一的 acquire/release 点**：`RequestScheduler.workerLoop()`。每个路由工作线程先
`tryAcquirePermit(500ms)` 再出队路由，finally 释放。许可耗尽时：工作线程在 500ms 超时
获取上自旋，请求滞留队列，**该层不产生拒绝**——吞吐降到 0 直到许可恢复。

## ResourceMeasureFactory

Spring 注入所有 `ResourceMeasure` bean（Prefill / Decode 两个实现），按各自
`getResourceMeasureIndicator()` 组成 `EnumMap<ResourceMeasureIndicatorEnum, ResourceMeasure>`。
角色→指标映射硬编码在 `FlexlbConfig.getResourceMeasureIndicator()`：
PDFUSION/PREFILL/VIT → `WAIT_TIME`，DECODE → `REMAINING_KV_CACHE`。

## 观测

`ResourceMonitorReporter`：gauge `app.worker.permit.capacity`，每 1s 上报
`dynamicWorkerManager.getTotalPermits()`。

## 相关配置（默认值）

`resourceCheckIntervalMs=10`、`scheduleWorkerSize=CPU核数`、`prefillQueueSizeThreshold=3`、
`decodeFullSpeedThreshold=40`、`decodeStopThreshold=80`、
`decodeAvailableMemoryThreshold=90`、`hysteresisBiasPercent=15`。
