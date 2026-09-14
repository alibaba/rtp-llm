# Spec: FlexLB Safe Config Hot Update

## Goal
- 让安全的调度数值与路由策略参数在配置更新后作用于已有 runtime。
- 阻止 scheduler、dispatcher、cache、consistency 等结构性配置产生新旧语义混用。

## Done Contract
- ConfigService 拒绝 restart-required 字段的运行时变化并保留 last-known-good。
- 现有 Prefill WorkerBatcher 在 listener 通知后使用新的窗口、队列和 fallback capacity。
- 针对性测试、全量测试和全仓 Spotless 通过。

## Scope
- In: scheduler 数值、请求级 router 参数、logging 等安全更新。
- In: scheduler/ordering/decision 类型、完整 dispatcher 和构造期组件配置的更新保护。
- Out: 在线重建 scheduler、delivery、KVCM、ZooKeeper、Optimizer 或 worker-sync executor。

## Facts / Constraints
- ConfigService 通过 AtomicReference 发布不可变配置快照，并同步调用 update listeners。
- WorkerBatcher 的 decision 与 queue 参数需要按单次计算读取同一份当前配置快照。
- Endpoint 创建可与配置更新并发；因此参数读取以 ConfigService 当前快照为准，listener 负责唤醒和 cache invalidation。

## Restated Understanding
- 当前任务是参考 FlexlbLogManager listener 模式补齐安全 hot update。
- 结构性变化不做部分应用，必须拒绝并要求重启。
- 不修改 Python/C++ 数据面，也不处理其它 code-review finding。

## Checkpoint Summary
- 当前进度：完成配置消费路径梳理，准备先写失败测试。
- 下一步 1：测试 restart-required 更新被拒绝。
- 下一步 2：测试 listener 使已有 WorkerBatcher 使用新决策参数。
- 下一步 3：实现最小 guard 和 endpoint listener，执行验证。
- 风险：更新与 endpoint publication 并发、旧请求需保持自己的 request snapshot。
- 验证方式：ConfigService/WorkerBatcher/EndpointRegistry 定向测试、全量测试、Spotless。
- Execution Approval: Approved

## Change Log
- 2026-09-09: 用户确认采用“安全参数热更新，结构性变化拒绝”的边界。
- 2026-09-09: 增加 restart-required guard、endpoint listener、配置版本 fencing 和 listener 注销。

## Validation
- Self-check: 最终代码审查未发现未处理的实现问题。
- Static checks: `./mvnw spotless:check -Pspotless-check` 通过。
- Runtime / Test: `./mvnw test` 全 reactor 通过；定向 hot-update 测试通过。
- 结果汇总：Done Contract 已满足。
- 核心目标是否已由证据证明完成：Yes
- 剩余风险：需要重启的配置更新当前以错误日志拒绝，没有独立 rejection metric。

## Resume / Handoff
- 当前状态：Completed
- 当前卡点：None
- 下一步唯一动作：如需提交，检查并仅暂存本任务文件。

## Project Sync Candidates
- 是否发现可复用项目事实：Yes
- 候选事实：FLEXLB_CONFIG 需要区分 hot-update 与 restart-required 字段。
- 同步状态：Not synced
