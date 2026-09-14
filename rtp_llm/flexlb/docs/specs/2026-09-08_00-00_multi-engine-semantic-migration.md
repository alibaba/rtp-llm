# Spec: Multi-engine semantic migration

## Goal

- 将 `feature/multi-engine-support` 与其后继 `feat/multi-engine-kv-event-subscriber` 中的 multi-engine 行为移植到当前 `feature/flexlb-mu-dev-multi-engine`（`adc44e8b01`）。
- 最终目标：在不回退当前 PriorityScheduler、异步生命周期及现有功能的前提下，完整保留 logical multi-engine worker 的生产语义。
- 本轮核心目标：先将较新分支的 multi-engine 实现和回归修复按当前架构重放，再以旧分支契约逐项补齐、测试和审查。
- 验收结果：生产逻辑使用 logical identity `ip:port@index`、网络仍使用 `ip:port`；`flexlb-mock-engine` 测试请求与配置维持 `ip:port` 且不新增 `@index` 测试，其他模块的 mock HTTP seam 可用参数化 helper 同时接受两种表示。

## Done Contract

- 完成：multi-engine 的状态扩展、路由、缓存/资源隔离、物理组健康门控和 `engine_index` 传输在当前架构中均有实现及行为测试。
- 证明：目标 Maven 测试、mock-engine 兼容测试、全 reactor Spotless 及 diff 检查通过；源码与旧/新分支的契约矩阵逐项审查无遗漏。
- 未完成：任一关键路径仍仅按物理地址识别 worker，或 `flexlb-mock-engine` 需把 mock endpoint 写成 `ip:port@index` 才能通过。

## Scope

- In：两个来源分支中的 multi-engine 逻辑及必要回归修复；最小的测试、架构文档同步。
- Out：移植两条来源分支中无关的旧调度器、在线评测工具、RPC/UniConfig 基础设施改造；推送或改写远端分支。
- 用户已切分的任务单元：当前 FlexLB 分支上的 multi-engine 语义迁移与 mock-engine RTP-LLM 兼容验证。
- 轻量评估：需要升级为 `deep`；原因是跨模块、历史分支已经与当前调度架构语义分叉。

## Facts / Constraints

- 当前工作树干净，当前分支与 `feature/flexlb-mu-dev` 同指 `adc44e8b01`。
- `feature/multi-engine-support` 位于 `511d6a120b`；其旧逻辑从 `8a0077526f` 起，后续包含 identity 和单引擎 KVCM 兼容修复。
- `feat/multi-engine-kv-event-subscriber` 位于 `57972d1749`；其最后两笔为 `b0081ee49b`（multi-engine）与 `57972d1749`（回归修复）。
- 两条来源分支均与当前分支在旧基础上分叉；直接整段 cherry-pick/rebase 会覆盖当前 PriorityScheduler 与异步运行时，不能作为实现策略。
- 生产 logical identity 是 `ip:port@index`，网络地址不携带 `@index`；N=1 内部仍为 index 0，但 wire `engine_index` 省略。
- 同物理 worker 的状态端口为 `worker_status_port + index`；资源、队列、缓存、评分按 index 隔离；物理组健康为所有 logical engines 的 AND。
- `flexlb-mock-engine` 是 RTP-LLM mock：其测试 fixture、HTTP 目标和配置均保持 `ip:port`，不写也不测试 `ip:port@index`；其它模块的 mock HTTP seam 可以参数化覆盖两种输入。

## Open Questions

- [ ] 根据实际冲突与现有 current-branch seam，确认需要补充的最小兼容测试集合。

## Restated Understanding

- 我理解当前任务是：把新、旧两代 multi-engine 的行为收敛到当前分支，而不是把旧架构整段覆盖回来。
- 当前核心目标是：较新实现优先，旧分支补足契约审计；生产使用 logical identity，`flexlb-mock-engine` 输入保持物理网络地址，其它 mock seam 兼容两种地址表示。
- 当前边界是：不迁移无关架构演进，不推送远端，不改变 mock engine 的地址协议。
- 暂不处理：来源分支的在线评测与其它非 multi-engine 功能。

## Goal Alignment Check

- 当前动作服务于核心目标：历史范围映射、契约提取和针对当前架构的语义 cherry-pick 准备。
- 当前路径在用户边界内：Yes。
- 更适合代码地形的路径：只移植 semantic delta，而非对旧分支做整段 rebase。

## Checkpoint Summary

- 当前任务理解：以新分支作为实现来源、旧分支作为行为基准，完成当前架构上的 multi-engine 迁移。
- 当前核心目标：保留 current 的调度/生命周期，并补齐所有 multi-engine 业务路径。
- 当前进度：工作树、分支拓扑和来源提交已经确认；正在抽取逐路径契约。
- 下一步 1：审计新旧变更和 current branch 的多引擎相关 seam。
- 下一步 2：在隔离 integration branch 上进行最小语义 cherry-pick 并处理冲突。
- 下一步 3：按公开状态/路由/mock HTTP seam 补测试和验证。
- 涉及文件 / 模块：`flexlb-common`、`flexlb-cache`、`flexlb-sync`、`flexlb-mock-engine`、架构文档。
- 风险：旧分支调度架构与当前分支不兼容；priority admission 与 decode eviction 可能绕开物理组健康门控。
- 验证方式：history/contract matrix、Maven focused tests、mock-engine integration tests、full-repository Spotless。
- Execution Approval: Approved (用户在本任务中明确请求 cherry-pick/rebase 与完整迁移)。

## Change Log

- 2026-09-08: 创建本 Feature Spec；确认采用 semantic migration，而非整段历史重放。
- 2026-09-08: 用户明确测试边界：不改 `flexlb-mock-engine`；其它模块的 mock HTTP 以参数化 helper 支持 physical/logical 两种地址。

## Validation

- Self-check: 尚未开始代码变更。
- Static checks: 当前工作树干净。
- Runtime / Test: 待执行。
- Human confirmation: 用户已明确了 mock endpoint 的地址兼容边界。
- 结果汇总：进行中。
- 核心目标是否已由证据证明完成：否。
- 若未完成，当前剩余差距：current branch 与两代 multi-engine 契约的逐路径差异尚未完成。
- 剩余风险：语义冲突需以 current scheduling invariant 为准。

## Resume / Handoff

- 当前状态：已完成来源拓扑与迁移策略确认，等待契约审计结果后进入隔离 integration branch。
- 当前卡点：无。
- 下一步唯一动作：从新旧来源提交提取 current branch 缺失的 multi-engine semantic delta。
- 下一轮核心目标：实现并验证第一条缺失 multi-engine 行为。

## Project Sync Candidates

- 是否发现可复用项目事实：Yes。
- 候选事实：multi-engine migration 不能整段重放旧 scheduler，必须保留 current PriorityScheduler 和异步生命周期，并按 logical-worker 契约逐路径移植。
- 建议同步位置：本次先保留 Feature Spec；任务完成后再评估是否更新项目架构文档。
- 同步状态：Not synced。
