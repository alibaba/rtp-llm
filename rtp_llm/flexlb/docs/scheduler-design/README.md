# FlexLB 调度重构文档

本轮实施依据是[目标类图、职责、成员与代码预算](proposed-component-classes.html)。它是用户确认的完整目标；不能用局部接口迁移替代整体改造，也不能将尚未完成的条目标为已落地。

当前实现阅读入口：[当前核心类图](../current-class-diagram.md)（2026-11-03 按工作区源码核对）。[组件边界改造设计](../component-boundary-repair-design.md)保留早期边界讨论；与本轮目标不一致时，以本轮目标类图为准。

以下文档按各自历史基线保留，其中的“当前”“本轮”“已落地”指原文版本，不代表当前工作区实现。

2026-10-13 投递与执行收敛的历史类图、成员和 API 见 [DeliveryDispatcher 收敛说明](delivery-dispatcher-convergence.md)。

历史开发入口：[请求调度与交付责任：类图、成员、API 和实现约束](target-class-api-review.md)（入口原标注 2026-10-03，正文更新 2026-10-08）。该轮目标是删除 RequestLifecycle，保留四方法 RequestScheduler 和 Direct/Queued 实现，将身份索引、单请求裁决、交付清理及资源账本划清边界，并补齐“确定不会 fetch 必须及时取消 Prefill”的协议。具体实施状态以固定提交源码为准。

[此前的设计改进评审稿](round-design-review.md)保留作为历史背景。其 PlacementStrategy、DirectPlacementCoordinator、GlobalQueueCoordinator 和不使用抽象基类的建议已经被后续讨论替代，不作为当前请求调度的开发依据。

整体设计先读[架构边界与设计约束](architecture-contract.md)：这是待落实的系统级提案，区分接入、应用编排、业务内核、技术适配及装配，规定依赖和所有权。类级交接不能替代这些约束。

历史类与接口评审见[早期目标类图和 API](target-class-api-review.md)，其中成员和协议不作为本轮实施目标。

此前请求内部迁移见[FlexLB 请求职责重构开发交接](request-ownership-handoff.md)。它基于 2026 年 10 月 2 日工作区，记录历史字段迁移和行为约束；当前迁移去向、实现批次和新增取消协议以当前开发入口为准。

历史稿曾保留 `BalanceContext` 和七阶段状态机；最新远端已是 `RequestContext` / `ContextResult` 与 OPEN / ENDING / CLOSED，不应按旧稿恢复。[此前的整体目标方案](final-design.md)保留供历史背景查阅，不能直接据此恢复已删除对象。

历史评审附录：

- [类时序与状态转移](sequence-flows.md)：按类展开五个关键交接，图描述目标职责。
- [复审前的 15 张线程链路图](thread-model-detailed-flows.md)：原图及文字完整保留；首页标注已否决的旧边，不能直接据此实施。
- [枚举现状审查快照](appendix-enum-inventory.md)：旧源码基线上的枚举分类与所有者清单，仅作迁移核对；目标取舍以最终方案为准。

局部实施与性能记录归档在 [evidence](evidence/)；它们记录已运行的验证及未通过项，不是新设计版本。
